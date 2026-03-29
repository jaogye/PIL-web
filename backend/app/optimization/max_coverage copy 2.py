"""
Capacitated Maximum Coverage Location Problem (CMCLP).

Two-phase heuristic:

Phase 0 – Pre-existing centres:
    Absorb demand from existing facilities using nearest-first assignment
    up to their capacity (preserves legacy behaviour for existing infra).

Phase 1 – Greedy Constructive:
    Candidate scoring uses marginal_coverage (O(nnz), vectorised).
    Assignment for the chosen candidate is demand-descending (first-fit-
    decreasing): areas with the highest demand within the service radius are
    served first, up to cap_max.  A facility is opened only when its actual
    filled load >= cap_min.  Repeats until no valid candidate remains.

Phase 2 – Local Search (facility swap):
    For each currently-open new facility, try replacing it with up to
    MAX_SWAP_CANDIDATES closed alternatives.  Each trial runs a full greedy
    re-assignment (demand-descending) starting from the post-Phase-0 state.
    A swap is accepted when the total demand covered by new facilities
    strictly increases and every facility in the new set meets cap_min.
    Iterates for up to max_iter_local rounds without improvement.
"""

from __future__ import annotations

import logging
import math
import numpy as np
from dataclasses import dataclass, field

from .sparse_matrix import MAX_DIST, SparseDistanceMatrix

logger = logging.getLogger(__name__)

MAX_SWAP_CANDIDATES = 10   # closed facilities evaluated per open facility per swap round


@dataclass
class MaxCoverageResult:
    facility_indices: list[int]
    assignment: list[int]
    covered_demand: float
    total_demand: float
    coverage_pct: float
    coverage_stats: dict = field(default_factory=dict)


def solve(
    distance_matrix: SparseDistanceMatrix,
    demand: np.ndarray,
    p: int,                          # kept for API compatibility; ignored
    radius: float,
    cap_min: float = 0.0,
    cap_max: float | None = None,
    pre_selected: list[int] | None = None,
    max_iter_local: int = 50,
) -> MaxCoverageResult:
    """
    Solve the Capacitated MCLP via the two-phase greedy + local-search heuristic.

    Parameters
    ----------
    distance_matrix : SparseDistanceMatrix
    demand          : np.ndarray, shape (n,)
    p               : int — ignored; facilities are placed until cap_min
                      cannot be satisfied.
    radius          : float — service radius in minutes.
    cap_min         : float — minimum load a facility must attract to be placed.
    cap_max         : float | None — maximum load per facility (None = no limit).
    pre_selected    : list[int] | None — indices fixed as existing facilities.
    max_iter_local  : int — local-search rounds without improvement before stopping.
    """
    dm = distance_matrix
    n = dm.n
    r16 = np.uint16(min(int(radius), MAX_DIST))
    cap_max_f = float(cap_max) if cap_max is not None else math.inf
    cap_min_f = float(cap_min)
    total_demand = float(np.sum(demand))
    pre_selected = list(pre_selected or [])

    remaining = demand.astype(np.float64).copy()
    selected: list[int] = []
    selected_set: set[int] = set()
    fac_load: dict[int, float] = {}
    fac_remaining: dict[int, float] = {}
    assigned_to = np.full(n, -1, dtype=np.int32)
    fac_to_idx: dict[int, int] = {}

    # ── Phase 0: Pre-existing facilities (nearest-first) ─────────────────── #
    for fac_pos, fac in enumerate(pre_selected):
        selected.append(fac)
        selected_set.add(fac)
        fac_load[fac] = 0.0
        fac_remaining[fac] = cap_max_f
        fac_to_idx[fac] = fac_pos

    for fac_pos, fac in enumerate(pre_selected):
        _assign_zone_nearest_first(
            dm, remaining, fac, fac_load, fac_remaining,
            radius, r16, selected_set, assigned_to, fac_pos, fac_to_idx,
        )

    # Save state after Phase 0 (starting point for Phase-2 re-assignment).
    post0_remaining = remaining.copy()
    pre_covered_demand = total_demand - float(np.sum(post0_remaining))

    # ── Phase 1: Greedy Constructive ─────────────────────────────────────── #
    new_opened: set[int] = set()

    while True:
        # Vectorised scoring: sum of remaining demand within radius per candidate.
        scores = dm.marginal_coverage(remaining, radius)
        effective = np.minimum(scores, cap_max_f) if cap_max is not None else scores
        if selected_set:
            effective[list(selected_set)] = -1.0

        best_j = int(np.argmax(effective))

        if effective[best_j] < cap_min_f - 1e-9:
            break   # No candidate can score enough demand.

        # Exact demand-first fill for the chosen candidate.
        load, taken = _zone_demand_first(dm, remaining, demand, best_j, r16, cap_max_f)

        if load < cap_min_f - 1e-9:
            break   # Exact fill is below cap_min (score was an overestimate).

        fac_pos = len(selected)
        fac_to_idx[best_j] = fac_pos
        selected.append(best_j)
        selected_set.add(best_j)
        new_opened.add(best_j)
        fac_load[best_j] = load
        fac_remaining[best_j] = max(0.0, cap_max_f - load)

        for i in taken:
            remaining[i] = 0.0
            assigned_to[i] = fac_pos

        logger.debug(
            "  CMCLP Phase 1: opened facility %d  load=%.1f  taken=%d areas",
            best_j, load, len(taken),
        )

    logger.info(
        "  CMCLP Phase 1 done: %d new facilities, covered demand=%.1f",
        len(new_opened),
        total_demand - float(np.sum(remaining)),
    )

    # ── Phase 2: Local Search (facility swap) ────────────────────────────── #
    if new_opened:
        covered_by_new = float(np.sum(post0_remaining)) - float(np.sum(remaining))
        new_opened, remaining, covered_by_new, fac_load = _local_search(
            dm, demand, post0_remaining, new_opened, selected_set,
            n, r16, cap_min_f, cap_max_f, max_iter_local,
            pre_selected, covered_by_new, fac_load,
        )

        # Rebuild selected / selected_set from final new_opened + pre_selected.
        selected = list(pre_selected) + sorted(new_opened)
        selected_set = set(selected)

        logger.info(
            "  CMCLP Phase 2 done: %d new facilities, covered demand=%.1f",
            len(new_opened),
            pre_covered_demand + covered_by_new,
        )

    # ── Remove facilities below cap_min (defensive) ──────────────────────── #
    valid = [f for f in selected if fac_load.get(f, 0.0) >= cap_min_f - 1e-9]
    if len(valid) < len(selected):
        logger.warning(
            "  CMCLP: dropped %d facilities below cap_min after placement",
            len(selected) - len(valid),
        )
    selected = valid

    # ── Coverage & assignment ─────────────────────────────────────────────── #
    covered = remaining < demand - 1e-9
    covered_dem = float(np.sum(demand[covered]))
    coverage_pct = (
        round(covered_dem / total_demand * 100, 2) if total_demand > 0 else 0.0
    )

    assignment = dm.assign(selected) if selected else [-1] * n

    return MaxCoverageResult(
        facility_indices=sorted(selected),
        assignment=assignment,
        covered_demand=covered_dem,
        total_demand=total_demand,
        coverage_pct=coverage_pct,
        coverage_stats=_compute_stats(
            demand, dm, selected, radius, covered, fac_load, cap_min_f, cap_max_f
        ),
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 1 helper: demand-first fill                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def _zone_demand_first(
    dm: SparseDistanceMatrix,
    remaining: np.ndarray,
    demand: np.ndarray,
    j: int,
    r16: np.uint16,
    cap_max_f: float,
) -> tuple[float, list[int]]:
    """
    First-fit-decreasing fill for facility j.

    Sorts the unassigned areas within j's zone by demand descending, then
    greedily takes areas until cap_max would be exceeded.  The first area
    that does not fit causes the fill to stop (same break semantics as the
    reference heuristic).

    Returns (total_load, list_of_assigned_area_indices).
    """
    rows, vals = dm.col_neighbors(j)
    in_zone = vals <= r16
    if not np.any(in_zone):
        return 0.0, []

    zone = rows[in_zone]

    # Collect (demand, area_index) for areas with remaining demand.
    candidates = sorted(
        ((float(remaining[i]), int(i)) for i in zone if remaining[i] > 1e-9),
        reverse=True,
    )

    load = 0.0
    taken: list[int] = []
    for dem, i in candidates:
        if load + dem > cap_max_f:
            break   # First-fit-decreasing: stop on first item that doesn't fit.
        load += dem
        taken.append(i)

    return load, taken


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 2: local search                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def _full_greedy_assign(
    dm: SparseDistanceMatrix,
    post0_remaining: np.ndarray,
    facilities: set[int],
    r16: np.uint16,
    cap_max_f: float,
) -> tuple[float, dict[int, float], np.ndarray]:
    """
    Greedy demand-first re-assignment for a given set of new facilities,
    starting from the post-Phase-0 demand state.

    Facilities are processed in sorted order (smallest index first) for
    deterministic results.  Each area can only be assigned once.

    Returns (total_covered_demand, fac_load_dict, updated_remaining_array).
    """
    remaining = post0_remaining.copy()
    fac_load: dict[int, float] = {}
    total_covered = 0.0

    for j in sorted(facilities):
        rows, vals = dm.col_neighbors(j)
        in_zone = vals <= r16
        if not np.any(in_zone):
            fac_load[j] = 0.0
            continue

        zone = rows[in_zone]
        candidates = sorted(
            ((float(remaining[i]), int(i)) for i in zone if remaining[i] > 1e-9),
            reverse=True,
        )

        load = 0.0
        for dem, i in candidates:
            if load + dem > cap_max_f:
                break
            load += dem
            remaining[i] = 0.0

        fac_load[j] = load
        total_covered += load

    return total_covered, fac_load, remaining


def _local_search(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    post0_remaining: np.ndarray,
    new_opened: set[int],
    selected_set: set[int],
    n: int,
    r16: np.uint16,
    cap_min_f: float,
    cap_max_f: float,
    max_iter_local: int,
    pre_selected: list[int],
    covered_by_new: float,
    fac_load: dict[int, float],
) -> tuple[set[int], np.ndarray, float, dict[int, float]]:
    """
    Iterative swap-based local search on the set of new (non-pre-selected)
    facilities.

    For each open facility j_open, tries swapping it for each of up to
    MAX_SWAP_CANDIDATES closed alternatives.  Accepts the swap if the total
    demand covered by new facilities improves and every facility in the new
    set meets cap_min.

    Returns updated (new_opened, remaining, covered_by_new, fac_load).
    """
    pre_set = set(pre_selected)
    # Current remaining after Phase 1 (reconstruct from known coverage).
    _, _, remaining = _full_greedy_assign(dm, post0_remaining, new_opened, r16, cap_max_f)

    for iteration in range(max_iter_local):
        improved = False

        # Closed candidates: not pre-selected, not currently open.
        closed = [j for j in range(n) if j not in selected_set and j not in pre_set]

        for j_open in list(new_opened):
            for j_closed in closed[:MAX_SWAP_CANDIDATES]:
                trial_set = (new_opened - {j_open}) | {j_closed}

                new_cov, new_loads, new_remaining = _full_greedy_assign(
                    dm, post0_remaining, trial_set, r16, cap_max_f,
                )

                # Reject if any facility in the trial set falls below cap_min.
                if any(new_loads.get(f, 0.0) < cap_min_f - 1e-9 for f in trial_set):
                    continue

                if new_cov > covered_by_new + 1e-6:
                    new_opened = trial_set
                    covered_by_new = new_cov
                    remaining = new_remaining
                    fac_load.update(new_loads)
                    # Remove the swapped-out facility from fac_load.
                    fac_load.pop(j_open, None)
                    # Update selected_set.
                    selected_set.discard(j_open)
                    selected_set.add(j_closed)
                    improved = True
                    logger.debug(
                        "  CMCLP Phase 2 iter %d: swapped %d → %d  covered=%.1f",
                        iteration, j_open, j_closed, covered_by_new,
                    )
                    break

            if improved:
                break

        if not improved:
            break

    return new_opened, remaining, covered_by_new, fac_load


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 0 helper: nearest-first assignment for pre-existing facilities         #
# ─────────────────────────────────────────────────────────────────────────── #

def _assign_zone_nearest_first(
    dm: SparseDistanceMatrix,
    remaining: np.ndarray,
    fac: int,
    fac_load: dict[int, float],
    fac_remaining: dict[int, float],
    radius: float,
    r16: np.uint16,
    selected_set: set[int],
    assigned_to: np.ndarray,
    fac_pos: int,
    fac_to_idx: dict[int, int],
) -> None:
    """
    Assign demand from fac's zone nearest-first up to its remaining capacity.
    Used only for Phase 0 (pre-existing facilities).
    """
    rows, vals = dm.col_neighbors(fac)
    in_zone = vals <= r16
    if not np.any(in_zone):
        return

    zone_rows = rows[in_zone]
    zone_vals = vals[in_zone]
    order = np.argsort(zone_vals, kind="stable")
    zone_rows = zone_rows[order]

    cap = fac_remaining[fac]
    avail = remaining[zone_rows]

    if cap == math.inf:
        for area in zone_rows:
            if remaining[area] > 1e-9 and assigned_to[area] == -1:
                assigned_to[area] = fac_pos
        remaining[zone_rows] -= avail
        fac_load[fac] += float(np.sum(avail))
        return

    if cap <= 1e-9:
        return

    cumsum = np.cumsum(avail)
    if cumsum[-1] <= cap:
        for area in zone_rows:
            if remaining[area] > 1e-9 and assigned_to[area] == -1:
                assigned_to[area] = fac_pos
        remaining[zone_rows] -= avail
        fac_load[fac] += float(cumsum[-1])
        fac_remaining[fac] -= float(cumsum[-1])
        return

    k = int(np.searchsorted(cumsum, cap, side="right"))
    take = np.zeros(len(avail), dtype=np.float64)
    if k > 0:
        take[:k] = avail[:k]
    partial = cap - (float(cumsum[k - 1]) if k > 0 else 0.0)
    if k < len(avail) and partial > 1e-9:
        take[k] = partial

    for j in range(min(k + 1, len(zone_rows))):
        if take[j] > 1e-9 and assigned_to[zone_rows[j]] == -1:
            assigned_to[zone_rows[j]] = fac_pos

    remaining[zone_rows] -= take
    fac_load[fac] += float(np.sum(take))
    fac_remaining[fac] = 0.0


# ─────────────────────────────────────────────────────────────────────────── #
# Statistics                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def _compute_stats(
    demand: np.ndarray,
    dm: SparseDistanceMatrix,
    facilities: list[int],
    radius: float,
    covered: np.ndarray,
    fac_load: dict[int, float],
    cap_min: float,
    cap_max: float,
) -> dict:
    total_demand = float(np.sum(demand))
    covered_demand = float(np.sum(demand[covered]))
    loads = [fac_load.get(f, 0.0) for f in facilities]

    return {
        "total_demand": round(total_demand, 2),
        "covered_demand": round(covered_demand, 2),
        "uncovered_demand": round(total_demand - covered_demand, 2),
        "coverage_pct": (
            round(covered_demand / total_demand * 100, 2) if total_demand > 0 else 0.0
        ),
        "service_radius_minutes": radius,
        "num_facilities": len(facilities),
        "num_covered_areas": int(np.sum(covered)),
        "num_uncovered_areas": int(np.sum(~covered)),
        "cap_min": cap_min,
        "cap_max": cap_max if cap_max != math.inf else None,
        "avg_facility_load": round(float(np.mean(loads)), 2) if loads else 0.0,
        "min_facility_load": round(min(loads), 2) if loads else 0.0,
        "max_facility_load": round(max(loads), 2) if loads else 0.0,
    }
