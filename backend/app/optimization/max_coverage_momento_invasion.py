"""
Capacitated Maximum Coverage Location Problem (CMCLP).

Phase 0 – Pre-existing centres:
    Absorb demand from existing facilities using nearest-first assignment
    up to their capacity.

Phase 1 – Greedy Constructive (momentum-based alternating):
    For every candidate area i, compute a momentum indicator:

        momentum[i] = Σ_j  remaining[j] · speed(i,j)

    where speed(i,j) = euclidean_km(i,j) / (travel_time_min(i,j) / 60).

    Facilities are placed in alternating turns:
      • Max turn : open the unselected candidate with the highest momentum.
      • Min turn : open the unselected candidate with the lowest momentum.

    Both turns use demand-descending (first-fit-decreasing) fill up to
    cap_max.  A candidate is considered only when its zone contains
    enough unassigned demand to meet cap_min.

    After each placement the momentum is updated incrementally in O(k)
    using the CSC structure — no full recomputation needed.

    The number of min turns per max turn is controlled by equity_ratio
    (default 1 → strict 1-max / 1-min alternation).

Phase 2A – Complementary placement (access-time minimisation):
    Iteratively places new facilities that reduce total weighted access
    time (TWAT = Σ demand_i × distance(i, nearest_facility_i)).
    A candidate may steal areas from OTHER new clusters if it is closer
    to those areas, as long as invaded clusters stay above cap_min.
    Stops when no placement reduces TWAT.

Phase 2B – Redundancy pruning:
    Iteratively removes new facilities whose served areas can all be
    reassigned to neighbouring open facilities within the service radius
    without any receiving facility exceeding cap_max.  Facilities are
    considered for removal in ascending load order (lightest first).
    Produces a more homogeneous spatial distribution.
"""

from __future__ import annotations

import logging
import math
import numpy as np
from dataclasses import dataclass, field

from .sparse_matrix import MAX_DIST, SparseDistanceMatrix

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_FAILURES = 10  # stop Phase 1 after this many consecutive failed fills


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
    equity_ratio: int = 1,
    xy: np.ndarray | None = None,    # (n, 2) float64 lon/lat coordinates
) -> MaxCoverageResult:
    """
    Solve the Capacitated MCLP via momentum-based greedy + local-search.

    Parameters
    ----------
    distance_matrix : SparseDistanceMatrix
    demand          : np.ndarray, shape (n,)
    p               : int — ignored.
    radius          : float — service radius in minutes.
    cap_min         : float — minimum load a facility must attract to open.
    cap_max         : float | None — maximum load per facility (None = no limit).
    pre_selected    : list[int] | None — indices fixed as existing facilities.
    equity_ratio    : int — min turns per max turn (1 = strict alternation).
    xy              : np.ndarray, shape (n, 2) — WGS-84 lon/lat per area.
                      Required for momentum.  Falls back to plain greedy if None.
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
            r16, selected_set, assigned_to, fac_pos, fac_to_idx,
        )

    # Save post-Phase-0 state for Phase-2 re-assignment.
    post0_remaining = remaining.copy()
    pre_covered_demand = total_demand - float(np.sum(post0_remaining))

    # ── Phase 1: Greedy Constructive ──────────────────────────────────────── #
    new_opened: set[int] = set()

    use_momentum = xy is not None and xy.shape == (n, 2)

    if use_momentum:
        csr_speeds, csc_speeds = _precompute_pair_speeds(dm, xy)
        momentum = _compute_momentum(dm, remaining, csr_speeds)
        new_opened, remaining, fac_load, assigned_to, fac_to_idx, selected, selected_set = \
            _phase1_momentum(
                dm, demand, remaining, momentum, csc_speeds,
                selected, selected_set, new_opened,
                fac_load, fac_remaining, fac_to_idx, assigned_to,
                r16, cap_min_f, cap_max_f, equity_ratio,
            )
    else:
        new_opened, remaining, fac_load, assigned_to, fac_to_idx, selected, selected_set = \
            _phase1_greedy(
                dm, demand, remaining,
                selected, selected_set, new_opened,
                fac_load, fac_remaining, fac_to_idx, assigned_to,
                r16, cap_min_f, cap_max_f,
            )

    logger.info(
        "  CMCLP Phase 1 done: %d new facilities, covered demand=%.1f",
        len(new_opened),
        total_demand - float(np.sum(remaining)),
    )

    # ── Phase 2A: Complementary placement minimising weighted access time ──── #
    n_pre = len(pre_selected)
    if new_opened and cap_min_f > 0:
        _phase2a_access_time(
            dm, demand, remaining, selected, selected_set,
            fac_load, fac_remaining, fac_to_idx, assigned_to,
            n_pre, radius, cap_min_f, cap_max_f,
        )
        logger.info(
            "  CMCLP Phase 2A done: %d total new facilities",
            len(selected) - n_pre,
        )

    # ── Phase 2B: Prune redundant facilities ──────────────────────────────── #
    if len(selected) > n_pre:
        _phase2b_prune(
            dm, demand, remaining, selected, selected_set,
            fac_load, fac_remaining, fac_to_idx, assigned_to,
            n_pre, radius, cap_min_f, cap_max_f,
        )
        new_opened = set(selected[n_pre:])
        logger.info(
            "  CMCLP Phase 2B done: %d new facilities after pruning",
            len(new_opened),
        )

    # ── Remove facilities below cap_min (defensive) ───────────────────────── #
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
# Momentum helpers                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def _euclidean_speeds(
    xi: np.ndarray, yi: np.ndarray,
    xj: np.ndarray, yj: np.ndarray,
    tt_min: np.ndarray,
) -> np.ndarray:
    """speed(i,j) = euclidean_km(i,j) / (tt_min(i,j) / 60)  for paired arrays."""
    lat_mid = np.radians((yi + yj) / 2.0)
    dx_km = (xj - xi) * 111.0 * np.cos(lat_mid)
    dy_km = (yj - yi) * 111.0
    d_km = np.sqrt(dx_km ** 2 + dy_km ** 2)
    tt_h = tt_min / 60.0
    return np.where(tt_h > 0.0, d_km / tt_h, 0.0)


def _precompute_pair_speeds(
    dm: SparseDistanceMatrix,
    xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute speed(i,j) for every stored pair, in both CSR and CSC order.

    CSR speeds: indexed same as dm.csr_val  — area _csr_i[k] → neighbor csr_col[k].
    CSC speeds: indexed same as dm.csc_val  — area csc_row[k] → facility _csc_j[k].

    Both represent the same physical quantity (speed from area to facility),
    just in different traversal orders needed for momentum init vs incremental update.
    """
    csr_speeds = _euclidean_speeds(
        xy[dm._csr_i, 0], xy[dm._csr_i, 1],
        xy[dm.csr_col, 0], xy[dm.csr_col, 1],
        dm.csr_val.astype(np.float64),
    )
    csc_speeds = _euclidean_speeds(
        xy[dm.csc_row, 0], xy[dm.csc_row, 1],
        xy[dm._csc_j, 0], xy[dm._csc_j, 1],
        dm.csc_val.astype(np.float64),
    )
    return csr_speeds, csc_speeds


def _compute_momentum(
    dm: SparseDistanceMatrix,
    remaining: np.ndarray,
    csr_speeds: np.ndarray,
) -> np.ndarray:
    """
    Vectorised momentum computation (O(nnz)):

        momentum[i] = Σ_j  remaining[j] · speed(i,j)

    Uses CSR structure: for each area i, sums speed(i,j) * remaining[j]
    over all stored neighbors j.
    """
    weighted = remaining[dm.csr_col] * csr_speeds   # (nnz,)
    return np.bincount(
        dm._csr_i.astype(np.intp), weights=weighted, minlength=dm.n
    )


def _update_momentum(
    dm: SparseDistanceMatrix,
    momentum: np.ndarray,
    csc_speeds: np.ndarray,
    assigned_area: int,
    delta: float,
) -> None:
    """
    Incremental momentum update (O(k)) when assigned_area loses `delta` demand.

    For every area i' that has assigned_area as a stored neighbor, subtract
    delta * speed(i', assigned_area) from momentum[i'].

    Uses CSC column = assigned_area to find all such i' in O(1) lookup.
    """
    s = int(dm.csc_ptr[assigned_area])
    e = int(dm.csc_ptr[assigned_area + 1])
    if s < e:
        momentum[dm.csc_row[s:e]] -= delta * csc_speeds[s:e]


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 1 – momentum-based alternating greedy                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def _phase1_momentum(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    remaining: np.ndarray,
    momentum: np.ndarray,
    csc_speeds: np.ndarray,
    selected: list[int],
    selected_set: set[int],
    new_opened: set[int],
    fac_load: dict[int, float],
    fac_remaining: dict[int, float],
    fac_to_idx: dict[int, int],
    assigned_to: np.ndarray,
    r16: np.uint16,
    cap_min_f: float,
    cap_max_f: float,
    equity_ratio: int,
) -> tuple[set, np.ndarray, dict, np.ndarray, dict, list, set]:
    """
    Alternating max/min momentum greedy placement.

    Turn cycle: equity_ratio max turns followed by 1 min turn.
    A turn is skipped (not counted) when no valid candidate can be found
    for that turn type.  Stops when no valid candidate exists for any turn.
    """
    n = dm.n
    # Turn counter: determines max vs min.
    # Cycle length = equity_ratio + 1.
    # Turns 0 .. equity_ratio-1 in the cycle → max; turn equity_ratio → min.
    turn_in_cycle = 0
    cycle_len = equity_ratio + 1
    consecutive_no_valid = 0

    while consecutive_no_valid < 2:
        # Vectorised candidate scoring: sum of remaining demand within radius.
        scores = dm.marginal_coverage(remaining, radius=float(r16))
        effective = np.minimum(scores, cap_max_f) if cap_max_f < math.inf else scores
        effective[list(selected_set)] = -1.0

        valid_mask = effective >= cap_min_f - 1e-9

        if not np.any(valid_mask):
            break

        is_max_turn = (turn_in_cycle < equity_ratio)

        if is_max_turn:
            mom_masked = np.where(valid_mask, momentum, -np.inf)
            best_j = int(np.argmax(mom_masked))
        else:
            mom_masked = np.where(valid_mask, momentum, np.inf)
            best_j = int(np.argmin(mom_masked))

        # Exact demand-first fill for the chosen candidate.
        load, taken = _zone_demand_first(dm, remaining, demand, best_j, r16, cap_max_f)

        if load < cap_min_f - 1e-9:
            # Candidate failed exact fill — try the other turn type next.
            consecutive_no_valid += 1
            turn_in_cycle = (turn_in_cycle + 1) % cycle_len
            continue

        consecutive_no_valid = 0
        turn_in_cycle = (turn_in_cycle + 1) % cycle_len

        # Open the facility.
        fac_pos = len(selected)
        fac_to_idx[best_j] = fac_pos
        selected.append(best_j)
        selected_set.add(best_j)
        new_opened.add(best_j)
        fac_load[best_j] = load
        fac_remaining[best_j] = max(0.0, cap_max_f - load)

        # Assign areas and update momentum incrementally.
        for i in taken:
            old_rem = remaining[i]
            remaining[i] = 0.0
            assigned_to[i] = fac_pos
            _update_momentum(dm, momentum, csc_speeds, i, old_rem)

        logger.debug(
            "  CMCLP Phase 1 (%s): opened facility %d  load=%.1f  taken=%d",
            "MAX" if is_max_turn else "MIN", best_j, load, len(taken),
        )

    return new_opened, remaining, fac_load, assigned_to, fac_to_idx, selected, selected_set


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 1 – plain greedy fallback (no xy coordinates)                          #
# ─────────────────────────────────────────────────────────────────────────── #

def _phase1_greedy(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    remaining: np.ndarray,
    selected: list[int],
    selected_set: set[int],
    new_opened: set[int],
    fac_load: dict[int, float],
    fac_remaining: dict[int, float],
    fac_to_idx: dict[int, int],
    assigned_to: np.ndarray,
    r16: np.uint16,
    cap_min_f: float,
    cap_max_f: float,
) -> tuple[set, np.ndarray, dict, np.ndarray, dict, list, set]:
    """Standard demand-coverage greedy (used when xy is unavailable)."""
    while True:
        scores = dm.marginal_coverage(remaining, radius=float(r16))
        effective = np.minimum(scores, cap_max_f) if cap_max_f < math.inf else scores
        if selected_set:
            effective[list(selected_set)] = -1.0

        best_j = int(np.argmax(effective))
        if effective[best_j] < cap_min_f - 1e-9:
            break

        load, taken = _zone_demand_first(dm, remaining, demand, best_j, r16, cap_max_f)
        if load < cap_min_f - 1e-9:
            break

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

    return new_opened, remaining, fac_load, assigned_to, fac_to_idx, selected, selected_set


# ─────────────────────────────────────────────────────────────────────────── #
# Demand-first fill helper                                                      #
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

    Sorts unassigned areas in j's zone by remaining demand descending, then
    greedily assigns areas that fit within cap_max (skipping those that do not).

    Returns (total_load, list_of_assigned_area_indices).
    """
    rows, vals = dm.col_neighbors(j)
    in_zone = vals <= r16
    if not np.any(in_zone):
        return 0.0, []

    zone = rows[in_zone]
    candidates = sorted(
        ((float(remaining[i]), int(i)) for i in zone if remaining[i] > 1e-9),
        reverse=True,
    )

    load = 0.0
    taken: list[int] = []
    for dem, i in candidates:
        if load + dem > cap_max_f:
            continue   # skip items that don't fit; try smaller ones
        load += dem
        taken.append(i)

    return load, taken


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 2A – Complementary placement minimising weighted access time           #
# ─────────────────────────────────────────────────────────────────────────── #

def _phase2a_access_time(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    remaining: np.ndarray,
    selected: list[int],
    selected_set: set[int],
    fac_load: dict[int, float],
    fac_remaining: dict[int, float],
    fac_to_idx: dict[int, int],
    assigned_to: np.ndarray,
    n_pre: int,
    radius: float,
    cap_min: float,
    cap_max: float,
) -> None:
    """
    Greedy complementary placement that minimises total weighted access time
    (TWAT = Σ demand_i × distance(i, assigned_facility_i)).

    Each iteration selects the unselected candidate c that produces the
    largest negative Δ(TWAT) when placed:
      - c may absorb unassigned areas within its radius for free.
      - c may steal a served area from another NEW cluster if c is closer
        to that area than its current facility, provided the invaded
        cluster stays above cap_min after the steal.
      - c must accumulate between cap_min and cap_max total demand.

    Stops when no candidate can reduce TWAT.
    """
    while True:
        best_c       = -1
        best_delta   = 0.0          # must become strictly negative to accept
        best_areas: list[tuple[int, int, float]] = []  # (area, old_fac_pos, amount)
        best_load    = 0.0

        for c in range(dm.n):
            if c in selected_set:
                continue

            zone_rows, zone_dists = dm.col_neighbors_full(c, radius)
            if len(zone_rows) == 0:
                continue

            order      = np.argsort(zone_dists, kind="stable")
            zone_rows  = zone_rows[order]
            zone_dists = zone_dists[order]

            c_load   = 0.0
            delta    = 0.0
            inv_load: dict[int, float] = {}   # old_fac -> demand being stolen from it
            cand_areas: list[tuple[int, int, float]] = []

            for nbr, d_c in zip(zone_rows.tolist(), zone_dists.tolist()):
                nbr = int(nbr)
                if c_load >= cap_max - 1e-9:
                    break

                dem = float(demand[nbr])
                if dem <= 1e-9:
                    continue

                asgn_pos = int(assigned_to[nbr])

                if asgn_pos == -1:
                    # Unassigned area: assign to c (new coverage, always beneficial).
                    take = min(dem, cap_max - c_load)
                    if take < 1e-9:
                        continue
                    c_load += take
                    # Δ is negative (improvement): gaining coverage with finite distance.
                    delta  -= take * d_c
                    cand_areas.append((nbr, -1, take))
                    continue

                # Pre-existing cluster: not invasible.
                if asgn_pos < n_pre:
                    continue

                # Area belongs to a new cluster — steal only if c is strictly closer.
                old_fac = selected[asgn_pos]
                d_old   = dm.distance_time(nbr, old_fac)
                if d_c >= d_old - 1e-9:
                    continue

                # Amount currently assigned to old_fac for this area.
                amount = dem - float(remaining[nbr])
                if amount <= 1e-9:
                    continue

                take = min(amount, cap_max - c_load)
                if take < 1e-9:
                    continue

                # Feasibility: old_fac must keep >= cap_min after losing `take`.
                already_stolen = inv_load.get(old_fac, 0.0)
                if fac_load[old_fac] - already_stolen - take < cap_min - 1e-9:
                    # Try a smaller steal that keeps old_fac at exactly cap_min.
                    take = fac_load[old_fac] - already_stolen - cap_min
                    if take < 1e-9:
                        continue

                c_load += take
                delta  += take * (d_c - d_old)          # negative when d_c < d_old
                inv_load[old_fac] = already_stolen + take
                cand_areas.append((nbr, asgn_pos, take))

            if c_load < cap_min - 1e-9:
                continue

            if delta < best_delta - 1e-6:
                best_delta = delta
                best_c     = c
                best_areas = cand_areas
                best_load  = c_load

        if best_c == -1:
            break  # No candidate improves weighted access time.

        # ── Place best_c ──────────────────────────────────────────────────── #
        fac_pos = len(selected)
        fac_to_idx[best_c] = fac_pos
        selected.append(best_c)
        selected_set.add(best_c)
        fac_load[best_c]     = 0.0
        fac_remaining[best_c] = cap_max

        for area, old_pos, take in best_areas:
            if old_pos >= n_pre:
                old_fac = selected[old_pos]
                fac_load[old_fac]      = max(0.0, fac_load[old_fac] - take)
                if fac_remaining[old_fac] != math.inf:
                    fac_remaining[old_fac] += take
                remaining[area] += take          # restore to pool

            # Assign to c (may be partial for a stolen area).
            remaining[area]      = max(0.0, remaining[area] - take)
            fac_load[best_c]    += take
            if fac_remaining[best_c] != math.inf:
                fac_remaining[best_c] -= take
            if remaining[area] < 1e-9:
                assigned_to[area] = fac_pos     # fully served by c now

        logger.debug(
            "  CMCLP Phase 2A: placed c=%d  Δwatt=%.2f  load=%.1f  areas=%d",
            best_c, best_delta, best_load, len(best_areas),
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 2B – Prune facilities whose demand can be absorbed by neighbours       #
# ─────────────────────────────────────────────────────────────────────────── #

def _phase2b_prune(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    remaining: np.ndarray,
    selected: list[int],
    selected_set: set[int],
    fac_load: dict[int, float],
    fac_remaining: dict[int, float],
    fac_to_idx: dict[int, int],
    assigned_to: np.ndarray,
    n_pre: int,
    radius: float,
    cap_min: float,
    cap_max: float,
) -> None:
    """
    Iteratively remove new facilities that are made redundant by their
    neighbours.

    A facility fac is redundant when every area it serves can be reassigned
    to a different open facility within the service radius without any
    receiving facility exceeding cap_max.

    New facilities are considered for removal in ascending load order
    (lightest first) — these are the most likely to be redundant clusters
    left over from Phase 1.  The loop repeats until no further removal
    is possible.
    """
    changed = True
    while changed:
        changed = False

        # Build area → facility mapping for quick lookup.
        area_fac: dict[int, int] = {}          # area index → facility node index
        fac_areas: dict[int, list[int]] = {}   # facility node → served area indices
        for area in range(dm.n):
            pos = int(assigned_to[area])
            if pos < 0 or pos >= len(selected):
                continue
            fac = selected[pos]
            area_fac[area] = fac
            fac_areas.setdefault(fac, []).append(area)

        # Candidates: new (non-pre-existing) facilities, lightest load first.
        candidates = sorted(
            [f for f in selected[n_pre:] if fac_load.get(f, 0.0) >= 0.0],
            key=lambda f: fac_load.get(f, 0.0),
        )

        for fac in candidates:
            served = fac_areas.get(fac, [])
            if not served:
                # Empty facility — remove immediately.
                _remove_facility(fac, selected, selected_set, fac_load,
                                 fac_remaining, fac_to_idx, assigned_to)
                changed = True
                break

            # Simulate reassignment of each served area to another facility.
            temp_remaining_cap: dict[int, float] = dict(fac_remaining)
            reassign: dict[int, int] = {}   # area → receiving facility
            feasible = True

            for area in sorted(served, key=lambda a: float(demand[a]), reverse=True):
                dem = float(demand[area])
                if dem <= 1e-9:
                    reassign[area] = -1
                    continue

                # Find nearest other open facility within radius with capacity.
                best_alt  = -1
                best_dist = math.inf
                for g in selected_set:
                    if g == fac:
                        continue
                    d = dm.distance_time(area, g)
                    if d > radius:
                        continue
                    cap_left = temp_remaining_cap.get(g, 0.0)
                    if cap_left < dem - 1e-9:
                        continue
                    if d < best_dist:
                        best_dist = d
                        best_alt  = g

                if best_alt == -1:
                    feasible = False
                    break

                reassign[area] = best_alt
                temp_remaining_cap[best_alt] -= dem

            if not feasible:
                continue

            # Apply reassignment.
            for area, g in reassign.items():
                if g == -1:
                    continue
                dem = float(demand[area])
                fac_load[g] += dem
                if fac_remaining[g] != math.inf:
                    fac_remaining[g] -= dem
                assigned_to[area] = fac_to_idx[g]

            removed_load = fac_load.get(fac, 0.0)
            _remove_facility(fac, selected, selected_set, fac_load,
                             fac_remaining, fac_to_idx, assigned_to)
            logger.debug(
                "  CMCLP Phase 2B: removed facility %d  (load=%.1f redistributed)",
                fac, removed_load,
            )
            changed = True
            break   # restart with updated selected list


def _remove_facility(
    fac: int,
    selected: list[int],
    selected_set: set[int],
    fac_load: dict[int, float],
    fac_remaining: dict[int, float],
    fac_to_idx: dict[int, int],
    assigned_to: np.ndarray,
) -> None:
    """
    Remove fac from selected and rebuild fac_to_idx.
    Areas that pointed to the old position of any shifted facility are
    updated so assigned_to stays consistent with the new positions.
    """
    old_pos = fac_to_idx[fac]
    selected.pop(old_pos)
    selected_set.discard(fac)
    fac_load.pop(fac, None)
    fac_remaining.pop(fac, None)
    fac_to_idx.pop(fac, None)

    # Rebuild position index and fix assigned_to for facilities that shifted left.
    for new_pos in range(old_pos, len(selected)):
        f = selected[new_pos]
        fac_to_idx[f] = new_pos

    # Shift assigned_to values that were > old_pos down by 1.
    mask = assigned_to > old_pos
    assigned_to[mask] -= 1
    # Areas that pointed exactly to old_pos are now unassigned (fac was removed).
    assigned_to[assigned_to == old_pos] = -1


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 0 helper: nearest-first for pre-existing facilities                    #
# ─────────────────────────────────────────────────────────────────────────── #

def _assign_zone_nearest_first(
    dm: SparseDistanceMatrix,
    remaining: np.ndarray,
    fac: int,
    fac_load: dict[int, float],
    fac_remaining: dict[int, float],
    r16: np.uint16,
    selected_set: set[int],
    assigned_to: np.ndarray,
    fac_pos: int,
    fac_to_idx: dict[int, int],
) -> None:
    """Assign demand nearest-first up to capacity (Phase 0 only)."""
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

