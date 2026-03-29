
"""
Capacitated Maximum Coverage Location Problem (CMCLP).

Algorithm faithful to Recubrimiento.java (legacy LIP system):

Phase 0 – Pre-existing centres (IniActual):
    Absorb demand from existing facilities nearest-first up to their capacity.

Phase 1 – Greedy Add (ColocaCluster):
    Repeatedly pick the unassigned candidate that maximises total accessible
    unassigned demand (within radius, up to cap_max).  Stop when no candidate
    can attract >= cap_min demand.

Phase 2 – Complementary clusters (ColocaClusterComplementarios):
    When Phase 1 stops (no unassigned node alone meets cap_min), allow placing
    a facility that "invades" demand from OTHER new clusters (not pre-existing).
    Selection criterion: minimum invasion cost while still reaching cap_min.
    Invasion is rejected if any invaded cluster would fall below cap_min.
    Iterates until no valid complementary candidate exists.
    Only active when cap_min > 0.
"""

from __future__ import annotations

import logging
import math
import numpy as np
from dataclasses import dataclass, field

from .sparse_matrix import MAX_DIST, SparseDistanceMatrix

logger = logging.getLogger(__name__)


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
    p: int,                         # kept for API compatibility; ignored
    radius: float,
    cap_min: float = 0.0,
    cap_max: float | None = None,
    pre_selected: list[int] | None = None,
) -> MaxCoverageResult:
    """
    Solve the Capacitated MCLP via the three-phase Recubrimiento heuristic.

    Parameters
    ----------
    distance_matrix : SparseDistanceMatrix
    demand          : np.ndarray, shape (n,)
    p               : int — ignored; facilities are placed until cap_min
                      cannot be satisfied, not until a fixed count is reached.
    radius          : float — service radius in minutes.
    cap_min         : float — minimum demand a facility must attract to be placed.
    cap_max         : float | None — maximum demand a facility can serve (None = ∞).
    pre_selected    : list[int] | None — indices fixed as existing facilities.
    """
    dm = distance_matrix
    n = dm.n
    cap_max_f = float(cap_max) if cap_max is not None else math.inf
    cap_min_f = float(cap_min)
    total_demand = float(np.sum(demand))
    pre_selected = list(pre_selected or [])
    n_pre_selected = len(pre_selected)

    remaining: np.ndarray = demand.astype(np.float64).copy()
    selected: list[int] = []
    selected_set: set[int] = set()
    fac_load: dict[int, float] = {}       # total demand assigned to each facility
    fac_remaining: dict[int, float] = {}  # remaining capacity of each facility

    # assigned_to[i]: index into `selected` list for area i's primary cluster.
    # -1 = unassigned.  0..n_pre_selected-1 = pre-existing.  >= n_pre_selected = new.
    assigned_to: np.ndarray = np.full(n, -1, dtype=np.int32)

    # fac_to_idx: maps facility node-index -> its position in `selected` list.
    fac_to_idx: dict[int, int] = {}

    # ── Phase 0: Pre-existing (existing) facilities ─────────────────────── #
    for fac_pos, fac in enumerate(pre_selected):
        selected.append(fac)
        selected_set.add(fac)
        fac_load[fac] = 0.0
        fac_remaining[fac] = cap_max_f
        fac_to_idx[fac] = fac_pos

    if pre_selected:
        for fac_pos, fac in enumerate(pre_selected):
            _assign_zone(dm, remaining, fac, fac_load, fac_remaining,
                         radius, selected_set, assigned_to, fac_pos, fac_to_idx)

    # ── Phase 1: Greedy Add ─────────────────────────────────────────────── #
    while True:
        # Sum remaining demand within radius per candidate (stored + estimated).
        raw_gain = dm.marginal_coverage_full(remaining, radius)

        # Cap at max capacity (upper bound on what the best candidate can absorb).
        effective_gain = np.minimum(raw_gain, cap_max_f) if cap_max is not None else raw_gain

        # Exclude already-selected facilities.
        if selected_set:
            effective_gain[list(selected_set)] = -1.0

        best = int(np.argmax(effective_gain))

        if effective_gain[best] < cap_min_f - 1e-9:
            break  # No candidate can attract enough unassigned demand.

        fac_pos = len(selected)
        fac_to_idx[best] = fac_pos
        selected.append(best)
        selected_set.add(best)
        fac_load[best] = 0.0
        fac_remaining[best] = cap_max_f

        _assign_zone(dm, remaining, best, fac_load, fac_remaining,
                     radius, selected_set, assigned_to, fac_pos, fac_to_idx)

    # ── Phase 2: Complementary clusters with invasion ───────────────────── #
    # Mirrors ColocaClusterComplementarios: when no unassigned node alone meets
    # cap_min, allow stealing demand from other NEW clusters (not pre-existing).
    if cap_min_f > 0:
        _phase2_complementary(
            dm, demand, remaining, selected, selected_set,
            fac_load, fac_remaining, fac_to_idx, assigned_to,
            n_pre_selected, radius, cap_min_f, cap_max_f,
        )

    # ── Remove facilities below cap_min (defensive check) ──────────────── #
    valid = [f for f in selected if fac_load.get(f, 0.0) >= cap_min_f - 1e-9]
    if len(valid) < len(selected):
        logger.warning(
            "  CMCLP: dropped %d facilities below cap_min after placement",
            len(selected) - len(valid),
        )
    selected = valid

    # ── Coverage & assignment ──────────────────────────────────────────── #
    # An area is "covered" if any of its demand was assigned to a facility.
    covered = remaining < demand - 1e-9

    covered_dem = float(np.sum(demand[covered]))
    coverage_pct = (
        round(covered_dem / total_demand * 100, 2) if total_demand > 0 else 0.0
    )

    # Nearest-facility assignment array (used by the route for visualisation).
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
# Phase 2 – Complementary cluster placement with cluster invasion               #
# ─────────────────────────────────────────────────────────────────────────── #

def _phase2_complementary(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    remaining: np.ndarray,
    selected: list[int],
    selected_set: set[int],
    fac_load: dict[int, float],
    fac_remaining: dict[int, float],
    fac_to_idx: dict[int, int],
    assigned_to: np.ndarray,
    n_pre_selected: int,
    radius: float,
    cap_min: float,
    cap_max: float,
) -> None:
    """
    Phase 2 (ColocaClusterComplementarios): place complementary facilities by
    allowing them to steal demand from previously placed NEW clusters.

    For each unassigned candidate node the function simulates:
      - how much total demand (unassigned + stealable from new clusters)
        is accessible within the service radius;
      - the invasion cost = total demand stolen from new clusters;
      - feasibility = no invaded cluster falls below cap_min after the steal.

    The candidate with the minimum feasible invasion cost that still reaches
    cap_min is selected.  The loop repeats until no valid candidate exists.
    """
    while True:
        best_candidate: int = -1
        best_invasion: float = math.inf
        best_demand: float = 0.0

        # Candidates: unassigned nodes not already selected as a centre.
        unassigned_mask = (assigned_to == -1) & (remaining > 1e-9)

        for candidate in np.where(unassigned_mask)[0]:
            candidate = int(candidate)
            if candidate in selected_set:
                continue

            # -- Simulation: count accessible demand and invasion cost -------- #
            col_rows, col_vals = dm.col_neighbors_full(candidate, radius)
            if len(col_rows) == 0:
                continue

            order = np.argsort(col_vals, kind="stable")
            zone_areas = col_rows[order]

            total_dem = float(demand[candidate])   # include own demand (Java: nDemanda = demanda)
            invasion = 0.0
            invasion_per_pos: dict[int, float] = {}  # fac_pos -> stolen demand
            feasible = True

            for nbr in zone_areas:
                nbr = int(nbr)
                if nbr == candidate:
                    continue

                # Skip cluster centres (Java: !nodos[nInd].centro)
                if nbr in selected_set:
                    continue

                nbr_pos = int(assigned_to[nbr])

                # Skip areas owned by pre-existing clusters (not invasible)
                if 0 <= nbr_pos < n_pre_selected:
                    continue

                # Count this area's demand (unassigned OR assigned to a new cluster)
                total_dem += float(demand[nbr])

                # Track invasion if the neighbour belongs to a new cluster
                if nbr_pos >= n_pre_selected:
                    stolen = float(demand[nbr] - remaining[nbr])  # amount assigned to that cluster
                    if stolen < 1e-9:
                        # Fractional edge-case: count full demand as invasion cost
                        stolen = float(demand[nbr])
                    invasion += stolen
                    prev = invasion_per_pos.get(nbr_pos, 0.0)
                    invasion_per_pos[nbr_pos] = prev + stolen

                    # Feasibility: invaded cluster must keep >= cap_min after steal
                    invaded_fac = selected[nbr_pos]
                    remaining_load = fac_load[invaded_fac] - invasion_per_pos[nbr_pos]
                    if remaining_load < cap_min - 1e-9:
                        feasible = False
                        break

                if total_dem >= cap_min:
                    break

            if not feasible or total_dem < cap_min - 1e-9:
                continue

            # Better candidate: lower invasion; ties broken by more demand.
            if invasion < best_invasion - 1e-9 or (
                abs(invasion - best_invasion) < 1e-9 and total_dem > best_demand
            ):
                best_invasion = invasion
                best_candidate = candidate
                best_demand = total_dem

        if best_candidate == -1:
            break  # No valid complementary candidate found.

        logger.debug(
            "  CMCLP Phase 2: complementary facility at area %d "
            "(invasion=%.1f, reachable_demand=%.1f)",
            best_candidate, best_invasion, best_demand,
        )

        # -- Execution: steal from invaded clusters, then place facility ----- #
        col_rows, col_vals = dm.col_neighbors_full(best_candidate, radius)
        order = np.argsort(col_vals, kind="stable")
        zone_areas = col_rows[order]

        accumulated = float(demand[best_candidate])

        for nbr in zone_areas:
            nbr = int(nbr)
            if nbr == best_candidate:
                continue
            if nbr in selected_set:
                continue

            nbr_pos = int(assigned_to[nbr])

            # Skip pre-existing cluster members
            if 0 <= nbr_pos < n_pre_selected:
                continue

            # Steal from a new cluster: restore the area's demand to "unassigned"
            if nbr_pos >= n_pre_selected:
                invaded_fac = selected[nbr_pos]
                stolen = float(demand[nbr] - remaining[nbr])
                if stolen < 1e-9:
                    stolen = float(demand[nbr])
                fac_load[invaded_fac] = max(0.0, fac_load[invaded_fac] - stolen)
                if fac_remaining[invaded_fac] != math.inf:
                    fac_remaining[invaded_fac] += stolen
                remaining[nbr] = float(demand[nbr])  # fully unassigned
                assigned_to[nbr] = -1

            # Count accessible demand (now unassigned after potential steal)
            if int(assigned_to[nbr]) == -1:
                accumulated += float(demand[nbr])

            if accumulated >= cap_min:
                break  # Reached cap_min; stop stealing (matches Java behaviour)

        # Place the new complementary facility
        fac_pos = len(selected)
        fac_to_idx[best_candidate] = fac_pos
        selected.append(best_candidate)
        selected_set.add(best_candidate)
        fac_load[best_candidate] = 0.0
        fac_remaining[best_candidate] = cap_max

        # Assign zone (up to cap_max) for the new complementary facility
        _assign_zone(
            dm, remaining, best_candidate, fac_load, fac_remaining,
            radius, selected_set, assigned_to, fac_pos, fac_to_idx,
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Demand-assignment helpers                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def _assign_zone(
    dm: SparseDistanceMatrix,
    remaining: np.ndarray,
    fac: int,
    fac_load: dict[int, float],
    fac_remaining: dict[int, float],
    radius: float,
    selected_set: set[int],
    assigned_to: np.ndarray,
    fac_pos: int,
    fac_to_idx: dict[int, int],
) -> None:
    """
    Assign demand from fac's zone to fac (nearest-first, up to cap_max).

    Also updates assigned_to[area] = fac_pos for any area that receives its
    first demand assignment from this facility.

    Demand that exceeds fac's capacity (overflow) is redirected to other
    selected facilities within radius that still have remaining capacity,
    sorted by their distance to each overflowed area.
    """
    # Full zone: stored neighbors + estimated-distance areas within radius.
    zone_rows, zone_vals = dm.col_neighbors_full(fac, radius)

    if len(zone_rows) == 0:
        return

    # Sort zone areas nearest-first so they get capacity priority.
    order = np.argsort(zone_vals, kind="stable")
    zone_rows = zone_rows[order]
    zone_vals = zone_vals[order]

    cap = fac_remaining[fac]
    avail = remaining[zone_rows]  # unassigned demand in each zone area

    if cap == math.inf:
        # No limit: assign all available demand in zone.
        assigned = float(np.sum(avail))
        for area in zone_rows:
            if remaining[area] > 1e-9 and assigned_to[area] == -1:
                assigned_to[area] = fac_pos
        remaining[zone_rows] -= avail
        fac_load[fac] += assigned
        return

    if cap <= 1e-9:
        # Facility already full; everything is overflow.
        overflow_start = 0
    else:
        cumsum = np.cumsum(avail)
        if cumsum[-1] <= cap:
            # All zone demand fits within capacity.
            for area in zone_rows:
                if remaining[area] > 1e-9 and assigned_to[area] == -1:
                    assigned_to[area] = fac_pos
            remaining[zone_rows] -= avail
            fac_load[fac] += float(cumsum[-1])
            fac_remaining[fac] -= float(cumsum[-1])
            return

        # Find the cutoff: areas 0..k-1 are fully taken; area k is partial.
        k = int(np.searchsorted(cumsum, cap, side="right"))
        take = np.zeros(len(avail), dtype=np.float64)
        if k > 0:
            take[:k] = avail[:k]
        partial = cap - (float(cumsum[k - 1]) if k > 0 else 0.0)
        if k < len(avail) and partial > 1e-9:
            take[k] = partial

        # Update assigned_to for areas receiving demand from this facility.
        for j in range(min(k + 1, len(zone_rows))):
            if take[j] > 1e-9 and assigned_to[zone_rows[j]] == -1:
                assigned_to[zone_rows[j]] = fac_pos

        remaining[zone_rows] -= take
        fac_load[fac] += float(np.sum(take))
        fac_remaining[fac] = 0.0
        overflow_start = k  # area k still has (avail[k] - partial) remaining

    # ── Overflow redistribution ─────────────────────────────────────────── #
    # For each overflowed area, redirect remaining demand to the nearest other
    # selected facility within radius that has available capacity.
    for i in zone_rows[overflow_start:].tolist():
        if remaining[i] <= 1e-9:
            continue

        # Candidate facilities: selected, within radius, with capacity, != fac.
        # Use distance_time() so estimated distances are included.
        candidates = sorted(
            (dm.distance_time(i, other), other)
            for other in selected_set
            if other != fac and dm.distance_time(i, other) <= radius
        )

        for _, other in candidates:
            if remaining[i] <= 1e-9:
                break
            cap_o = fac_remaining[other]
            if cap_o <= 1e-9:
                continue
            take = min(remaining[i], cap_o)
            if assigned_to[i] == -1 and take > 1e-9:
                assigned_to[i] = fac_to_idx[other]
            remaining[i] -= take
            fac_load[other] += take
            if cap_o != math.inf:
                fac_remaining[other] -= take


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


