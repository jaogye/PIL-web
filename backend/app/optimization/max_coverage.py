"""
Capacitated Maximum Coverage Location Problem (CMCLP).

Objective: maximise total demand served by placing as many facilities as needed,
subject to:
  - Each facility serves demand only from areas within the service radius.
  - Each facility serves at most cap_max demand (nearest areas take priority).
  - Overflow beyond cap_max is redirected to the nearest other facility
    within radius that still has remaining capacity.
  - A facility is only placed if it can attract at least cap_min demand
    from currently unassigned demand within its zone.
  - Facilities are added greedily until no candidate can attract >= cap_min.

Algorithm: Greedy Add with capacity-aware marginal gain.
  Each step uses marginal_coverage() — O(nnz) — to find the best candidate,
  then assigns demand via vectorised numpy (nearest-first within zone).
  Overflow redistribution uses CSR rows, O(k) per overflowed area.
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
    Solve the Capacitated MCLP.

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

    remaining: np.ndarray = demand.astype(np.float64).copy()
    selected: list[int] = []
    selected_set: set[int] = set()
    fac_load: dict[int, float] = {}      # total demand assigned to each facility
    fac_remaining: dict[int, float] = {} # remaining capacity of each facility

    # ── Pre-selected (existing) facilities ─────────────────────────────── #
    for fac in pre_selected:
        selected.append(fac)
        selected_set.add(fac)
        fac_load[fac] = 0.0
        fac_remaining[fac] = cap_max_f

    if pre_selected:
        for fac in pre_selected:
            _assign_zone(dm, remaining, fac, fac_load, fac_remaining,
                         radius, selected_set)

    # ── Phase 1: Greedy Add ─────────────────────────────────────────────── #
    while True:
        # O(nnz): for each candidate j, sum of remaining demand in its zone.
        raw_gain = dm.marginal_coverage(remaining, radius)

        # Cap at max capacity (upper bound on what the best candidate can absorb).
        effective_gain = np.minimum(raw_gain, cap_max_f) if cap_max is not None else raw_gain

        # Exclude already-selected facilities.
        if selected_set:
            effective_gain[list(selected_set)] = -1.0

        best = int(np.argmax(effective_gain))

        if effective_gain[best] < cap_min_f - 1e-9:
            break  # No candidate can attract enough unassigned demand.

        # Place facility at best.
        selected.append(best)
        selected_set.add(best)
        fac_load[best] = 0.0
        fac_remaining[best] = cap_max_f

        _assign_zone(dm, remaining, best, fac_load, fac_remaining,
                     radius, selected_set)

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
    # Since we only assign within radius, remaining[i] < demand[i] iff covered.
    covered = remaining < demand - 1e-9

    covered_dem = float(np.sum(demand[covered]))
    coverage_pct = (
        round(covered_dem / total_demand * 100, 2) if total_demand > 0 else 0.0
    )

    # Nearest-facility assignment array (used by the route for visualization).
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
) -> None:
    """
    Assign demand from fac's zone to fac (nearest-first, up to cap_max).

    Demand that exceeds fac's capacity (overflow) is redirected to other
    selected facilities within radius that still have remaining capacity,
    sorted by their distance to each overflowed area.
    """
    rows, vals = dm.col_neighbors(fac)
    r16 = np.uint16(min(int(radius), MAX_DIST))
    in_zone = vals <= r16

    if not np.any(in_zone):
        return

    zone_rows = rows[in_zone]
    zone_vals = vals[in_zone]

    # Sort zone areas nearest-first so they get capacity priority.
    order = np.argsort(zone_vals, kind="stable")
    zone_rows = zone_rows[order]

    cap = fac_remaining[fac]
    avail = remaining[zone_rows]  # unassigned demand in each zone area

    if cap == math.inf:
        # No limit: assign all available demand in zone.
        assigned = float(np.sum(avail))
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
            remaining[zone_rows] -= avail
            fac_load[fac] += float(cumsum[-1])
            fac_remaining[fac] -= float(cumsum[-1])
            return

        # Find the cutoff: areas 0..k-1 are fully taken; area k is partial.
        # searchsorted 'right': cumsum[k-1] <= cap < cumsum[k]
        k = int(np.searchsorted(cumsum, cap, side="right"))
        take = np.zeros(len(avail), dtype=np.float64)
        if k > 0:
            take[:k] = avail[:k]
        partial = cap - (float(cumsum[k - 1]) if k > 0 else 0.0)
        if k < len(avail) and partial > 1e-9:
            take[k] = partial

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

        s = int(dm.csr_ptr[i])
        e = int(dm.csr_ptr[i + 1])
        nbrs = dm.csr_col[s:e]
        dsts = dm.csr_val[s:e]

        # Candidate facilities: selected, within radius, with capacity, != fac.
        candidates = sorted(
            (int(dsts[k]), int(nbrs[k]))
            for k in range(len(nbrs))
            if int(nbrs[k]) in selected_set
            and int(nbrs[k]) != fac
            and dsts[k] <= r16
        )

        for _, other in candidates:
            if remaining[i] <= 1e-9:
                break
            cap_o = fac_remaining[other]
            if cap_o <= 1e-9:
                continue
            take = min(remaining[i], cap_o)
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
