"""
Capacity-constrained census-area assignment.

Pure functions operating on SparseDistanceMatrix and numpy arrays.
No database or schema imports — safe to use from any layer.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .sparse_matrix import SparseDistanceMatrix


def sorted_facilities_for_area(
    dm: SparseDistanceMatrix,
    area_idx: int,
    fac_set: set[int],
) -> list[tuple[float, int]]:
    """
    Return (distance, facility_index) pairs sorted by travel time ascending.

    Uses dm.distance_time() which returns the stored value when available
    and falls back to the harmonic-mean speed estimate for missing pairs.
    """
    pairs = [(dm.distance_time(area_idx, fac), fac) for fac in fac_set]
    pairs.sort()
    return pairs


def single_capacity_pass(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    facility_indices: list[int],
    facility_capacities: dict[int, float],
    radius: float | None = None,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, float]]:
    """
    One greedy pass: assign each area to its nearest facility(ies) subject to
    capacity, processing areas in ascending order of distance to nearest facility.

    If radius is provided, only facilities within that travel-time radius of the
    area are eligible.  Areas with no facility within radius are left unassigned.

    Returns
    -------
    (assignments, fac_demand)
        assignments[area_idx] = [(facility_idx, demand_amount), ...]
        fac_demand[facility_idx] = total demand absorbed
    """
    fac_set = set(facility_indices)
    remaining: dict[int, float] = {f: facility_capacities.get(f, math.inf) for f in facility_indices}
    fac_demand: dict[int, float] = defaultdict(float)
    assignments: dict[int, list[tuple[int, float]]] = {}

    # Nearest-first: areas closest to any facility get capacity priority.
    min_dists = dm.min_dist_to_set(facility_indices)
    area_order = np.argsort(min_dists, kind="stable").tolist()

    for i in area_order:
        d = float(demand[i])
        if d <= 1e-9:
            continue

        area_asgn: list[tuple[int, float]] = []
        rem = d
        for dist, fac in sorted_facilities_for_area(dm, i, fac_set):
            if rem <= 1e-9:
                break
            if radius is not None and dist > radius:
                break  # sorted by distance; all subsequent are farther
            cap = remaining.get(fac, 0.0)
            if cap <= 1e-9:
                continue
            take = min(rem, cap)
            area_asgn.append((fac, take))
            remaining[fac] -= take
            fac_demand[fac] += take
            rem -= take

        if area_asgn:
            assignments[i] = area_asgn

    return assignments, dict(fac_demand)


def capacity_assignment(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    solver_facility_indices: list[int],
    facility_capacities: dict[int, float],
    min_capacity: float | None,
    pre_selected_set: set[int],
    radius: float | None = None,
) -> tuple[list[int], dict[int, list[tuple[int, float]]]]:
    """
    Assign census areas to facilities with capacity constraints.

    Areas are processed nearest-first so that the closest areas have priority
    in filling a facility's capacity.  When an area's demand exceeds the
    remaining capacity of its nearest facility, the overflow goes to the next
    nearest facility with available capacity (partial assignment).

    If radius is provided, only facilities within that travel-time radius are
    considered for each area.  Areas with no facility within radius are left
    unassigned (uncovered).

    After assignment, any *planned* (non-pre-selected) facility whose total
    assigned demand is below min_capacity is discarded, and the entire
    assignment is recomputed without it.  This repeats until stable.

    Returns
    -------
    (final_facility_indices, assignments)
        assignments[area_idx] = [(facility_idx, demand_amount), ...]
    """
    active = list(solver_facility_indices)
    assignments: dict[int, list[tuple[int, float]]] = {}

    for _ in range(len(solver_facility_indices) + 1):
        if not active:
            break
        assignments, fac_demand = single_capacity_pass(dm, demand, active, facility_capacities, radius)

        if not min_capacity:
            break

        to_discard = [
            f for f in active
            if f not in pre_selected_set and fac_demand.get(f, 0.0) < min_capacity
        ]
        if not to_discard:
            break

        discard_set = set(to_discard)
        active = [f for f in active if f not in discard_set]

    return active, assignments
