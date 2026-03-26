"""
P-Center Optimization — Layered Search.

Implements the L-Layered Search algorithm from:
  Kramer, Iori, Vidal (2018). "Mathematical models and search algorithms
  for the capacitated p-center problem." arXiv:1803.04865

Key idea: the optimal radius r* lies in the sorted set of distinct distances D.
Testing r is a feasibility check (can p facilities cover all areas within r?).
Rather than binary search, the paper proposes an L-Layered Search that limits
feasible solves to O(L) while keeping total work at O(L · N^(1/L)), where N = |D|.

  Strategy     Total subproblems   Feasible subproblems
  ─────────    ─────────────────   ────────────────────
  Sequential   O(N)                1
  2-Layers     O(√N)               2
  3-Layers     O(N^(1/3))          3            ← default
  L-Layers     O(L · N^(1/L))      L
  Binary       O(log N)            O(log N)

Feasibility oracle: greedy dominating-set (exact for the uncapacitated p-center
when all areas are candidate facilities — always finds a cover if one exists).

Upper bound: Furthest Insertion construction heuristic (Section 6.3) to open p
facilities greedily and narrow the search interval before Layered Search begins.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .sparse_matrix import MAX_DIST, SparseDistanceMatrix

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result dataclass                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class PCenterResult:
    facility_indices: list[int]
    assignment: list[int]
    optimal_radius: float
    coverage_stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Public entry point                                                             #
# --------------------------------------------------------------------------- #

def solve(
    distance_matrix: SparseDistanceMatrix,
    demand: np.ndarray,
    p: int,
    pre_selected: list[int] | None = None,
    layers: int = 3,
) -> PCenterResult:
    """
    Solve the uncapacitated p-center problem via L-Layered Search.

    Parameters
    ----------
    distance_matrix : SparseDistanceMatrix
    demand : np.ndarray, shape (n,)
        Population demand per area (used for coverage statistics only).
    p : int
        Number of facilities to open.
    pre_selected : list[int] or None
        Area indices forced open (existing infrastructure).
    layers : int
        Number of layers L in the Layered Search (default 3).
        Larger L reduces total subproblems but increases feasible solves.
        Recommended: L ∈ {2, 3, 4} (Section 5.3 of the paper).
    """
    dm = distance_matrix
    n = dm.n
    pre_selected = list(pre_selected or [])

    # Trivial case: all areas are facilities.
    if p >= n:
        indices = list(range(n))
        md = dm.min_dist_to_set(indices)
        return PCenterResult(
            facility_indices=indices,
            assignment=list(range(n)),
            optimal_radius=0.0,
            coverage_stats=_compute_stats(demand, md, 0.0),
        )

    # ------------------------------------------------------------------ #
    # Phase 0 – Candidate radii                                           #
    # Optimal r* must be one of the N distinct stored distance values.    #
    # (Decomposition principle, Section 3 of the paper.)                  #
    # ------------------------------------------------------------------ #
    _t0 = time.perf_counter()
    unique_radii = dm.unique_radii()   # sorted uint16 array
    N = len(unique_radii)
    logger.warning(
        "  p-center Phase 0 (unique_radii n=%d, N=%d): %.2fs",
        n, N, time.perf_counter() - _t0,
    )

    # ------------------------------------------------------------------ #
    # Phase 1 – Initial upper bound via Furthest Insertion                #
    # (Section 6.3 — construction heuristic for upper bound.)             #
    # Narrows the search interval [i_low, i_up] before Layered Search.   #
    # ------------------------------------------------------------------ #
    _t1 = time.perf_counter()
    ub_facilities = _furthest_insertion(dm, n, p, pre_selected)
    ub_radius = float(np.max(dm.min_dist_to_set(ub_facilities)))

    # i_up: index in unique_radii of the upper-bound radius (last known feasible).
    i_up = int(np.searchsorted(unique_radii, ub_radius, side="right")) - 1
    i_up = max(0, min(i_up, N - 1))
    # i_low = -1 means "all values up to index 0 are unknown"; we use 0-based
    # indexing and treat the entry just before index 0 as infeasible sentinel.
    i_low = -1   # last known infeasible index (sentinel: nothing tested yet)

    logger.warning(
        "  p-center Phase 1 (UB radius=%.1f, i_up=%d/%d): %.2fs",
        ub_radius, i_up, N - 1, time.perf_counter() - _t1,
    )

    # ------------------------------------------------------------------ #
    # Phase 2 – Layered Search (Algorithm 1, Kramer et al. 2018)          #
    # ------------------------------------------------------------------ #
    _t2 = time.perf_counter()
    solve_stats = {"total": 0, "feasible": 0}

    def oracle(idx: int) -> tuple[bool, list[int] | None]:
        """
        Feasibility check for radius unique_radii[idx].
        Returns (is_feasible, facility_list_or_None).
        """
        solve_stats["total"] += 1
        r = float(unique_radii[idx])
        facs = _greedy_cover(dm, n, p, r, pre_selected)
        if facs is not None:
            solve_stats["feasible"] += 1
        return (facs is not None), facs

    opt_radius, opt_facilities = _layered_search(
        oracle=oracle,
        unique_radii=unique_radii,
        L=max(1, layers),
        i_low=i_low,
        i_up=i_up,
        incumbent_radius=ub_radius,
        incumbent_facilities=ub_facilities,
    )

    logger.warning(
        "  p-center Phase 2 (L=%d layered search: %d total, %d feasible): %.2fs",
        layers, solve_stats["total"], solve_stats["feasible"],
        time.perf_counter() - _t2,
    )

    # ------------------------------------------------------------------ #
    # Phase 3 – Fill up to p facilities (furthest insertion on top of    #
    # the greedy result) and recompute the achieved radius.              #
    #                                                                    #
    # The greedy oracle returns the *minimum* covering set at opt_radius #
    # (possibly < p facilities).  Adding the remaining budget via        #
    # furthest insertion further reduces the maximum distance.           #
    # ------------------------------------------------------------------ #
    _t3 = time.perf_counter()
    if len(opt_facilities) < p:
        opt_facilities = _furthest_insertion(dm, n, p, opt_facilities)
        # Recompute actual radius achieved with the full p-facility set.
        opt_radius = float(np.max(dm.min_dist_to_set(opt_facilities)))
        logger.warning(
            "  p-center Phase 3 (fill to p=%d facilities, new radius=%.1f): %.2fs",
            p, opt_radius, time.perf_counter() - _t3,
        )

    min_dist = dm.min_dist_to_set(opt_facilities)
    assignment = dm.assign(opt_facilities)
    logger.warning(
        "  p-center Phase 3 (final assignment n=%d, facilities=%d): %.2fs",
        n, len(opt_facilities), time.perf_counter() - _t3,
    )

    return PCenterResult(
        facility_indices=sorted(opt_facilities),
        assignment=assignment,
        optimal_radius=opt_radius,
        coverage_stats=_compute_stats(demand, min_dist, opt_radius),
    )


# --------------------------------------------------------------------------- #
# Layered Search — Algorithm 1 (Kramer et al. 2018, Section 5.3)               #
# --------------------------------------------------------------------------- #

def _layered_search(
    oracle,
    unique_radii: np.ndarray,
    L: int,
    i_low: int,
    i_up: int,
    incumbent_radius: float,
    incumbent_facilities: list[int],
) -> tuple[float, list[int]]:
    """
    Recursively narrow [i_low, i_up] to find the minimum feasible index.

    Parameters
    ----------
    oracle : callable(int) -> (bool, list[int] | None)
        Tests feasibility at a given index; returns (feasible, facilities).
    unique_radii : sorted array of candidate radii.
    L : remaining number of layers (decremented at each recursive call).
    i_low : last known INFEASIBLE index (or -1 if none tested).
    i_up  : last known FEASIBLE index.
    incumbent_radius, incumbent_facilities : current best solution.

    Returns
    -------
    (optimal_radius, optimal_facilities)

    Complexity (Proposition 9):
        Total subproblems : O(L · (i_up - i_low)^(1/L))
        Feasible solves   : O(L)
    """
    # Base case: no interior points remain between i_low and i_up.
    if i_up <= i_low + 1:
        return incumbent_radius, incumbent_facilities

    span = i_up - i_low - 1   # number of untested indices in (i_low, i_up)

    # Step size for this layer (Proposition 9 in the paper).
    # L=1 → delta=1 (sequential scan); L=2 → O(√span); L=3 → O(span^(2/3)).
    if L <= 1:
        delta = 1
    else:
        delta = max(1, math.ceil(span ** ((L - 1) / L)))

    i = i_low
    is_feasible = False
    found_facilities: list[int] | None = None

    # Scan forward in steps of `delta` until a feasible radius is found
    # or we reach (or would exceed) i_up.
    while not is_feasible and i + delta < i_up:
        i += delta
        is_feasible, found_facilities = oracle(i)

    if is_feasible:
        # Update incumbent and recurse left to find a smaller feasible radius.
        new_radius = float(unique_radii[i])
        return _layered_search(
            oracle=oracle,
            unique_radii=unique_radii,
            L=max(1, L - 1),
            i_low=i - delta,
            i_up=i,
            incumbent_radius=new_radius,
            incumbent_facilities=found_facilities,  # type: ignore[arg-type]
        )
    else:
        # All sampled points were infeasible; recurse right.
        return _layered_search(
            oracle=oracle,
            unique_radii=unique_radii,
            L=max(1, L - 1),
            i_low=i,
            i_up=i_up,
            incumbent_radius=incumbent_radius,
            incumbent_facilities=incumbent_facilities,
        )


# --------------------------------------------------------------------------- #
# Furthest Insertion — initial upper bound heuristic (Section 6.3)             #
# --------------------------------------------------------------------------- #

def _furthest_insertion(
    dm: SparseDistanceMatrix,
    n: int,
    p: int,
    pre_selected: list[int],
) -> list[int]:
    """
    Furthest-insertion construction heuristic.

    Greedily opens p facilities by iteratively selecting the area that is
    farthest from all currently open facilities and opening it as the next
    facility.  This is the uncapacitated analogue of the construction phase
    described in Section 6.3 of Kramer et al. (2018) and gives a 2-approximation
    for the p-center objective.

    Parameters
    ----------
    dm : SparseDistanceMatrix
    n  : int — number of areas
    p  : int — total facilities to open (including pre_selected)
    pre_selected : list[int] — already-open facilities

    Returns
    -------
    list[int] — indices of p open facilities (no duplicates)
    """
    selected = list(pre_selected)
    needed = p - len(selected)
    if needed <= 0:
        return selected

    # Initialize min_dist (uint32) from pre-selected facilities.
    if selected:
        min_dist = dm.min_dist_to_set(selected)
        # Mark already-selected as distance 0 so they are never chosen again.
        for s in selected:
            min_dist[s] = 0
    else:
        # No pre-selected: open the first area as a seed, then add needed-1 more.
        first = 0
        selected.append(first)
        needed -= 1
        min_dist = dm.init_min_dist(first)
        # KEY: set seed distance to 0 to prevent it being re-selected.
        min_dist[first] = 0

    for _ in range(needed):
        # Area farthest from any open facility is the next facility candidate.
        # (Argmax over uint32 distances; ties broken by index order.)
        worst = int(np.argmax(min_dist))
        selected.append(worst)
        # KEY: mark as selected (distance 0) BEFORE updating neighbors,
        # so it is never picked again even if not in its own stored neighbors.
        min_dist[worst] = 0
        dm.update_min_dist(min_dist, worst)

    return selected


# --------------------------------------------------------------------------- #
# Greedy dominating-set feasibility oracle                                      #
# --------------------------------------------------------------------------- #

def _greedy_cover(
    dm: SparseDistanceMatrix,
    n: int,
    p: int,
    radius: float,
    pre_selected: list[int],
) -> list[int] | None:
    """
    Greedy dominating-set check: can p facilities cover all n areas within radius?

    At each step, the facility covering the most yet-uncovered areas is added.
    For the uncapacitated p-center this greedy is exact (if a feasible cover of
    size p exists, the greedy always finds one of size ≤ p).

    Pre-check (Section 6.1 of the paper): if no single stored neighbor of any
    uncovered area lies within radius, the instance is immediately infeasible.

    Parameters
    ----------
    dm           : SparseDistanceMatrix
    n            : total number of areas
    p            : maximum facilities allowed
    radius       : coverage radius to test
    pre_selected : facilities already open

    Returns
    -------
    list[int] of facility indices on success, None if infeasible.
    """
    uncovered = np.ones(n, dtype=bool)
    selected: list[int] = list(pre_selected)

    # Mark areas covered by pre-selected facilities.
    for fac in selected:
        covered = dm.covered_by(fac, radius)
        uncovered[covered] = False

    needed = p - len(selected)

    while np.any(uncovered):
        if needed <= 0:
            # Used all p facilities but areas remain uncovered.
            return None

        # Infeasibility pre-check (Section 6.1): count how many uncovered areas
        # each candidate facility can reach within radius.
        counts = dm.cover_counts(uncovered, radius)

        # Exclude already-selected facilities.
        for s in selected:
            counts[s] = -1

        best = int(np.argmax(counts))
        if counts[best] <= 0:
            # No candidate covers any remaining area → infeasible.
            return None

        selected.append(best)
        needed -= 1
        uncovered[dm.covered_by(best, radius)] = False

    return selected


# --------------------------------------------------------------------------- #
# Coverage statistics                                                            #
# --------------------------------------------------------------------------- #

def _compute_stats(
    demand: np.ndarray,
    min_dist: np.ndarray,
    radius: float,
) -> dict:
    f64 = min_dist.astype(np.float64)
    total_demand = float(np.sum(demand))
    covered_mask = min_dist <= int(radius)
    covered_demand = float(np.sum(demand[covered_mask]))
    return {
        "total_demand": total_demand,
        "covered_demand": round(covered_demand, 2),
        "coverage_pct": (
            round(covered_demand / total_demand * 100, 2) if total_demand > 0 else 0.0
        ),
        "optimal_radius_minutes": round(radius, 2),
        "avg_travel_time_minutes": (
            round(float(np.dot(demand, f64) / total_demand), 2)
            if total_demand > 0
            else 0.0
        ),
        "num_facilities": 0,  # filled by caller
    }
