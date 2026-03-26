"""
P-Median Optimization Algorithm.

The p-median problem locates p facilities to minimize the total
demand-weighted travel time from each census area to its nearest facility.

Algorithm: Greedy Add + 1-opt Exchange improvement heuristic.
  Phase 1 (Greedy Add): uses SparseDistanceMatrix.cost_reductions() to pick
    the candidate that gives the greatest demand-weighted cost reduction.
    Complexity: O(p × nnz) — fully vectorised.
  Phase 2 (Exchange): vectorised 1-opt swap.
    For each facility being considered for removal, the md_without array is
    built in O(n) using precomputed top-2 nearest facilities.  The best
    replacement is found with cost_reductions() in O(nnz).
    Complexity per iteration: O(p × (n + nnz))  ← no Python inner loop over
    candidates; replaces the old O(p × n × p × k) Python-loop approach.
    Skipped for n > 10 000.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from dataclasses import dataclass, field

from .sparse_matrix import MAX_DIST, SparseDistanceMatrix

logger = logging.getLogger(__name__)


@dataclass
class PMedianResult:
    facility_indices: list[int]
    assignment: list[int]
    total_cost: float
    coverage_stats: dict = field(default_factory=dict)


def solve(
    distance_matrix: SparseDistanceMatrix,
    demand: np.ndarray,
    p: int,
    capacity: np.ndarray | None = None,
    max_exchange_iters: int = 50,
    pre_selected: list[int] | None = None,
) -> PMedianResult:
    """
    Solve the p-median problem.

    Parameters
    ----------
    distance_matrix : SparseDistanceMatrix
    demand : np.ndarray, shape (n,)
    p : int
    max_exchange_iters : int
        Maximum 1-opt exchange iterations (skipped for n > 10 000).
    pre_selected : list[int] or None
        Indices locked as facilities (existing infrastructure).
    """
    dm = distance_matrix
    n = dm.n
    pre_selected = list(pre_selected or [])

    if p >= n:
        indices = list(range(n))
        md = dm.min_dist_to_set(indices)
        return PMedianResult(
            facility_indices=indices,
            assignment=list(range(n)),
            total_cost=0.0,
            coverage_stats=_compute_stats(demand, md),
        )

    # ------------------------------------------------------------------ #
    # Phase 1 – Greedy Add                                                 #
    # ------------------------------------------------------------------ #
    _t1 = time.perf_counter()
    selected: list[int] = list(pre_selected)

    if pre_selected:
        min_dist = dm.min_dist_to_set(pre_selected)
    else:
        first = int(np.argmax(demand))
        selected.append(first)
        min_dist = dm.init_min_dist(first)

    selected_set = set(selected)
    needed = p - len(selected)

    for _ in range(needed):
        reduction = dm.cost_reductions(demand, min_dist)
        reduction[list(selected_set)] = -np.inf
        best = int(np.argmax(reduction))
        selected.append(best)
        selected_set.add(best)
        dm.update_min_dist(min_dist, best)

    logger.warning("  p-median Phase 1 (greedy add, p=%d, n=%d): %.2fs",
                   p, n, time.perf_counter() - _t1)

    # ------------------------------------------------------------------ #
    # Phase 2 – Vectorised 1-opt Exchange (skipped for large n)           #
    # ------------------------------------------------------------------ #
    _t2 = time.perf_counter()
    pre_selected_set = set(pre_selected)

    if max_exchange_iters > 0 and n <= 10_000:
        min_dist = dm.min_dist_to_set(selected)
        current_cost = float(np.dot(demand, min_dist.astype(np.float64)))

        # Precompute nearest facility and second-nearest distance per area.
        nearest_fac, second_dist = _top2(dm, selected, min_dist)

        # ── Precompute per-entry arrays used inside cost_reductions ────────
        # These are constant across all Phase-2 iterations and facility swaps;
        # computing them here avoids ~900 redundant array allocations per run.
        demand_at_csc    = demand[dm.csc_row]          # (nnz,) float64
        nearest_at_csc   = nearest_fac[dm.csc_row]     # (nnz,) int32
        second_at_csc    = second_dist[dm.csc_row].astype(np.int64)  # (nnz,) int64
        csc_j_intp       = dm._csc_j.astype(np.intp)   # (nnz,) intp
        # md_at_csc_base is rebuilt whenever min_dist changes (swap accepted).
        md_at_csc_base   = min_dist[dm.csc_row].astype(np.int64)     # (nnz,) int64

        improved = True
        iterations = 0
        while improved and iterations < max_exchange_iters:
            improved = False
            iterations += 1

            for i, fac in enumerate(selected):
                if fac in pre_selected_set:
                    continue

                # Build md_without_i in O(nnz) using in-place modify/restore.
                # Areas whose nearest facility is `fac` fall back to second_dist.
                mask = nearest_at_csc == fac
                saved = md_at_csc_base[mask].copy()
                md_at_csc_base[mask] = second_at_csc[mask]

                cost_without = float(
                    np.dot(demand, np.where(nearest_fac == fac,
                                            second_dist, min_dist).astype(np.float64))
                )

                # Vectorised cost_reductions using precomputed arrays.
                impr     = np.maximum(0, md_at_csc_base - dm._csc_val_i64)
                weighted = demand_at_csc * impr
                reduction = np.bincount(csc_j_intp, weights=weighted, minlength=n)
                reduction[list(selected_set)] = -np.inf

                best_cand = int(np.argmax(reduction))

                # Restore md_at_csc_base before any early-continue.
                md_at_csc_base[mask] = saved

                if reduction[best_cand] <= 0:
                    continue

                new_cost = cost_without - float(reduction[best_cand])
                if new_cost < current_cost - 1e-9:
                    selected_set.discard(fac)
                    selected_set.add(best_cand)
                    selected[i] = best_cand
                    current_cost = new_cost
                    min_dist = dm.min_dist_to_set(selected)
                    nearest_fac, second_dist = _top2(dm, selected, min_dist)
                    # Refresh derived arrays after the swap.
                    nearest_at_csc  = nearest_fac[dm.csc_row]
                    second_at_csc   = second_dist[dm.csc_row].astype(np.int64)
                    md_at_csc_base  = min_dist[dm.csc_row].astype(np.int64)
                    improved = True
                    break

        logger.warning(
            "  p-median Phase 2 (vectorised 1-opt, %d iters, n=%d): %.2fs",
            iterations, n, time.perf_counter() - _t2,
        )
    else:
        logger.warning("  p-median Phase 2: skipped (n=%d > 10000)", n)

    # ------------------------------------------------------------------ #
    # Build assignment and stats                                           #
    # ------------------------------------------------------------------ #
    min_dist = dm.min_dist_to_set(selected)
    assignment = dm.assign(selected)

    return PMedianResult(
        facility_indices=sorted(selected),
        assignment=assignment,
        total_cost=float(np.dot(demand, min_dist.astype(np.float64))),
        coverage_stats=_compute_stats(demand, min_dist),
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _top2(
    dm: SparseDistanceMatrix,
    facilities: list[int],
    min_dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each area compute:
      nearest_fac[a]  – area index of the nearest facility
      second_dist[a]  – uint32 distance to the second-nearest facility

    Used to build md_without_i in O(n) during Phase 2 instead of
    re-running min_dist_to_set(p-1 facilities) each time.
    """
    n = dm.n
    # nearest_fac: which facility (by area index) is closest to each area.
    nearest_fac = np.array(dm.assign(facilities), dtype=np.int32)

    # second_dist: for areas where facility fac is nearest, second_dist[a]
    # is the distance to the closest OTHER facility.
    second_dist = np.full(n, MAX_DIST, dtype=np.uint32)

    for fac in facilities:
        rows, vals = dm.col_neighbors(fac)
        v32 = vals.astype(np.uint32)
        # For areas where fac is NOT the nearest, it is a candidate for second-nearest.
        not_nearest = nearest_fac[rows] != fac
        rows_nn = rows[not_nearest]
        vals_nn = v32[not_nearest]
        better = vals_nn < second_dist[rows_nn]
        second_dist[rows_nn[better]] = vals_nn[better]

    return nearest_fac, second_dist


def _compute_stats(demand: np.ndarray, min_dist: np.ndarray) -> dict:
    f64 = min_dist.astype(np.float64)
    total_demand = float(np.sum(demand))
    total_cost = float(np.dot(demand, f64))
    return {
        "total_demand": total_demand,
        "total_cost": round(total_cost, 2),
        "avg_travel_time_minutes": round(total_cost / total_demand, 2) if total_demand > 0 else 0.0,
        "max_travel_time_minutes": round(float(np.max(f64)), 2),
        "num_facilities": 0,  # filled by caller
    }
