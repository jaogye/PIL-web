"""
Capacitated Maximum Coverage Location Problem with Closest Assignment Constraints
(CMCLP-CAC) — GRASP heuristic.

Reference: Slot, A. "Capacitated Maximum Covering Location Problem with Closest
Assignment Constraints", Tilburg University.

Algorithm:
  Phase 0  Pre-existing facilities: fixed open.
  Phase 1  GRASP construction (_N_GRASP_ITER independent runs, best kept):
           Proportional greedy with exponential bias α=0.05 (pgreedy_exp).
           At each step, each closed candidate j is scored by the uncovered
           demand within its service radius (stored pairs); candidates are
           ranked descending and selected with probability ∝ exp(−α · rank).
           Exactly p − |pre_selected| new facilities are placed per run.
  Phase 2  First Improvement local search per GRASP run:
           For each open non-fixed facility fl, a neighbourhood of closed
           candidates fe is built from stored neighbors of areas in fl's zone.
           The top _LS_CANDIDATES (by uncovered marginal gain) are evaluated
           via full CAC re-assignment.  The first improving swap is accepted
           and the search restarts.  Terminates when no improvement is found.
"""

from __future__ import annotations

import logging
import math
import numpy as np
from dataclasses import dataclass, field

from .sparse_matrix import MAX_DIST, SparseDistanceMatrix

logger = logging.getLogger(__name__)

_ALPHA = 0.05         # pgreedy_exp exponential bias parameter
_N_GRASP_ITER = 5     # independent GRASP runs (construction + LS); best kept
_MAX_LS_ROUNDS = 20   # maximum LS passes per run
_LS_CANDIDATES = 25   # closed-facility candidates to evaluate per open facility


@dataclass
class MaxCoverageResult:
    facility_indices: list[int]
    assignment: list[int]
    covered_demand: float
    total_demand: float
    coverage_pct: float
    coverage_stats: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────── #
# Public API                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def solve(
    distance_matrix: SparseDistanceMatrix,
    demand: np.ndarray,
    p: int,
    radius: float,
    cap_min: float = 0.0,
    cap_max: float | None = None,
    pre_selected: list[int] | None = None,
    **kwargs,
) -> MaxCoverageResult:
    """
    Solve CMCLP-CAC via GRASP (pgreedy_exp construction + First Improvement LS).

    Parameters
    ----------
    distance_matrix : SparseDistanceMatrix
    demand          : (n,) float64
    p               : total facilities to open (pre_selected count toward p)
    radius          : service radius in minutes
    cap_min         : minimum demand a facility must attract (else it is dropped)
    cap_max         : maximum demand per facility (None = uncapacitated)
    pre_selected    : fixed existing facility indices
    """
    dm = distance_matrix
    n = dm.n
    r16 = np.uint16(min(int(radius), MAX_DIST))
    cap_max_f = float(cap_max) if cap_max is not None else math.inf
    cap_min_f = float(cap_min)
    total_demand = float(np.sum(demand))
    pre_sel = list(pre_selected or [])
    demand_f = demand.astype(np.float64)

    rng = np.random.default_rng()
    p_new = max(0, p - len(pre_sel))
    n_iters = _N_GRASP_ITER if p_new > 0 else 1

    # Precompute zones once: sorted (rows, dists) within radius for every facility.
    # Avoids repeated O(n) col_neighbors_full calls inside the hot LS loop.
    zones = _precompute_zones(dm, n, radius)

    best_covered = -1.0
    best_opened: list[int] = []

    for iteration in range(n_iters):
        opened = _pgreedy_exp_construction(
            dm, demand_f, p_new, radius, r16, zones, pre_sel, _ALPHA, rng
        )
        opened, cov_dem = _first_improvement_ls(
            dm, demand_f, opened, radius, r16, zones, pre_sel, cap_max_f
        )
        logger.debug(
            "GRASP iter %d: %d facilities, covered=%.1f", iteration, len(opened), cov_dem
        )
        if cov_dem > best_covered:
            best_covered = cov_dem
            best_opened = opened[:]

    # Final assignment with best solution
    covered_demand, fac_load, covered_mask, nearest_fac = _cac_assignment(
        demand_f, best_opened, zones, cap_max_f
    )

    # Drop facilities below cap_min
    valid_opened = [j for j in best_opened if fac_load.get(j, 0.0) >= cap_min_f - 1e-9]
    if len(valid_opened) < len(best_opened):
        logger.warning(
            "CMCLP: dropped %d facilities below cap_min", len(best_opened) - len(valid_opened)
        )
        covered_demand, fac_load, covered_mask, nearest_fac = _cac_assignment(
            demand_f, valid_opened, zones, cap_max_f
        )

    coverage_pct = round(covered_demand / total_demand * 100, 2) if total_demand > 0 else 0.0
    assignment = np.where(covered_mask, nearest_fac, np.int32(-1)).tolist()

    logger.info(
        "CMCLP GRASP: %d facilities, covered=%.1f/%.1f (%.1f%%)",
        len(valid_opened), covered_demand, total_demand, coverage_pct,
    )

    return MaxCoverageResult(
        facility_indices=sorted(valid_opened),
        assignment=assignment,
        covered_demand=covered_demand,
        total_demand=total_demand,
        coverage_pct=coverage_pct,
        coverage_stats=_compute_stats(
            demand_f, dm, valid_opened, radius, covered_mask, fac_load, cap_min_f, cap_max_f
        ),
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Zone precomputation                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def _precompute_zones(
    dm: SparseDistanceMatrix,
    n: int,
    radius: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    For each facility j, compute (rows, dists) for all areas within radius
    (stored + estimated pairs), sorted by distance ascending.  Called once.
    """
    zones: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(n):
        rows, dists = dm.col_neighbors_full(j, radius)
        if len(rows) > 0:
            order = np.argsort(dists, kind="stable")
            zones.append((rows[order], dists[order]))
        else:
            zones.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)))
    return zones


# ─────────────────────────────────────────────────────────────────────────── #
# CAC assignment                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def _cac_assignment(
    demand: np.ndarray,
    opened_list: list[int],
    zones: list[tuple[np.ndarray, np.ndarray]],
    cap_max_f: float,
) -> tuple[float, dict[int, float], np.ndarray, np.ndarray]:
    """
    Closest Assignment Constraint assignment with capacity.

    Each area is assigned to its nearest open facility within radius.
    Fills the facility nearest-first up to cap_max_f; areas beyond capacity
    are uncovered.

    Returns
    -------
    covered_demand : float
    fac_load       : dict[int, float]
    covered_mask   : (n,) bool
    nearest_fac    : (n,) int32 — nearest open facility index (-1 if none in radius)
    """
    n = len(demand)
    nearest_dist = np.full(n, float(MAX_DIST))
    nearest_fac = np.full(n, -1, dtype=np.int32)

    for j in opened_list:
        rows, dists = zones[j]
        if len(rows) == 0:
            continue
        better = dists < nearest_dist[rows]
        nearest_dist[rows[better]] = dists[better]
        nearest_fac[rows[better]] = j

    # Group areas by nearest facility using argsort to avoid per-facility n-scan
    valid_areas = np.where(nearest_fac >= 0)[0]
    if len(valid_areas) == 0:
        return 0.0, {j: 0.0 for j in opened_list}, np.zeros(n, dtype=bool), nearest_fac

    valid_facs_arr = nearest_fac[valid_areas]
    sort_order = np.argsort(valid_facs_arr, kind="stable")
    sorted_areas = valid_areas[sort_order]
    sorted_facs = valid_facs_arr[sort_order]

    diff = np.diff(sorted_facs)
    boundaries = np.concatenate([[0], np.where(diff != 0)[0] + 1, [len(sorted_facs)]])

    covered_mask = np.zeros(n, dtype=bool)
    fac_load: dict[int, float] = {j: 0.0 for j in opened_list}
    covered_demand = 0.0

    for k in range(len(boundaries) - 1):
        start, end = int(boundaries[k]), int(boundaries[k + 1])
        j = int(sorted_facs[start])
        idx = sorted_areas[start:end]
        dists_j = nearest_dist[idx]
        demands_j = demand[idx]

        # Sort within facility by distance (CAC: nearest-first fill)
        inner_order = np.argsort(dists_j, kind="stable")
        sorted_demands = demands_j[inner_order]

        if cap_max_f == math.inf:
            covered_mask[idx] = True
            load = float(np.sum(sorted_demands))
        else:
            cumsum = np.cumsum(sorted_demands)
            kk = int(np.searchsorted(cumsum, cap_max_f + 1e-9, side="left"))
            if kk > 0:
                covered_mask[idx[inner_order[:kk]]] = True
                load = float(cumsum[kk - 1])
            else:
                load = 0.0

        fac_load[j] = load
        covered_demand += load

    return covered_demand, fac_load, covered_mask, nearest_fac


# ─────────────────────────────────────────────────────────────────────────── #
# GRASP construction — pgreedy_exp                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def _pgreedy_exp_construction(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    p_new: int,
    radius: float,
    r16: np.uint16,
    zones: list[tuple[np.ndarray, np.ndarray]],
    pre_sel: list[int],
    alpha: float,
    rng: np.random.Generator,
) -> list[int]:
    """
    Proportional greedy construction with exponential bias (pgreedy_exp).

    At each step, marginal demand coverage is computed for all closed
    candidates using the stored-pairs-only marginal_coverage (O(nnz), vectorised).
    Candidates are ranked by score descending, then one is selected with
    probability ∝ exp(−α · rank).  nearest_dist is maintained incrementally.
    """
    n = len(demand)
    opened = list(pre_sel)
    opened_set = set(opened)

    nearest_dist = np.full(n, float(MAX_DIST))
    for j in pre_sel:
        rows, dists = zones[j]
        if len(rows):
            better = dists < nearest_dist[rows]
            nearest_dist[rows[better]] = dists[better]

    for _ in range(p_new):
        uncov = np.where(nearest_dist <= radius, 0.0, demand)
        gains = dm.marginal_coverage(uncov, radius)
        if opened_set:
            gains[list(opened_set)] = 0.0

        nonzero = np.where(gains > 1e-9)[0]
        if len(nonzero) == 0:
            remaining = [j for j in range(n) if j not in opened_set]
            if not remaining:
                break
            chosen = remaining[int(rng.integers(0, len(remaining)))]
        else:
            order = np.argsort(-gains[nonzero])
            ranked = nonzero[order]
            scores = np.exp(-alpha * np.arange(len(ranked), dtype=np.float64))
            scores /= scores.sum()
            chosen = int(ranked[int(rng.choice(len(ranked), p=scores))])

        opened.append(chosen)
        opened_set.add(chosen)

        rows, dists = zones[chosen]
        if len(rows):
            better = dists < nearest_dist[rows]
            nearest_dist[rows[better]] = dists[better]

    return opened


# ─────────────────────────────────────────────────────────────────────────── #
# First Improvement local search                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def _first_improvement_ls(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    opened: list[int],
    radius: float,
    r16: np.uint16,
    zones: list[tuple[np.ndarray, np.ndarray]],
    pre_sel: list[int],
    cap_max_f: float,
) -> tuple[list[int], float]:
    """
    First Improvement local search over (fl, fe) swap pairs.

    For each open non-pre-selected facility fl, a neighbourhood of closed
    candidates fe is built from the stored neighbors of areas in fl's zone.
    The top _LS_CANDIDATES (scored by uncovered marginal gain) are evaluated
    via full CAC re-assignment.  The first improving swap is accepted and the
    search restarts.  Stops when no improving swap exists or _MAX_LS_ROUNDS
    passes have been completed.

    Returns the final opened list and its covered demand.
    """
    pre_sel_set = set(pre_sel)
    opened_set = set(opened)

    current_covered, _, _, _ = _cac_assignment(demand, list(opened_set), zones, cap_max_f)

    for _round in range(_MAX_LS_ROUNDS):
        improved = False

        # Cache opened_list once per round for repeated use
        opened_list = sorted(opened_set)
        opened_arr = np.array(opened_list, dtype=np.int32)

        # Uncovered marginal gains for candidate scoring (stored pairs, O(nnz))
        md = dm.min_dist_to_set(opened_list)
        uncov = np.where(md > r16, demand, 0.0)
        round_gains = dm.marginal_coverage(uncov, radius)
        round_gains[opened_list] = 0.0

        for fl in opened_list:
            if fl in pre_sel_set:
                continue

            rows_fl, _ = zones[fl]
            if len(rows_fl) == 0:
                continue

            sample = rows_fl[:20]
            parts = [dm.csr_col[dm.csr_ptr[i]:dm.csr_ptr[i + 1]] for i in sample]
            if not parts:
                continue
            all_nbrs = np.concatenate(parts)
            fe_candidates = np.setdiff1d(np.unique(all_nbrs), opened_arr)
            if len(fe_candidates) == 0:
                continue

            if len(fe_candidates) > _LS_CANDIDATES:
                cand_scores = round_gains[fe_candidates]
                top_k = min(_LS_CANDIDATES, len(fe_candidates))
                top_idx = np.argpartition(-cand_scores, top_k - 1)[:top_k]
                fe_candidates = fe_candidates[top_idx]

            base_opened = [j for j in opened_set if j != fl]

            for fe in fe_candidates:
                new_cov, _, _, _ = _cac_assignment(
                    demand, base_opened + [int(fe)], zones, cap_max_f
                )
                if new_cov > current_covered + 1e-6:
                    opened_set.discard(fl)
                    opened_set.add(int(fe))
                    logger.debug(
                        "LS: swap fl=%d → fe=%d  coverage %.1f → %.1f",
                        fl, int(fe), current_covered, new_cov,
                    )
                    current_covered = new_cov
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return list(opened_set), current_covered


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
