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

    Both turns fill demand using nearest-first order (by travel time) up to
    cap_min.  Areas are taken whole and in ascending distance order; filling
    stops as soon as the facility load reaches cap_min.  Areas that would
    overflow cap_max are skipped.

    A candidate is considered only when its zone contains enough unassigned
    demand to meet cap_min.

    After each placement the momentum is updated incrementally in O(k)
    using the CSC structure — no full recomputation needed.

    The number of min turns per max turn is controlled by equity_ratio
    (default 1 → strict 1-max / 1-min alternation).

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


def _debug_coverage_snapshot(
    dm: "SparseDistanceMatrix",
    remaining: np.ndarray,
    r16: np.uint16,
    cap_min_f: float,
    cap_max_f: float,
    selected_set: set,
    tag: str,
) -> None:
    """
    Log a snapshot of the coverage landscape: how many candidates exceed cap_min,
    and the distribution of marginal coverage scores.  Used to diagnose zero-facility
    situations before placement starts.
    """
    scores = dm.marginal_coverage(remaining, radius=float(r16))
    effective = np.minimum(scores, cap_max_f) if cap_max_f < math.inf else scores.copy()
    if selected_set:
        effective[list(selected_set)] = -1.0
    n_valid = int(np.sum(effective >= cap_min_f - 1e-9))
    top5 = np.sort(effective)[::-1][:5].tolist()
    logger.info(
        "CMCLP DEBUG [%s]: remaining_demand=%.1f  candidates_above_cap_min=%d  "
        "top5_effective_scores=%s  cap_min=%.1f  cap_max=%s",
        tag,
        float(np.sum(remaining)),
        n_valid,
        [round(v, 1) for v in top5],
        cap_min_f,
        str(round(cap_max_f, 1)) if cap_max_f != math.inf else "inf",
    )


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
    Solve the Capacitated MCLP via momentum-based greedy + redundancy pruning.

    Parameters
    ----------
    distance_matrix : SparseDistanceMatrix
    demand          : np.ndarray, shape (n,)
    p               : int — ignored (number of facilities is determined by cap_min).
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

    # DEBUG: log all input parameters so the exact scenario is traceable.
    logger.info(
        "CMCLP DEBUG solve() start: n=%d total_demand=%.1f p=%d radius=%s "
        "cap_min=%.1f cap_max=%s pre_selected=%d xy_arg=%s dm.xy=%s",
        n, total_demand, p, radius,
        cap_min_f, str(cap_max_f) if cap_max_f != math.inf else "None",
        len(pre_selected),
        "yes" if xy is not None else "NO",
        "yes (shape %s)" % str(dm.xy.shape) if dm.xy is not None else "NO",
    )

    # Use coordinates from the distance matrix when not supplied explicitly.
    # Without xy, momentum cannot be computed and all facilities would be
    # placed in the densest cluster, which Phase 2B would then prune down to 1.
    if xy is None and dm.xy is not None and dm.xy.shape == (n, 2):
        xy = dm.xy
        logger.info("CMCLP DEBUG: xy auto-detected from dm.xy  shape=%s", str(xy.shape))
    elif xy is None:
        logger.warning(
            "CMCLP DEBUG: xy is None and dm.xy unavailable — momentum disabled, greedy fallback"
        )

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

    n_pre = len(pre_selected)
    logger.info(
        "CMCLP DEBUG Phase 0 done: %d pre-existing facilities, "
        "remaining_demand=%.1f (absorbed=%.1f)",
        n_pre,
        float(np.sum(remaining)),
        total_demand - float(np.sum(remaining)),
    )

    # ── Phase 1: Greedy Constructive ──────────────────────────────────────── #
    new_opened: set[int] = set()

    # Momentum requires valid xy coordinates and non-degenerate speed values.
    # Degenerate momentum (all-zero or NaN) causes argmax/argmin to always pick
    # the same index, producing two consecutive failures and zero facilities.
    use_momentum = False
    if xy is not None and xy.shape == (n, 2) and not np.any(np.isnan(xy)):
        csr_speeds, csc_speeds = _precompute_pair_speeds(dm, xy)
        momentum = _compute_momentum(dm, remaining, csr_speeds)
        finite_mom = momentum[np.isfinite(momentum)]
        mom_max = float(np.max(finite_mom)) if len(finite_mom) > 0 else 0.0
        logger.info(
            "CMCLP DEBUG momentum: n_finite=%d  nan_count=%d  "
            "min=%.4f  max=%.4f  mean=%.4f",
            len(finite_mom),
            int(np.sum(~np.isfinite(momentum))),
            float(np.min(finite_mom)) if len(finite_mom) > 0 else float("nan"),
            mom_max,
            float(np.mean(finite_mom)) if len(finite_mom) > 0 else float("nan"),
        )
        # Only use momentum if it has meaningful variation across candidates.
        if len(finite_mom) > 0 and mom_max > 1e-9:
            use_momentum = True
            logger.info("CMCLP DEBUG: momentum path ENABLED")
        else:
            logger.warning(
                "CMCLP DEBUG: momentum degenerate (max=%.2e) — falling back to greedy",
                mom_max,
            )
    else:
        logger.warning(
            "CMCLP DEBUG: cannot compute momentum — xy=%s has_nan=%s",
            "None" if xy is None else str(xy.shape),
            str(bool(np.any(np.isnan(xy)))) if xy is not None else "n/a",
        )

    # DEBUG: check marginal coverage before placement starts.
    _debug_coverage_snapshot(dm, remaining, r16, cap_min_f, cap_max_f, selected_set, "pre-Phase1")

    if use_momentum:
        new_opened, remaining, fac_load, assigned_to, fac_to_idx, selected, selected_set = \
            _phase1_momentum(
                dm, demand, remaining, momentum, csc_speeds,
                selected, selected_set, new_opened,
                fac_load, fac_remaining, fac_to_idx, assigned_to,
                r16, cap_min_f, cap_max_f, equity_ratio,
            )
        # If momentum path placed nothing despite valid candidates, fall back to greedy.
        if not new_opened:
            logger.warning(
                "CMCLP: momentum path placed 0 facilities, falling back to greedy"
            )
            use_momentum = False

    if not use_momentum:
        new_opened, remaining, fac_load, assigned_to, fac_to_idx, selected, selected_set = \
            _phase1_greedy(
                dm, demand, remaining,
                selected, selected_set, new_opened,
                fac_load, fac_remaining, fac_to_idx, assigned_to,
                r16, cap_min_f, cap_max_f,
            )

    logger.info(
        "CMCLP DEBUG Phase 1 done: %d new facilities, covered demand=%.1f / %.1f",
        len(new_opened),
        total_demand - float(np.sum(remaining)),
        total_demand,
    )


    # ── Phase 2B: Prune redundant facilities ──────────────────────────────── #
    # NOTE: Phase 2B is temporarily disabled to observe raw Phase 1 placement.
    n_before_prune = len(selected) - n_pre
    if len(selected) > n_pre:
        logger.info(
            "CMCLP DEBUG Phase 2B start: %d new facilities to evaluate for pruning",
         n_before_prune,
        )
        _phase2b_prune(
            dm, demand, remaining, selected, selected_set,
            fac_load, fac_remaining, fac_to_idx, assigned_to,
            n_pre, radius, cap_min_f, cap_max_f,
        )
        new_opened = set(selected[n_pre:])
        logger.info(
            "CMCLP DEBUG Phase 2B done: %d → %d new facilities (removed %d)",
            n_before_prune,
            len(new_opened),
            n_before_prune - len(new_opened),
        )

    # ── Remove facilities below cap_min (defensive) ───────────────────────── #
    below = [f for f in selected if fac_load.get(f, 0.0) < cap_min_f - 1e-9]
    if below:
        logger.warning(
            "CMCLP DEBUG cap_min filter: dropping %d facilities below cap_min=%.1f  "
            "loads=%s",
            len(below),
            cap_min_f,
            [round(fac_load.get(f, 0.0), 1) for f in below],
        )
    valid = [f for f in selected if fac_load.get(f, 0.0) >= cap_min_f - 1e-9]
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
    """
    s = int(dm.csc_ptr[assigned_area])
    e = int(dm.csc_ptr[assigned_area + 1])
    if s < e:
        momentum[dm.csc_row[s:e]] -= delta * csc_speeds[s:e]


def _find_peripheral_candidate(
    dm: "SparseDistanceMatrix",
    valid_mask: np.ndarray,
    momentum: np.ndarray,
    remaining: np.ndarray,
    r16: np.uint16,
    cap_min_f: float,
    cap_max_f: float,
) -> tuple[int, float, list]:
    """Exhaustive peripheral search for equity_ratio == 0 MIN turns.

    Scans all valid candidates in ascending momentum order until one satisfies
    cap_min.  This ensures a facility is placed in the most remote feasible zone
    even when the most isolated candidates are too sparse.

    Returns (best_j, load, taken).  best_j == -1 when no feasible candidate exists.
    """
    tried = np.zeros(dm.n, dtype=bool)
    best_j = -1
    load, taken = 0.0, []
    while True:
        search_mask = valid_mask & ~tried
        if not np.any(search_mask):
            break
        mom_masked = np.where(search_mask, momentum, np.inf)
        candidate = int(np.argmin(mom_masked))
        tried[candidate] = True
        load, taken = _zone_nearest_to_capmin(
            dm, remaining, candidate, r16, cap_min_f, cap_max_f
        )
        if load >= cap_min_f - 1e-9:
            best_j = candidate
            break
    return best_j, load, taken


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
    Each placed facility fills demand nearest-first up to cap_min.
    A turn is skipped (not counted) when no valid candidate is found.
    Stops when no valid candidate exists for any turn type.
    """
    turn_in_cycle = 0
    cycle_len = equity_ratio + 1
    consecutive_no_valid = 0
    turn_number = 0

    while consecutive_no_valid < 2:
        scores = dm.marginal_coverage(remaining, radius=float(r16))
        effective = np.minimum(scores, cap_max_f) if cap_max_f < math.inf else scores
        effective[list(selected_set)] = -1.0

        valid_mask = effective >= cap_min_f - 1e-9
        n_valid = int(np.sum(valid_mask))

        if not np.any(valid_mask):
            logger.info(
                "CMCLP DEBUG momentum loop exit: no valid candidates left "
                "(placed=%d, turn=%d)",
                len(new_opened), turn_number,
            )
            break

        is_max_turn = (turn_in_cycle < equity_ratio)
        turn_type = "MAX" if is_max_turn else "MIN"

        # ── Candidate selection ────────────────────────────────────────── #
        # equity_ratio == 0: pure peripheral mode.
        # Instead of stopping after 2 consecutive fill failures, scan ALL
        # valid candidates in ascending momentum order until one satisfies
        # cap_min.  This ensures a facility is placed in the most remote
        # feasible zone even when the most isolated candidates are too sparse.
        if not is_max_turn and equity_ratio == 0:
            best_j, load, taken = _find_peripheral_candidate(
                dm, valid_mask, momentum, remaining, r16, cap_min_f, cap_max_f
            )

            if best_j == -1:
                logger.info(
                    "CMCLP DEBUG equity exhaustive search: all valid candidates "
                    "tried, none satisfies cap_min=%.1f  (placed=%d, turn=%d)",
                    cap_min_f, len(new_opened), turn_number,
                )
                break  # no feasible peripheral facility exists — stop Phase 1

            logger.debug(
                "  CMCLP Phase 1 (MIN-exhaustive): opened facility %d  load=%.1f",
                best_j, load,
            )
        else:
            # Standard MAX or MIN turn (equity_ratio > 0).
            if is_max_turn:
                mom_masked = np.where(valid_mask, momentum, -np.inf)
                best_j = int(np.argmax(mom_masked))
            else:
                mom_masked = np.where(valid_mask, momentum, np.inf)
                best_j = int(np.argmin(mom_masked))

            best_score = float(effective[best_j])
            best_mom   = float(momentum[best_j])

            load, taken = _zone_nearest_to_capmin(
                dm, remaining, best_j, r16, cap_min_f, cap_max_f
            )

            if load < cap_min_f - 1e-9:
                consecutive_no_valid += 1
                logger.warning(
                    "CMCLP DEBUG turn %d (%s): fill FAILED  j=%d score=%.1f mom=%.4f "
                    "fill_load=%.1f cap_min=%.1f n_taken=%d n_valid=%d  "
                    "consecutive_fails=%d",
                    turn_number, turn_type, best_j, best_score, best_mom,
                    load, cap_min_f, len(taken), n_valid, consecutive_no_valid,
                )
                turn_in_cycle = (turn_in_cycle + 1) % cycle_len
                turn_number += 1
                continue

        consecutive_no_valid = 0
        turn_in_cycle = (turn_in_cycle + 1) % cycle_len
        turn_number += 1

        fac_pos = len(selected)
        fac_to_idx[best_j] = fac_pos
        selected.append(best_j)
        selected_set.add(best_j)
        new_opened.add(best_j)
        fac_load[best_j] = load
        fac_remaining[best_j] = max(0.0, cap_max_f - load)

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
    """Standard demand-coverage greedy with nearest-first fill (used when xy is unavailable)."""
    iteration = 0
    while True:
        scores = dm.marginal_coverage(remaining, radius=float(r16))
        effective = np.minimum(scores, cap_max_f) if cap_max_f < math.inf else scores
        if selected_set:
            effective[list(selected_set)] = -1.0

        best_j = int(np.argmax(effective))
        best_score = float(effective[best_j])

        if best_score < cap_min_f - 1e-9:
            logger.info(
                "CMCLP DEBUG greedy exit: best_score=%.1f < cap_min=%.1f  "
                "placed=%d  iteration=%d",
                best_score, cap_min_f, len(new_opened), iteration,
            )
            break

        load, taken = _zone_nearest_to_capmin(dm, remaining, best_j, r16, cap_min_f, cap_max_f)
        if load < cap_min_f - 1e-9:
            logger.warning(
                "CMCLP DEBUG greedy exit: fill FAILED  j=%d score=%.1f "
                "fill_load=%.1f cap_min=%.1f n_taken=%d  placed=%d  iteration=%d",
                best_j, best_score, load, cap_min_f, len(taken), len(new_opened), iteration,
            )
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

        iteration += 1

    return new_opened, remaining, fac_load, assigned_to, fac_to_idx, selected, selected_set


# ─────────────────────────────────────────────────────────────────────────── #
# Nearest-first fill up to cap_min                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def _zone_nearest_to_capmin(
    dm: SparseDistanceMatrix,
    remaining: np.ndarray,
    j: int,
    r16: np.uint16,
    cap_min_f: float,
    cap_max_f: float,
) -> tuple[float, list[int]]:
    """
    Nearest-first fill for facility j, stopping once cap_min is reached.

    Areas within the service radius are sorted by travel time ascending.
    Whole areas are taken in that order; an area is skipped only if adding
    it would exceed cap_max.  Filling stops as soon as the accumulated load
    reaches cap_min.

    Returns (total_load, list_of_assigned_area_indices).
    """
    rows, vals = dm.col_neighbors(j)
    in_zone = vals <= r16
    if not np.any(in_zone):
        return 0.0, []

    zone_rows = rows[in_zone]
    zone_vals = vals[in_zone]
    order = np.argsort(zone_vals, kind="stable")
    zone_rows = zone_rows[order]

    load = 0.0
    taken: list[int] = []
    target = cap_min_f  # stop once this threshold is reached

    for i in zone_rows:
        if remaining[i] <= 1e-9:
            continue
        dem = float(remaining[i])
        if load + dem > cap_max_f:
            continue  # skip areas that would overflow hard capacity ceiling
        load += dem
        taken.append(int(i))
        if load >= target - 1e-9:
            break  # minimum capacity satisfied; stop

    return load, taken


def _can_reassign_facility(
    dm: "SparseDistanceMatrix",
    fac: int,
    demand: np.ndarray,
    served: list[int],
    fac_remaining: dict[int, float],
    selected_set: set[int],
    radius: float,
    cap_max_f: float,
) -> tuple[bool, dict[int, int]]:
    """Check whether all areas served by fac can be feasibly reassigned to neighbours.

    Areas are processed in descending demand order so that the heaviest areas
    claim capacity first.  Remaining capacity is tracked speculatively without
    mutating fac_remaining.

    Returns (feasible, reassign) where reassign maps area -> alternative facility
    (or -1 for zero-demand areas).  If feasible is False, reassign is partial.
    """
    temp_remaining_cap: dict[int, float] = dict(fac_remaining)
    reassign: dict[int, int] = {}
    feasible = True

    for area in sorted(served, key=lambda a: float(demand[a]), reverse=True):
        dem = float(demand[area])
        if dem <= 1e-9:
            reassign[area] = -1
            continue

        best_alt = -1
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
                best_alt = g

        if best_alt == -1:
            feasible = False
            break

        reassign[area] = best_alt
        temp_remaining_cap[best_alt] -= dem

    return feasible, reassign


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
    (lightest first).  The loop repeats until no further removal is possible.
    """
    changed = True
    while changed:
        changed = False

        area_fac: dict[int, int] = {}
        fac_areas: dict[int, list[int]] = {}
        for area in range(dm.n):
            pos = int(assigned_to[area])
            if pos < 0 or pos >= len(selected):
                continue
            fac = selected[pos]
            area_fac[area] = fac
            fac_areas.setdefault(fac, []).append(area)

        candidates = sorted(
            [f for f in selected[n_pre:] if fac_load.get(f, 0.0) >= 0.0],
            key=lambda f: fac_load.get(f, 0.0),
        )

        for fac in candidates:
            served = fac_areas.get(fac, [])
            if not served:
                _remove_facility(fac, selected, selected_set, fac_load,
                                 fac_remaining, fac_to_idx, assigned_to)
                changed = True
                break

            feasible, reassign = _can_reassign_facility(
                dm, fac, demand, served, fac_remaining, selected_set, radius, cap_max
            )

            if not feasible:
                continue

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
            break
    

def _remove_facility(
    fac: int,
    selected: list[int],
    selected_set: set[int],
    fac_load: dict[int, float],
    fac_remaining: dict[int, float],
    fac_to_idx: dict[int, int],
    assigned_to: np.ndarray,
) -> None:
    """Remove fac from selected and rebuild fac_to_idx."""
    old_pos = fac_to_idx[fac]
    selected.pop(old_pos)
    selected_set.discard(fac)
    fac_load.pop(fac, None)
    fac_remaining.pop(fac, None)
    fac_to_idx.pop(fac, None)

    for new_pos in range(old_pos, len(selected)):
        f = selected[new_pos]
        fac_to_idx[f] = new_pos

    # Capture both masks BEFORE any modification.
    # Reassigned areas already point to another facility at position > old_pos.
    # If we decrement first, they land on old_pos and the cleanup below would
    # incorrectly erase them — causing a cascading removal of all facilities.
    mask_stranded = assigned_to == old_pos   # truly lost (facility removed, not reassigned)
    mask_shift    = assigned_to > old_pos    # need index correction after pop
    assigned_to[mask_shift] -= 1
    assigned_to[mask_stranded] = -1


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
