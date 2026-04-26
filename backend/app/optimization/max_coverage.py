"""
Capacitated Maximum Coverage Location Problem (CMCLP).

Phase 0 â€“ Pre-existing centres:
    Absorb demand from existing facilities using nearest-first assignment
    up to their capacity.

Phase 1 â€“ Greedy Constructive (momentum-based alternating):
    For every candidate area i, compute a momentum indicator:

        momentum[i] = Î£_j  remaining[j] Â· speed(i,j)

    where speed(i,j) = euclidean_km(i,j) / (travel_time_min(i,j) / 60).

    Facilities are placed in alternating turns:
      â€¢ Max turn : open the unselected candidate with the highest momentum.
      â€¢ Min turn : open the unselected candidate with the lowest momentum.

    Both turns fill demand using nearest-first order (by travel time) up to
    cap_min.  Areas are taken whole and in ascending distance order; filling
    stops as soon as the facility load reaches cap_min.  Areas that would
    overflow cap_max are skipped.

    A candidate is considered only when its zone contains enough unassigned
    demand to meet cap_min.

    After each placement the momentum is updated incrementally in O(k)
    using the CSC structure â€” no full recomputation needed.

    The number of min turns per max turn is controlled by equity_ratio
    (default 1 â†’ strict 1-max / 1-min alternation).

Phase 2B â€“ Redundancy pruning:
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
from .max_coverage_phases import (
    _assign_zone_nearest_first,
    _compute_momentum,
    _compute_stats,
    _debug_coverage_snapshot,
    _phase1_greedy,
    _phase1_momentum,
    _phase2b_prune,
    _precompute_pair_speeds,
)

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
    p               : int â€” ignored (number of facilities is determined by cap_min).
    radius          : float â€” service radius in minutes.
    cap_min         : float â€” minimum load a facility must attract to open.
    cap_max         : float | None â€” maximum load per facility (None = no limit).
    pre_selected    : list[int] | None â€” indices fixed as existing facilities.
    equity_ratio    : int â€” min turns per max turn (1 = strict alternation).
    xy              : np.ndarray, shape (n, 2) â€” WGS-84 lon/lat per area.
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
            "CMCLP DEBUG: xy is None and dm.xy unavailable â€” momentum disabled, greedy fallback"
        )

    remaining = demand.astype(np.float64).copy()
    selected: list[int] = []
    selected_set: set[int] = set()
    fac_load: dict[int, float] = {}
    fac_remaining: dict[int, float] = {}
    assigned_to = np.full(n, -1, dtype=np.int32)
    fac_to_idx: dict[int, int] = {}

    # â”€â”€ Phase 0: Pre-existing facilities (nearest-first) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ #
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

    # â”€â”€ Phase 1: Greedy Constructive â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ #
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
                "CMCLP DEBUG: momentum degenerate (max=%.2e) â€” falling back to greedy",
                mom_max,
            )
    else:
        logger.warning(
            "CMCLP DEBUG: cannot compute momentum â€” xy=%s has_nan=%s",
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


    # â”€â”€ Phase 2B: Prune redundant facilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ #
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
            "CMCLP DEBUG Phase 2B done: %d â†’ %d new facilities (removed %d)",
            n_before_prune,
            len(new_opened),
            n_before_prune - len(new_opened),
        )

    # â”€â”€ Remove facilities below cap_min (defensive) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ #
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

    # â”€â”€ Coverage & assignment â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ #
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
