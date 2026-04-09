"""
Bump Hunter orchestration service.

Bridges the pure bump_hunter algorithm (app.optimization.bump_hunter) with the
database layer (models, session) and the API schema layer.

Responsibilities:
  - Load existing DB facilities in complete_existing mode.
  - Invoke the bump_hunter solver or use fixed_census_area_ids for reoptimization.
  - Run capacity-aware assignment via single_capacity_pass.
  - Build FacilityLocation / ServedAreaInfo response objects.
  - Persist results back to OptimizationScenario.result_stats.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.facility import Facility, FacilityStatus, FacilityType
from app.models.optimization import OptimizationScenario, ScenarioStatus
from app.optimization import bump_hunter_solve
from app.optimization.assignment import single_capacity_pass
from app.optimization.sparse_matrix import MAX_DIST, SparseDistanceMatrix
from app.schemas.optimization import (
    FacilityLocation,
    OptimizationMode,
    OptimizationRequest,
    ServedAreaInfo,
)

logger = logging.getLogger(__name__)


async def run_bump_hunter(
    payload: OptimizationRequest,
    db: AsyncSession,
    areas: list,
    demand: np.ndarray,
    distance_matrix: SparseDistanceMatrix,
    scenario_id: int,
    timing: dict,
    t0: float,
) -> None:
    """
    Full bump hunter pipeline: solve → assign → persist.

    Complete-existing mode:
        Existing DB facilities of the selected type are pre-loaded as fixed bump
        locations.  Their DB capacity limits the demand they can absorb.
        capacity=0 means the facility accepts no demand.

    Reoptimization mode (fixed_census_area_ids is set):
        The solver step is skipped; the provided census area IDs are used directly
        as facility locations.

    From-scratch mode:
        The gravity-score local-maxima solver identifies candidate bump locations.
        Each planned facility is limited by payload.max_capacity (∞ if not set).
    """
    area_id_to_idx = {a.id: i for i, a in enumerate(areas)}
    _t = time.perf_counter()

    # ── Load existing facilities (complete_existing mode, initial run only) ──
    existing_bump_indices: list[int] = []
    existing_bump_cap: dict[int, float] = {}  # bump_idx → raw DB capacity

    is_complete_existing = (
        payload.mode == OptimizationMode.COMPLETE_EXISTING
        and payload.facility_type
        and payload.fixed_census_area_ids is None
    )
    if is_complete_existing:
        try:
            fac_type = FacilityType(payload.facility_type)
        except ValueError:
            fac_type = None
        if fac_type:
            fac_result = await db.execute(
                select(Facility).where(
                    Facility.status == FacilityStatus.EXISTING,
                    Facility.facility_type == fac_type,
                    Facility.census_area_id.in_(list(area_id_to_idx.keys())),
                )
            )
            for f in fac_result.scalars().all():
                if f.census_area_id in area_id_to_idx:
                    idx = area_id_to_idx[f.census_area_id]
                    existing_bump_indices.append(idx)
                    existing_bump_cap[idx] = float(f.capacity) if f.capacity is not None else math.inf

    existing_set = set(existing_bump_indices)

    # ── Solver step ──
    if payload.fixed_census_area_ids is not None:
        # Reoptimization: skip solver, use the confirmed facility list directly.
        bump_indices = [
            area_id_to_idx[cid]
            for cid in payload.fixed_census_area_ids
            if cid in area_id_to_idx
        ]
        bh = None
    else:
        params = payload.parameters or {}
        k_nbrs_param = params.get("k_neighbors")
        k_vec_param  = params.get("k_vec")
        k_nbrs = int(k_nbrs_param) if k_nbrs_param else None
        k_vec  = int(k_vec_param)  if k_vec_param  else 500
        bh = await asyncio.to_thread(
            bump_hunter_solve,
            distance_matrix, demand,
            k_neighbors=k_nbrs,
            k_vec=k_vec,
        )
        bump_indices = bh.bump_indices

    timing["6_solver"] = time.perf_counter() - _t

    # Merge existing facility bumps (prepend so they appear first) with solver bumps.
    # Deduplicate: if the solver independently found an existing facility location,
    # keep it only once in the existing list (it carries the capacity constraint).
    all_bump_indices = existing_bump_indices + [b for b in bump_indices if b not in existing_set]

    # ── Capacity-aware assignment within service radius ──
    bh_radius = float(payload.service_radius)
    # Existing bumps: use their DB capacity.
    # Planned bumps: use payload.max_capacity (∞ when not specified).
    user_max_cap = float(payload.max_capacity) if payload.max_capacity is not None else math.inf
    bump_facility_caps: dict[int, float] = {
        bidx: existing_bump_cap.get(bidx, math.inf) if bidx in existing_set else user_max_cap
        for bidx in all_bump_indices
    }

    _t = time.perf_counter()
    assignments, _ = single_capacity_pass(
        distance_matrix, demand, all_bump_indices, bump_facility_caps, bh_radius
    )
    timing["7_assignment"] = time.perf_counter() - _t

    # Invert: bump_idx → [(area_idx, amount_assigned), ...]
    bump_to_served: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for area_idx, fac_assignments in assignments.items():
        for fac_idx, amount in fac_assignments:
            bump_to_served[fac_idx].append((area_idx, amount))

    # ── Build FacilityLocation objects with served_areas and aggregate stats ──
    locations: list[FacilityLocation] = []
    covered_ids: set[int] = set()
    _total_demand  = float(np.sum(demand))
    _total_covered = 0.0
    _weighted_tt   = 0.0
    _tt_demand     = 0.0
    _global_max_tt = 0.0

    for bidx in all_bump_indices:
        area = areas[bidx]
        served_pairs = bump_to_served.get(bidx, [])
        served_area_infos: list[ServedAreaInfo] = []
        covered_dem = 0.0
        valid_tts: list[float] = []

        for area_idx, amount in served_pairs:
            sa   = areas[area_idx]
            dist = distance_matrix.distance_time(area_idx, bidx)
            covered_dem    += amount
            _total_covered += amount
            covered_ids.add(sa.id)
            if dist < MAX_DIST:
                valid_tts.append(dist)
                _weighted_tt += amount * dist
                _tt_demand   += amount
                if dist > _global_max_tt:
                    _global_max_tt = dist
            if sa.x is not None and sa.y is not None:
                served_area_infos.append(
                    ServedAreaInfo(
                        census_area_id=sa.id,
                        area_code=sa.area_code,
                        x=sa.x,
                        y=sa.y,
                        assigned_demand=round(amount),
                        travel_time=round(dist, 1) if dist < MAX_DIST else None,
                    )
                )

        _is_existing = bidx in existing_set and is_complete_existing
        locations.append(FacilityLocation(
            census_area_id=area.id,
            area_code=area.area_code,
            name=area.name,
            x=area.x,
            y=area.y,
            covered_demand=round(covered_dem),
            assigned_areas=len(served_pairs),
            max_travel_time=float(max(valid_tts)) if valid_tts else None,
            is_existing=_is_existing,
            db_capacity=existing_bump_cap.get(bidx) if _is_existing else None,
            served_areas=served_area_infos,
        ))

    unassigned_areas = [
        {"census_area_id": a.id, "area_code": a.area_code,
         "name": a.name, "x": a.x, "y": a.y}
        for idx, a in enumerate(areas)
        if a.id not in covered_ids and float(demand[idx]) > 1e-9
    ]

    # ── Assemble stats ──
    bh_base_stats = bh.stats if bh is not None else {
        "num_bumps": len(all_bump_indices),
        "total_areas": len(areas),
        "total_demand": round(_total_demand),
    }
    stats = {
        **bh_base_stats,
        "covered_demand":          round(_total_covered),
        "uncovered_demand":        round(max(_total_demand - _total_covered, 0.0)),
        "coverage_pct":            round(_total_covered / _total_demand * 100, 2) if _total_demand > 0 else 0.0,
        "avg_travel_time_minutes": round(_weighted_tt / _tt_demand, 2) if _tt_demand > 0 else 0.0,
        "max_travel_time_minutes": round(_global_max_tt, 2),
        "_meta": {
            "facility_type":        payload.facility_type or "high_school",
            "mode":                 payload.mode.value,
            "target_population_id": payload.target_population_id,
            "scope_filters":        payload.scope_filters.model_dump() if payload.scope_filters else None,
        },
    }

    # ── Persist to DB ──
    scenario = await db.get(OptimizationScenario, scenario_id)
    scenario.status = ScenarioStatus.COMPLETED
    scenario.result_stats = {
        **stats,
        "_locations":       [loc.model_dump() for loc in locations],
        "_unassigned_areas": unassigned_areas,
    }
    scenario.completed_at = datetime.now(timezone.utc)
    await db.commit()

    timing["TOTAL"] = time.perf_counter() - t0
    logger.warning(
        "BG task completed  scenario=%d  model=bump_hunter  bumps=%d  %.2fs",
        scenario_id, len(locations), timing["TOTAL"],
    )
