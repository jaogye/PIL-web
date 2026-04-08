"""
REST endpoints for facility location optimization.

POST /optimization/run      – Submit and immediately run an optimization scenario.
GET  /optimization/         – List all scenarios.
GET  /optimization/{id}     – Get a specific scenario with results.
DELETE /optimization/{id}   – Delete a scenario.
"""

import asyncio
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session_factory
from app.dependencies import get_db, get_db_name
from app.models.census import CensusArea
from app.models.facility import Facility, FacilityStatus, FacilityType, FacilityTypeLookup
from app.models.target_population import CensusAreaPopulation
from app.optimization.sparse_matrix import MAX_DIST, SparseDistanceMatrix
from app.models.optimization import (
    ModelType,
    OptimizationResult,
    OptimizationScenario,
    ScenarioStatus,
)
from app.optimization import (
    bump_hunter_solve,
    max_coverage_solve,
    p_center_solve,
    p_median_solve,
    rebalancing_solve,
)
from app.schemas.optimization import (
    FacilityLocation,
    FacilityCapacityInfo,
    OptimizationMode,
    OptimizationRequest,
    OptimizationResponse,
    RebalancingRequest,
    RebalancingResponse,
    RebalancingTransferInfo,
    ScenarioSummary,
    ServedAreaInfo,
    UnassignedAreaInfo,
)

router = APIRouter(prefix="/optimization", tags=["optimization"])


async def _load_demand(
    db: AsyncSession,
    areas: list[CensusArea],
    target_population_id: int | None,
    facility_type_code: str | None,
) -> np.ndarray:
    """Return a demand vector aligned with `areas`.

    Resolves the target population in this priority order:
      1. Explicit target_population_id from the request.
      2. Default target_population_id for the given facility_type.
      3. The 'all_ages' group as final fallback.
    """
    tp_id = target_population_id

    if tp_id is None and facility_type_code:
        ft_row = await db.get(FacilityTypeLookup, facility_type_code)
        if ft_row is not None:
            tp_id = ft_row.default_target_population_id

    if tp_id is None:
        # Fallback: use all_ages
        from app.models.target_population import TargetPopulation
        tp_result = await db.execute(
            select(TargetPopulation).where(TargetPopulation.code == "all_ages")
        )
        tp = tp_result.scalar_one_or_none()
        if tp:
            tp_id = tp.id

    area_ids = [a.id for a in areas]
    pop_result = await db.execute(
        select(CensusAreaPopulation).where(
            CensusAreaPopulation.census_area_id.in_(area_ids),
            CensusAreaPopulation.target_population_id == tp_id,
        )
    )
    pop_map: dict[int, float] = {r.census_area_id: r.population for r in pop_result.scalars().all()}
    return np.array([pop_map.get(a.id, 0.0) for a in areas], dtype=np.float64)


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_optimization(
    payload: OptimizationRequest,
    db: AsyncSession = Depends(get_db),
    db_name: str = Depends(get_db_name),
):
    """Submit an optimization scenario.  Returns immediately; runs in background."""
    scenario = OptimizationScenario(
        name=payload.name,
        description=payload.description,
        model_type=ModelType(payload.model_type.value),
        p_facilities=payload.p_facilities if payload.p_facilities is not None else 1,
        service_radius=payload.service_radius,
        scope_filters=payload.scope_filters.model_dump() if payload.scope_filters else None,
        parameters={
            **(payload.parameters or {}),
            "_facility_type": payload.facility_type,
            "_mode": payload.mode.value,
        },
        status=ScenarioStatus.RUNNING,
    )
    db.add(scenario)
    await db.flush()
    scenario_id = scenario.id
    scenario_name = scenario.name
    await db.commit()

    # Keep only the 8 most recent scenarios; delete older ones (cascade removes their results).
    _MAX_SCENARIOS = 8
    old_result = await db.execute(
        select(OptimizationScenario.id)
        .order_by(OptimizationScenario.created_at.desc())
        .offset(_MAX_SCENARIOS)
    )
    old_ids = [row[0] for row in old_result.all()]
    if old_ids:
        await db.execute(
            delete(OptimizationScenario).where(OptimizationScenario.id.in_(old_ids))
        )
        await db.commit()

    asyncio.create_task(_run_optimization_bg(scenario_id, payload, db_name))

    return {"scenario_id": scenario_id, "name": scenario_name, "status": "running"}


@router.get("/", response_model=list[ScenarioSummary])
async def list_scenarios(db: AsyncSession = Depends(get_db)):
    """Return a summary list of all optimization scenarios."""
    result = await db.execute(
        select(OptimizationScenario).order_by(OptimizationScenario.created_at.desc())
    )
    scenarios = result.scalars().all()
    return [
        ScenarioSummary(
            id=s.id,
            name=s.name,
            model_type=s.model_type,
            p_facilities=s.p_facilities,
            status=s.status.value,
            created_at=s.created_at.isoformat(),
            stats=s.result_stats,
        )
        for s in scenarios
    ]


@router.get("/{scenario_id}", response_model=OptimizationResponse)
async def get_scenario(scenario_id: int, db: AsyncSession = Depends(get_db)):
    """Return a scenario with all its facility locations."""
    scenario = await db.get(OptimizationScenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    if scenario.status != ScenarioStatus.COMPLETED:
        return OptimizationResponse(
            scenario_id=scenario.id,
            name=scenario.name,
            model_type=scenario.model_type,
            p_facilities=scenario.p_facilities,
            service_radius=scenario.service_radius,
            status=scenario.status.value,
            facility_locations=[],
            stats={"error_message": scenario.error_message}
            if scenario.status == ScenarioStatus.FAILED
            else {},
        )

    # Return full cached response stored by the background task.
    if scenario.result_stats and "_locations" in scenario.result_stats:
        raw_stats = {
            k: v
            for k, v in scenario.result_stats.items()
            if not k.startswith("_") or k == "_meta"
        }
        locations = [
            FacilityLocation.model_validate(loc)
            for loc in scenario.result_stats["_locations"]
        ]
        unassigned = [
            UnassignedAreaInfo.model_validate(u)
            for u in scenario.result_stats.get("_unassigned_areas", [])
        ]
        return OptimizationResponse(
            scenario_id=scenario.id,
            name=scenario.name,
            model_type=scenario.model_type,
            p_facilities=scenario.p_facilities,
            service_radius=scenario.service_radius,
            status=scenario.status.value,
            facility_locations=locations,
            stats=raw_stats,
            unassigned_areas=unassigned,
        )

    # Fallback: reconstruct from DB (pre-async scenarios without _locations).
    result = await db.execute(
        select(OptimizationResult, CensusArea)
        .join(CensusArea, OptimizationResult.census_area_id == CensusArea.id)
        .where(OptimizationResult.scenario_id == scenario_id)
    )
    rows = result.all()
    locations = [
        FacilityLocation(
            census_area_id=area.id,
            area_code=area.area_code,
            name=area.name,
            x=area.x,
            y=area.y,
            covered_demand=res.covered_demand,
            assigned_areas=res.assigned_areas,
            max_travel_time=res.max_travel_time,
        )
        for res, area in rows
    ]
    return OptimizationResponse(
        scenario_id=scenario.id,
        name=scenario.name,
        model_type=scenario.model_type,
        p_facilities=scenario.p_facilities,
        service_radius=scenario.service_radius,
        status=scenario.status.value,
        facility_locations=locations,
        stats=scenario.result_stats or {},
    )


@router.post("/{scenario_id}/rebalance", response_model=RebalancingResponse)
async def rebalance_scenario(
    scenario_id: int,
    payload: RebalancingRequest,
    db: AsyncSession = Depends(get_db),
    db_name: str = Depends(get_db_name),
):
    """
    Run the capacity rebalancing algorithm on an existing completed scenario.

    Identifies over-served facilities (capacity > assigned demand) and
    under-served ones (demand > capacity), then proposes capacity transfers
    that maximise the reduction in unmet demand.
    """
    scenario = await db.get(OptimizationScenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    if scenario.status != ScenarioStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Scenario is not completed")

    locations_raw: list[dict] = (scenario.result_stats or {}).get("_locations", [])
    if not locations_raw:
        raise HTTPException(status_code=400, detail="Scenario has no facility locations stored")

    # ── Load census areas with the same scope as the original run ──
    query = select(CensusArea)
    sf = scenario.scope_filters or {}
    if sf.get("parish_ids"):
        query = query.where(CensusArea.parish_id.in_(sf["parish_ids"]))
    else:
        if sf.get("province_codes"):
            query = query.where(CensusArea.province_code.in_(sf["province_codes"]))
        if sf.get("canton_codes"):
            query = query.where(CensusArea.canton_code.in_(sf["canton_codes"]))
        if sf.get("parish_codes"):
            query = query.where(CensusArea.parish_code.in_(sf["parish_codes"]))

    result = await db.execute(query)
    areas = result.scalars().all()
    if not areas:
        raise HTTPException(status_code=400, detail="No census areas found for this scenario's scope")

    area_id_to_idx = {a.id: i for i, a in enumerate(areas)}

    # Recover target_population_id stored when the scenario was originally run.
    _meta = (scenario.result_stats or {}).get("_meta", {})
    demand = await _load_demand(
        db, areas,
        _meta.get("target_population_id"),
        _meta.get("facility_type"),
    )

    # ── Load (possibly cached) distance matrix ──
    distance_matrix = await _load_distance_matrix(db, [a.id for a in areas])

    # ── Resolve facility indices and their covered demand ──
    facility_census_ids: list[int] = [loc["census_area_id"] for loc in locations_raw]
    facility_covered: list[float]  = [float(loc.get("covered_demand") or 0.0) for loc in locations_raw]

    facility_indices = [
        area_id_to_idx[cid]
        for cid in facility_census_ids
        if cid in area_id_to_idx
    ]
    if not facility_indices:
        raise HTTPException(status_code=400, detail="Facility census areas not found in scope")

    # ── Build capacity array ──
    if payload.capacity_per_facility is not None:
        cap_value = payload.capacity_per_facility
    else:
        total_covered = sum(facility_covered)
        cap_value = total_covered / len(facility_indices) if facility_indices else 1.0

    facility_capacity = np.full(len(facility_indices), cap_value, dtype=np.float64)

    # ── Run rebalancing (fast — no background task needed) ──
    rebal = await asyncio.to_thread(
        rebalancing_solve,
        distance_matrix,
        demand,
        facility_indices,
        facility_capacity,
        min_capacity=payload.min_capacity,
        max_transfers=payload.max_transfers,
    )

    # ── Build transfer response with geographic info ──
    fac_idx_to_area: dict[int, object] = {fi: areas[fi] for fi in facility_indices}

    transfers_info: list[RebalancingTransferInfo] = []
    for t in rebal.transfers:
        fa = fac_idx_to_area.get(t.from_facility)
        ta = fac_idx_to_area.get(t.to_facility)
        transfers_info.append(RebalancingTransferInfo(
            from_census_area_id=fa.id   if fa else t.from_facility,
            from_area_code=fa.area_code if fa else None,
            from_x=fa.x                 if fa else None,
            from_y=fa.y                 if fa else None,
            to_census_area_id=ta.id     if ta else t.to_facility,
            to_area_code=ta.area_code   if ta else None,
            to_x=ta.x                   if ta else None,
            to_y=ta.y                   if ta else None,
            amount=t.amount,
            impact=t.impact,
        ))

    # ── Build per-facility capacity info ──
    fac_capacities: list[FacilityCapacityInfo] = []
    for k, (fi, cov) in enumerate(zip(facility_indices, facility_covered)):
        area = fac_idx_to_area[fi]
        fac_capacities.append(FacilityCapacityInfo(
            census_area_id=area.id,
            area_code=area.area_code,
            covered_demand=round(cov, 2),
            old_capacity=round(cap_value, 2),
            new_capacity=round(rebal.new_capacity[k], 2),
        ))

    return RebalancingResponse(
        scenario_id=scenario_id,
        transfers=transfers_info,
        facility_capacities=fac_capacities,
        unmet_demand_before=rebal.unmet_demand_before,
        unmet_demand_after=rebal.unmet_demand_after,
        improvement_pct=rebal.improvement_pct,
        stats=rebal.stats,
    )


@router.delete("/{scenario_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scenario(scenario_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a scenario and its results."""
    scenario = await db.get(OptimizationScenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    await db.delete(scenario)


# --------------------------------------------------------------------------- #
# Background optimization task helpers                                          #
# --------------------------------------------------------------------------- #

async def _run_bump_hunter(
    payload: "OptimizationRequest",
    db: "AsyncSession",
    areas: list,
    demand: "np.ndarray",
    distance_matrix: "SparseDistanceMatrix",
    scenario_id: int,
    _timing: dict,
    _t0: float,
) -> None:
    """Handle the bump_hunter model flow: solve (or reuse fixed ids), assign areas, persist results.

    In complete_existing mode, existing DB facilities of the selected type are pre-loaded as
    fixed bump locations and their DB capacity (facilities.capacity) is respected during
    census-area assignment.  capacity=0 means the facility accepts no demand.
    """
    area_id_to_idx = {a.id: i for i, a in enumerate(areas)}
    _t = time.perf_counter()

    # --- Load existing facilities if complete_existing mode (initial run only) ---
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

    # --- Solver step ---
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
    _timing["6_solver"] = time.perf_counter() - _t

    # Merge existing facility bumps (pre-pend so they appear first) with solver bumps.
    # Deduplicate: if the solver independently found an existing facility location, keep
    # it only once in the existing list (it carries the capacity constraint).
    all_bump_indices = existing_bump_indices + [b for b in bump_indices if b not in existing_set]

    # --- Capacity-aware assignment within service radius ---
    bh_radius = float(payload.service_radius)
    # Existing bumps respect their DB capacity; new solver bumps are unconstrained.
    bump_facility_caps: dict[int, float] = {
        bidx: existing_bump_cap.get(bidx, math.inf) if bidx in existing_set else math.inf
        for bidx in all_bump_indices
    }

    # _single_capacity_pass assigns demand nearest-first with optional radius and
    # capacity limits, making it equivalent to the original simple assignment when
    # all capacities are infinite.
    _t = time.perf_counter()
    assignments, _ = _single_capacity_pass(
        distance_matrix, demand, all_bump_indices, bump_facility_caps, bh_radius
    )
    _timing["7_assignment"] = time.perf_counter() - _t

    # Invert assignments: bump_idx → [(area_idx, amount_assigned), ...]
    bump_to_served: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for area_idx, fac_assignments in assignments.items():
        for fac_idx, amount in fac_assignments:
            bump_to_served[fac_idx].append((area_idx, amount))

    # --- Build location objects with served_areas and stats ---
    locations = []
    covered_ids: set[int] = set()
    _total_demand_bh = float(np.sum(demand))
    _total_covered_bh = 0.0
    _weighted_tt_bh = 0.0
    _tt_demand_bh = 0.0
    _global_max_tt_bh = 0.0

    for bidx in all_bump_indices:
        area = areas[bidx]
        served_pairs = bump_to_served.get(bidx, [])  # [(area_idx, amount), ...]
        served_area_infos = []
        covered_dem = 0.0
        valid_tts = []

        for area_idx, amount in served_pairs:
            sa = areas[area_idx]
            dist = distance_matrix.distance_time(area_idx, bidx)
            covered_dem += amount
            _total_covered_bh += amount
            covered_ids.add(sa.id)
            if dist < MAX_DIST:
                valid_tts.append(dist)
                _weighted_tt_bh += amount * dist
                _tt_demand_bh += amount
                if dist > _global_max_tt_bh:
                    _global_max_tt_bh = dist
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

        max_tt = float(max(valid_tts)) if valid_tts else None
        _is_existing = bidx in existing_set and is_complete_existing

        locations.append(FacilityLocation(
            census_area_id=area.id,
            area_code=area.area_code,
            name=area.name,
            x=area.x,
            y=area.y,
            covered_demand=round(covered_dem),
            assigned_areas=len(served_pairs),
            max_travel_time=max_tt,
            is_existing=_is_existing,
            db_capacity=existing_bump_cap.get(bidx) if _is_existing else None,
            served_areas=served_area_infos,
        ))

    unassigned_areas_bh = [
        {"census_area_id": a.id, "area_code": a.area_code,
         "name": a.name, "x": a.x, "y": a.y}
        for idx, a in enumerate(areas)
        if a.id not in covered_ids and float(demand[idx]) > 1e-9
    ]

    bh_base_stats = bh.stats if bh is not None else {
        "num_bumps": len(all_bump_indices),
        "total_areas": len(areas),
        "total_demand": round(float(np.sum(demand))),
    }
    stats = {
        **bh_base_stats,
        "coverage_pct": round(_total_covered_bh / _total_demand_bh * 100, 2) if _total_demand_bh > 0 else 0.0,
        "avg_travel_time_minutes": round(_weighted_tt_bh / _tt_demand_bh, 2) if _tt_demand_bh > 0 else 0.0,
        "max_travel_time_minutes": round(_global_max_tt_bh, 2),
        "_meta": {
            "facility_type": payload.facility_type or "high_school",
            "mode": payload.mode.value,
            "target_population_id": payload.target_population_id,
        },
    }

    scenario = await db.get(OptimizationScenario, scenario_id)
    scenario.status = ScenarioStatus.COMPLETED
    scenario.result_stats = {
        **stats,
        "_locations": [loc.model_dump() for loc in locations],
        "_unassigned_areas": unassigned_areas_bh,
    }
    scenario.completed_at = datetime.now(timezone.utc)
    await db.commit()

    _timing["TOTAL"] = time.perf_counter() - _t0
    logger.warning(
        "BG task completed  scenario=%d  model=bump_hunter  bumps=%d  %.2fs",
        scenario_id, len(locations), _timing["TOTAL"],
    )


async def _resolve_pre_selected(
    payload: "OptimizationRequest",
    db: "AsyncSession",
    areas: list,
    area_id_to_idx: dict,
) -> tuple:
    """Resolve the pre-selected facility indices from the request payload.

    Handles two cases:
      - Reoptimization: split fixed_census_area_ids into kept and new indices.
      - COMPLETE_EXISTING mode: load existing DB facilities as pre-selected.

    Returns (pre_selected, pre_selected_set, existing_capacity, is_reoptimization,
             kept_indices, new_indices).
    """
    pre_selected: list[int] = []
    pre_selected_set: set[int] = set()
    existing_capacity: dict[int, float] = {}
    is_reoptimization = payload.fixed_census_area_ids is not None

    # Variables used only in the reoptimization path.
    kept_indices: list[int] = []
    new_indices:  list[int] = []

    if is_reoptimization:
        # Reoptimization: skip the solver and run a constrained nearest-assignment
        # over the exact set of facilities the user has confirmed.
        #
        # Kept facilities  = fixed_census_area_ids - reopt_new_facility_ids
        #   → must retain at least min_capacity demand (cap_min floor)
        # User-created     = reopt_new_facility_ids
        #   → empty at start; filled up to cap_max; for max_coverage,
        #     only areas within service_radius are eligible
        new_id_set = set(payload.reopt_new_facility_ids or [])

        for cid in payload.fixed_census_area_ids:
            if cid not in area_id_to_idx:
                continue
            idx = area_id_to_idx[cid]
            if cid in new_id_set:
                new_indices.append(idx)
            else:
                kept_indices.append(idx)

        pre_selected = kept_indices + new_indices
        pre_selected_set = set(pre_selected)

    elif payload.mode == OptimizationMode.COMPLETE_EXISTING and payload.facility_type:
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
            existing_facs = fac_result.scalars().all()
            seen_idx: set[int] = set()
            for f in existing_facs:
                if f.census_area_id in area_id_to_idx:
                    idx = area_id_to_idx[f.census_area_id]
                    cap = float(f.capacity) if f.capacity is not None else math.inf
                    if idx not in seen_idx:
                        pre_selected.append(idx)
                        seen_idx.add(idx)
                        # capacity=0 means no assignment; None (shouldn't happen, NOT NULL) → no limit
                        existing_capacity[idx] = cap
                    else:
                        # Multiple DB facilities at the same census area: sum capacities.
                        prev = existing_capacity[idx]
                        existing_capacity[idx] = (
                            math.inf if prev == math.inf or cap == math.inf
                            else prev + cap
                        )
            pre_selected_set = set(pre_selected)

    return (
        pre_selected,
        pre_selected_set,
        existing_capacity,  # idx → capacity (0 = no assignment, >0 = limit)
        is_reoptimization,
        kept_indices,
        new_indices,
    )


def _compute_unassigned_areas(
    areas: list,
    final_set: list,
    distance_matrix: "SparseDistanceMatrix",
    demand: "np.ndarray",
    assignment_radius,
    scope_area_ids: set,
) -> list:
    """Build the list of unassigned area dicts for areas with demand not covered by any facility.

    For each unassigned area computes travel time, Euclidean distance, and implied
    speed to the nearest open facility.
    """
    unassigned_areas_out = []
    for idx, a in enumerate(areas):
        if a.id in scope_area_ids or float(demand[idx]) <= 1e-9:
            continue

        # Find nearest open facility by travel time.
        nearest_fac_idx  = None
        nearest_tt       = float(MAX_DIST)
        for fac_idx in final_set:
            tt = distance_matrix.distance_time(idx, fac_idx)
            if tt < nearest_tt:
                nearest_tt      = tt
                nearest_fac_idx = fac_idx

        nearest_code    = None
        nearest_dist_km = None
        nearest_speed   = None
        tt_out          = None

        if nearest_fac_idx is not None and nearest_tt < MAX_DIST:
            nf_area  = areas[nearest_fac_idx]
            nearest_code = nf_area.area_code
            tt_out       = round(nearest_tt, 1)

            if (a.x is not None and a.y is not None
                    and nf_area.x is not None and nf_area.y is not None):
                lat_mid = math.radians((a.y + nf_area.y) / 2.0)
                dx_km   = (nf_area.x - a.x) * 111.0 * math.cos(lat_mid)
                dy_km   = (nf_area.y - a.y) * 111.0
                d_km    = math.sqrt(dx_km * dx_km + dy_km * dy_km)
                nearest_dist_km = round(d_km, 3)
                if nearest_tt > 0:
                    nearest_speed = round(d_km / (nearest_tt / 60.0), 1)

        # Within service radius but no capacity available → capacity-constrained exclusion.
        capacity_unassigned = (
            assignment_radius is not None
            and nearest_tt <= assignment_radius
        )

        unassigned_areas_out.append({
            "census_area_id": a.id,
            "area_code":      a.area_code,
            "name":           a.name,
            "x":              a.x,
            "y":              a.y,
            "demand":         round(float(demand[idx])),
            "nearest_facility_code":             nearest_code,
            "nearest_facility_travel_time_min":  tt_out,
            "nearest_facility_distance_km":      nearest_dist_km,
            "travel_speed_kmh":                  nearest_speed,
            "avg_speed_kmh":                     a.avg_speed_kmh,
            "capacity_unassigned":               capacity_unassigned,
        })
    return unassigned_areas_out


# --------------------------------------------------------------------------- #
# Background optimization task                                                  #
# --------------------------------------------------------------------------- #

async def _run_optimization_bg(
    scenario_id: int,
    payload: OptimizationRequest,
    db_name: str,
) -> None:
    """Run the full optimization pipeline in the background."""
    _t0 = time.perf_counter()
    _timing: dict[str, float] = {}
    SessionLocal = get_session_factory(db_name)

    try:
        async with SessionLocal() as db:
            # 1. Load census areas.
            query = select(CensusArea)
            if payload.scope_filters:
                f = payload.scope_filters
                if f.parish_ids:
                    query = query.where(CensusArea.parish_id.in_(f.parish_ids))
                else:
                    if f.province_codes:
                        query = query.where(CensusArea.province_code.in_(f.province_codes))
                    if f.canton_codes:
                        query = query.where(CensusArea.canton_code.in_(f.canton_codes))
                    if f.parish_codes:
                        query = query.where(CensusArea.parish_code.in_(f.parish_codes))
            
            result = await db.execute(query)
            areas = result.scalars().all()
            _timing["1_load_areas"] = time.perf_counter() - _t0

            if payload.model_type.value not in ("bump_hunter", "max_coverage") and len(areas) < payload.p_facilities:
                raise ValueError(
                    f"Not enough census areas ({len(areas)}) for p={payload.p_facilities}"
                )

            # 2. Build demand vector from the selected target population.
            demand = await _load_demand(
                db, areas, payload.target_population_id, payload.facility_type
            )

            # 2b. Build coordinate and speed arrays for travel-time estimation.
            xy = np.array(
                [(float(a.x or 0.0), float(a.y or 0.0)) for a in areas],
                dtype=np.float64,
            )
            speeds = np.array(
                [
                    float(a.avg_speed_kmh) if a.avg_speed_kmh else _FALLBACK_SPEED_KMH
                    for a in areas
                ],
                dtype=np.float64,
            )
            
            # 3. Load distance matrix and attach coordinates for full-pair estimation.
            _t = time.perf_counter()
            distance_matrix = await _load_distance_matrix(db, [a.id for a in areas])
            distance_matrix.xy = xy
            distance_matrix.speeds = speeds
            _timing["3_load_distance_matrix"] = time.perf_counter() - _t
            
            # ── BUMP HUNTER (separate flow — no solver dispatch or assignment) ──
            if payload.model_type.value == "bump_hunter":
                await _run_bump_hunter(
                    payload, db, areas, demand, distance_matrix,
                    scenario_id, _timing, _t0,
                )
                return
            
            # 4. Resolve pre-selected indices.
            area_id_to_idx = {a.id: i for i, a in enumerate(areas)}
            (
                pre_selected,
                pre_selected_set,
                existing_capacity,
                is_reoptimization,
                kept_indices,
                new_indices,
            ) = await _resolve_pre_selected(payload, db, areas, area_id_to_idx)

            # 5. Load scenario from DB.
            scenario = await db.get(OptimizationScenario, scenario_id)
            
            # 6. Run solver OR reoptimization constrained-assignment.
            _t = time.perf_counter()

            if is_reoptimization:
                # Skip the solver entirely.  Assign census areas directly to the
                # user-confirmed set of facilities using nearest-first order.
                cap_min_f = float(payload.min_capacity or 0.0)
                cap_max_f = float(payload.max_capacity) if payload.max_capacity else math.inf
                _assignment_radius = (
                    float(payload.service_radius) if payload.service_radius else None
                )
                # Build per-facility capacity overrides for user-added facilities.
                per_fac_cap: dict[int, tuple[float, float]] | None = None
                if payload.per_facility_capacity_overrides and new_indices:
                    new_idx_set = set(new_indices)
                    per_fac_cap = {}
                    for cid_str, caps in payload.per_facility_capacity_overrides.items():
                        cid = int(cid_str)
                        if cid in area_id_to_idx:
                            idx = area_id_to_idx[cid]
                            if idx in new_idx_set:
                                mn = float(caps.get("min_capacity") or cap_min_f)
                                mx_raw = caps.get("max_capacity")
                                mx = float(mx_raw) if mx_raw is not None else cap_max_f
                                per_fac_cap[idx] = (mn, mx)
                    if not per_fac_cap:
                        per_fac_cap = None

                final_facility_indices, assignments = await asyncio.to_thread(
                    _reopt_assignment,
                    distance_matrix, demand,
                    kept_indices, new_indices,
                    cap_min_f, cap_max_f,
                    payload.model_type == ModelType.MAX_COVERAGE,
                    _assignment_radius,
                    per_fac_cap,
                )
                _timing["6_reopt_assignment"] = time.perf_counter() - _t

                # Build minimal stats (travel-time fields are filled in step 10.5).
                total_dem = float(np.sum(demand))
                assigned_dem = sum(
                    sum(amt for _, amt in asgns)
                    for asgns in assignments.values()
                )
                stats: dict = {
                    "total_demand": round(total_dem),
                    "covered_demand": round(assigned_dem),
                    "uncovered_demand": round(total_dem - assigned_dem),
                    "coverage_pct": (
                        round(assigned_dem / total_dem * 100, 2) if total_dem > 0 else 0.0
                    ),
                    "num_facilities": len(final_facility_indices),
                    "service_radius_minutes": payload.service_radius,
                    "cap_min": payload.min_capacity,
                    "cap_max": payload.max_capacity,
                }

            else:
                # Standard path: run the solver then capacity-constrain the assignment.
                # In COMPLETE_EXISTING mode p means "additional facilities to place
                # on top of existing ones", so the solver receives the combined total.
                effective_p = payload.p_facilities
                if (
                    payload.mode == OptimizationMode.COMPLETE_EXISTING
                    and pre_selected
                    and payload.p_facilities is not None  # max_coverage has no fixed p
                ):
                    effective_p = len(pre_selected) + payload.p_facilities

                facility_indices, stats = await asyncio.to_thread(
                    _run_solver, payload, demand, distance_matrix, pre_selected, effective_p
                )
                _timing["6_solver"] = time.perf_counter() - _t

                # 7. Build per-facility capacity map.
                # Existing facilities: use their DB capacity (0 → no assignment, >0 → limit).
                # New/optimized facilities: use max_capacity from request (None → no limit).
                max_cap = float(payload.max_capacity) if payload.max_capacity else math.inf
                facility_capacities: dict[int, float] = {
                    fac: existing_capacity[fac]
                    if fac in pre_selected_set and fac in existing_capacity
                    else max_cap
                    for fac in facility_indices
                }

                # 8. Capacity-constrained assignment (CPU-bound).
                # For max_coverage, only assign areas within the service radius so
                # that uncovered areas (beyond radius of every facility) remain
                # unassigned and are not counted as covered.
                _assignment_radius = (
                    float(payload.service_radius)
                    if payload.model_type == ModelType.MAX_COVERAGE
                    else None
                )
                _t = time.perf_counter()
                final_facility_indices, assignments = await asyncio.to_thread(
                    _capacity_assignment,
                    distance_matrix,
                    demand,
                    facility_indices,
                    facility_capacities,
                    payload.min_capacity,
                    pre_selected_set,
                    _assignment_radius,
                )
                _timing["8_capacity_assignment"] = time.perf_counter() - _t
                # Update num_facilities to reflect the count after min_capacity filtering.
                stats["num_facilities"] = len(final_facility_indices)

            # 9. Deduplicate and invert assignments.
            # Guard against duplicate indices that would produce multiple DB rows
            # for the same (scenario_id, census_area_id) pair.
            seen: set[int] = set()
            final_facility_indices = [
                idx for idx in final_facility_indices
                if not (idx in seen or seen.add(idx))
            ]
            stats["num_facilities"] = len(final_facility_indices)
            final_set = set(final_facility_indices)
            facility_to_served: dict[int, list[tuple[int, float]]] = defaultdict(list)
            for area_idx, fac_assignments in assignments.items():
                for fac_idx, amount in fac_assignments:
                    if fac_idx in final_set:
                        facility_to_served[fac_idx].append((area_idx, amount))

            # 10. Persist results.
            # served_with_tt: [(area_i, demand_amount, travel_time_minutes), ...]
            orm_results: list[tuple[int, object, list[tuple[int, float, float]], float | None]] = []
            for idx in final_facility_indices:
                area = areas[idx]
                served = facility_to_served.get(idx, [])
                covered_dem = float(sum(amt for _, amt in served))

                served_with_tt = [
                    (area_i, amt, distance_matrix.distance_time(area_i, idx))
                    for area_i, amt in served
                ]
                valid_tt = [tt for _, _, tt in served_with_tt if tt < MAX_DIST]
                max_tt = float(max(valid_tt)) if valid_tt else None

                served_ids = [[areas[area_i].id, round(amt)] for area_i, amt, _ in served_with_tt]

                orm_result = OptimizationResult(
                    scenario_id=scenario_id,
                    census_area_id=area.id,
                    covered_demand=covered_dem,
                    assigned_areas=len(served_with_tt),
                    max_travel_time=max_tt,
                    served_area_ids=served_ids,
                )
                db.add(orm_result)
                orm_results.append((idx, area, served_with_tt, max_tt))

            # 10.5. Compute travel-time stats from cached distances in orm_results.
            _total_assigned = 0.0
            _weighted_tt = 0.0
            _tt_demand = 0.0
            _actual_max_tt = 0.0
            for _idx, _area_obj, _served_with_tt, _max_tt in orm_results:
                for _area_i, _amt, _tt in _served_with_tt:
                    _total_assigned += _amt
                    if _tt < MAX_DIST:
                        _weighted_tt += _amt * _tt
                        _tt_demand += _amt
                if _max_tt is not None and _max_tt > _actual_max_tt:
                    _actual_max_tt = _max_tt

            _total_dem = float(np.sum(demand))
            stats["avg_travel_time_minutes"] = (
                round(_weighted_tt / _tt_demand, 2) if _tt_demand > 0 else 0.0
            )
            stats["max_travel_time_minutes"] = round(_actual_max_tt, 2)

            # Always use the capacity-constrained assigned demand for covered_demand
            # so that Summary stats match the Served Areas sheet totals.
            stats["covered_demand"]   = round(min(_total_assigned, _total_dem))
            stats["uncovered_demand"] = round(max(_total_dem - _total_assigned, 0.0))
            stats["coverage_pct"] = (
                round(min(_total_assigned, _total_dem) / _total_dem * 100, 2)
                if _total_dem > 0 else 0.0
            )

            # 11. Build full location objects (with served_areas for map).
            locations = []
            for idx, area, served_with_tt, max_tt in orm_results:
                served_area_infos = []
                for area_i, amt, tt in served_with_tt:
                    sa = areas[area_i]
                    if sa.x is not None and sa.y is not None:
                        served_area_infos.append(
                            ServedAreaInfo(
                                census_area_id=sa.id,
                                area_code=sa.area_code,
                                x=sa.x,
                                y=sa.y,
                                assigned_demand=round(amt),
                                travel_time=round(tt, 1) if tt < MAX_DIST else None,
                                avg_speed_kmh=sa.avg_speed_kmh,
                            )
                        )

                # is_existing: True only for DB-sourced existing facilities
                #   (complete_existing mode).  Reoptimization kept-facilities are
                #   model-placed and should remain blue on the map.
                # is_user_added: True only for facilities added by the user during
                #   reoptimization — shown in green.
                _new_indices_set = set(new_indices)  # empty outside reoptimization
                _is_user_added  = idx in _new_indices_set
                _is_existing    = (idx in pre_selected_set
                                   and not is_reoptimization
                                   and not _is_user_added)
                # Store the raw DB capacity for existing facilities so the frontend
                # can display it in the facility popup.
                _db_capacity = existing_capacity.get(idx) if _is_existing else None

                locations.append(
                    FacilityLocation(
                        census_area_id=area.id,
                        area_code=area.area_code,
                        name=area.name,
                        x=area.x,
                        y=area.y,
                        covered_demand=round(sum(amt for _, amt, _ in served_with_tt)),
                        assigned_areas=len(served_with_tt),
                        max_travel_time=max_tt,
                        avg_speed_kmh=area.avg_speed_kmh,
                        is_existing=_is_existing,
                        is_user_added=_is_user_added,
                        db_capacity=_db_capacity,
                        served_areas=served_area_infos,
                    )
                )

            # Add meta for frontend facility-type / mode lookup.
            stats["_meta"] = {
                "facility_type": payload.facility_type or "high_school",
                "mode": payload.mode.value,
                "target_population_id": payload.target_population_id,
                "scope_filters": payload.scope_filters.model_dump() if payload.scope_filters else None,
            }

            # 11.5 Compute unassigned areas (scope areas with demand > 0 not served
            # by any facility).  For each unassigned area also compute the distance,
            # travel time and implied speed to the nearest open facility.
            served_census_ids: set[int] = set()
            for loc in locations:
                served_census_ids.add(loc.census_area_id)
                for sa in loc.served_areas:
                    served_census_ids.add(sa.census_area_id)

            unassigned_areas_out = _compute_unassigned_areas(
                areas, final_facility_indices, distance_matrix, demand,
                _assignment_radius, served_census_ids,
            )

            # 12. Mark scenario completed; store full locations in result_stats.
            scenario.status = ScenarioStatus.COMPLETED
            scenario.result_stats = {
                **stats,
                "_locations": [loc.model_dump() for loc in locations],
                "_unassigned_areas": unassigned_areas_out,
            }
            scenario.completed_at = datetime.now(timezone.utc)
            await db.commit()

            _timing["TOTAL"] = time.perf_counter() - _t0
            logger.warning(
                "BG task completed  scenario=%d  model=%s  p=%d  %.2fs",
                scenario_id,
                payload.model_type.value,
                payload.p_facilities,
                _timing["TOTAL"],
            )

    except Exception as exc:
        logger.error("BG task FAILED scenario=%d: %s", scenario_id, exc, exc_info=True)
        try:
            async with SessionLocal() as db2:
                sc = await db2.get(OptimizationScenario, scenario_id)
                if sc:
                    sc.status = ScenarioStatus.FAILED
                    sc.error_message = str(exc)[:2000]
                    await db2.commit()
        except Exception:
            logger.error("Failed to update scenario %d to FAILED", scenario_id)
    finally:
        pass


# --------------------------------------------------------------------------- #
# Capacity-constrained assignment                                               #
# --------------------------------------------------------------------------- #

_FALLBACK_SPEED_KMH = 30.0  # km/h — used when avg_speed_kmh is NULL


def _reopt_assignment(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    kept_indices: list[int],
    new_indices: list[int],
    cap_min: float,
    cap_max: float,
    enforce_radius: bool,
    radius: float | None,
    per_facility_cap: dict[int, tuple[float, float]] | None = None,
) -> tuple[list[int], dict[int, list[tuple[int, float]]]]:
    """
    Constrained nearest-assignment for reoptimization.

    The solver is bypassed entirely.  Census areas are assigned to the set of
    facilities the user has confirmed (kept from the previous run + newly added).

    Kept facilities (from the previous optimization run):
      - No service-radius restriction (their placement was already radius-aware).
      - After the main pass, each kept facility whose load is below cap_min has
        demand pulled from the nearest user-created facilities until the floor
        is satisfied.  This prevents kept facilities from becoming empty when
        a nearby user-created facility attracts all their demand.

    User-created (newly added) facilities:
      - For max_coverage (enforce_radius=True): only census areas within
        `radius` minutes are eligible; areas outside cannot be routed there.
      - For all other models: no radius restriction.
      - Hard ceiling: cap_max.

    Both facility types share the same cap_max ceiling.
    Demand is split across multiple facilities only when a facility is full.
    """
    all_indices = kept_indices + new_indices
    kept_set    = set(kept_indices)
    new_set     = set(new_indices)
    fac_set     = set(all_indices)

    # Use per-facility max capacity override for user-added facilities when provided;
    # fall back to the global cap_max for facilities without an override.
    remaining: dict[int, float] = {
        f: (per_facility_cap[f][1] if per_facility_cap and f in per_facility_cap else cap_max)
        for f in all_indices
    }
    fac_load: dict[int, float]  = defaultdict(float)
    assignments: dict[int, list[tuple[int, float]]] = {}

    # Main pass: nearest-first, radius only for user-created in max_coverage.
    min_dists  = dm.min_dist_to_set(all_indices)
    area_order = np.argsort(min_dists, kind="stable").tolist()

    for i in area_order:
        d = float(demand[i])
        if d <= 1e-9:
            continue
        rem = d
        area_asgn: list[tuple[int, float]] = []
        for dist, fac in _sorted_facilities_for_area(dm, i, fac_set):
            if rem <= 1e-9:
                break
            # For max_coverage: no facility (kept or user-created) may serve an area
            # outside the service radius.
            if enforce_radius and radius is not None and dist > radius:
                continue
            cap_left = remaining.get(fac, 0.0)
            if cap_left <= 1e-9:
                continue
            take = min(rem, cap_left)
            area_asgn.append((fac, take))
            remaining[fac] -= take
            fac_load[fac]  += take
            rem -= take
        if area_asgn:
            assignments[i] = area_asgn

    # Floor enforcement: kept facilities must retain >= cap_min demand.
    # Pull demand from the nearest user-created facilities that serve areas
    # within the kept facility's neighbourhood, until the floor is satisfied.
    if cap_min > 0 and new_indices:
        for kf in kept_indices:
            deficit = cap_min - fac_load.get(kf, 0.0)
            if deficit <= 1e-9:
                continue
            rows, vals = dm.col_neighbors(kf)
            order = np.argsort(vals, kind="stable")
            for k in order:
                if deficit <= 1e-9:
                    break
                area_i = int(rows[k])
                if area_i not in assignments:
                    continue
                for donor, amt in [(f, a) for f, a in assignments[area_i] if f in new_set]:
                    steal = min(amt, deficit)
                    rebuilt: list[tuple[int, float]] = []
                    for f2, a2 in assignments[area_i]:
                        if f2 == donor:
                            leftover = a2 - steal
                            if leftover > 1e-9:
                                rebuilt.append((f2, leftover))
                        else:
                            rebuilt.append((f2, a2))
                    rebuilt.append((kf, steal))
                    assignments[area_i] = rebuilt
                    fac_load[donor]           -= steal
                    fac_load[kf]              = fac_load.get(kf, 0.0) + steal
                    remaining[donor]          += steal
                    deficit                   -= steal
                    if deficit <= 1e-9:
                        break

    return all_indices, assignments


def _capacity_assignment(
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
        assignments, fac_demand = _single_capacity_pass(dm, demand, active, facility_capacities, radius)

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


def _single_capacity_pass(
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
        for dist, fac in _sorted_facilities_for_area(dm, i, fac_set):
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


def _sorted_facilities_for_area(
    dm: SparseDistanceMatrix,
    area_idx: int,
    fac_set: set[int],
) -> list[tuple[float, int]]:
    """
    Return (distance, facility_index) pairs sorted by travel time to area_idx (ascending).

    Uses dm.distance_time() which returns the stored value when available
    and falls back to the harmonic-mean speed estimate for missing pairs.
    """
    pairs = [(dm.distance_time(area_idx, fac), fac) for fac in fac_set]
    pairs.sort()
    return pairs


# --------------------------------------------------------------------------- #
# Solver dispatch                                                               #
# --------------------------------------------------------------------------- #

def _run_solver(
    payload: OptimizationRequest,
    demand: np.ndarray,
    distance_matrix: SparseDistanceMatrix,
    pre_selected: list[int],
    override_p: int | None = None,
) -> tuple[list[int], dict]:
    """Dispatch to the appropriate solver and return (facility_indices, stats).

    override_p, when provided, replaces payload.p_facilities as the total
    facility count passed to the solver. Used for COMPLETE_EXISTING mode where
    the effective p = len(existing) + p_new.
    """
    model = payload.model_type.value
    p = override_p if override_p is not None else payload.p_facilities

    if model == "p_median":
        max_exchange = 0 if distance_matrix.n > 5_000 else 50
        result = p_median_solve(
            distance_matrix, demand, p,
            max_exchange_iters=max_exchange,
            pre_selected=pre_selected or None,
        )
        result.coverage_stats["num_facilities"] = len(result.facility_indices)
        return result.facility_indices, result.coverage_stats

    if model == "p_center":
        result = p_center_solve(
            distance_matrix, demand, p,
            pre_selected=pre_selected or None,
        )
        result.coverage_stats["num_facilities"] = len(result.facility_indices)
        return result.facility_indices, result.coverage_stats

    if model == "max_coverage":
        result = max_coverage_solve(
            distance_matrix, demand, p, payload.service_radius,
            cap_min=float(payload.min_capacity) if payload.min_capacity else 0.0,
            cap_max=float(payload.max_capacity) if payload.max_capacity else None,
            pre_selected=pre_selected or None,
        )
        return result.facility_indices, result.coverage_stats

    raise ValueError(f"Unknown model type: {model}")


# --------------------------------------------------------------------------- #
# Distance-matrix in-process cache                                              #
# --------------------------------------------------------------------------- #

# Keeps the last _DM_CACHE_SIZE SparseDistanceMatrix objects in memory.
# Key: hash of the sorted area_ids tuple.  Invalidated only on restart.
_DM_CACHE: dict[int, SparseDistanceMatrix] = {}
_DM_CACHE_SIZE = 8


def _dm_cache_key(area_ids: list[int]) -> int:
    return hash(tuple(sorted(area_ids)))


# --------------------------------------------------------------------------- #
# Misc helpers                                                                  #
# --------------------------------------------------------------------------- #

async def _load_distance_matrix(
    db: AsyncSession, area_ids: list[int]
) -> SparseDistanceMatrix:
    """
    Load the travel-time neighbor lists for the given area ids and return a
    SparseDistanceMatrix (CSR + CSC adjacency lists).

    Results are cached in memory (up to _DM_CACHE_SIZE entries).  A cache hit
    makes repeated runs with the same geographic scope nearly instant.

    Falls back to a k-NN Euclidean approximation if the DB table is empty.
    """
    from sqlalchemy import text

    cache_key = _dm_cache_key(area_ids)
    if cache_key in _DM_CACHE:
        logger.warning("  distance_matrix: CACHE HIT  n=%d", len(area_ids))
        return _DM_CACHE[cache_key]

    n = len(area_ids)
    id_to_idx = {aid: idx for idx, aid in enumerate(area_ids)}
    id_list = ",".join(str(i) for i in area_ids)

    _t_execute = time.perf_counter()
    rows = await db.execute(
        text(
            f"SELECT from_area_id, to_area_id, travel_time_minutes "
            f"FROM distance_matrix "
            f"WHERE from_area_id IN ({id_list}) AND to_area_id IN ({id_list})"
        )
    )
    _execute_time = time.perf_counter() - _t_execute

    _t_fetch = time.perf_counter()
    data = rows.fetchall()
    _fetch_time = time.perf_counter() - _t_fetch

    if data:
        # Extract columns with list comprehensions — much faster than
        # np.array(data, dtype=object) which boxes every element as a Python object.
        _t_extract = time.perf_counter()
        from_ids = np.array([r[0] for r in data], dtype=np.int32)
        to_ids   = np.array([r[1] for r in data], dtype=np.int32)
        times    = np.array([r[2] for r in data], dtype=np.float32)
        _extract_time = time.perf_counter() - _t_extract

        _t_index = time.perf_counter()
        # Build a flat lookup array: DB area_id → matrix index.
        max_id = int(max(from_ids.max(), to_ids.max())) + 1
        idx_map = np.full(max_id, -1, dtype=np.int32)
        for aid, idx in id_to_idx.items():
            if aid < max_id:
                idx_map[aid] = idx

        row_arr = idx_map[from_ids]
        col_arr = idx_map[to_ids]
        val_arr = np.clip(np.round(times).astype(np.int32), 0, MAX_DIST).astype(np.uint16)

        valid = (row_arr >= 0) & (col_arr >= 0) & (row_arr != col_arr)
        row_arr = row_arr[valid]
        col_arr = col_arr[valid]
        val_arr = val_arr[valid]
        _index_time = time.perf_counter() - _t_index

        _t_build = time.perf_counter()
        dm = SparseDistanceMatrix.from_coo(n, row_arr, col_arr, val_arr)
        _build_time = time.perf_counter() - _t_build

        logger.warning(
            "  distance_matrix: execute=%.3fs  fetch=%.3fs  extract=%.3fs"
            "  index=%.3fs  build=%.3fs  rows=%d  n=%d",
            _execute_time, _fetch_time, _extract_time,
            _index_time, _build_time, len(data), n,
        )

        # Store in cache; evict oldest entry when full.
        if len(_DM_CACHE) >= _DM_CACHE_SIZE:
            _DM_CACHE.pop(next(iter(_DM_CACHE)))
        _DM_CACHE[cache_key] = dm
        return dm

    # ── Fallback: k-NN Euclidean (dev/test when DB table is empty) ─────
    K = min(500, n - 1)
    result = await db.execute(
        select(CensusArea.id, CensusArea.x, CensusArea.y).where(
            CensusArea.id.in_(area_ids)
        )
    )
    coords = {row.id: (row.x or 0.0, row.y or 0.0) for row in result}
    xy = np.array([(coords[aid][0], coords[aid][1]) for aid in area_ids], dtype=np.float32)

    row_list, col_list, val_list = [], [], []
    for i in range(n):
        diff = xy - xy[i]
        dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        dists[i] = float(MAX_DIST)
        knn = np.argpartition(dists, K)[:K]
        vals = np.clip(dists[knn], 0, MAX_DIST).astype(np.uint16)
        row_list.append(np.full(K, i, dtype=np.int32))
        col_list.append(knn.astype(np.int32))
        val_list.append(vals)

    return SparseDistanceMatrix.from_coo(
        n,
        np.concatenate(row_list),
        np.concatenate(col_list),
        np.concatenate(val_list),
    )
