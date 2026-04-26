"""
REST endpoints for facility location optimization.

POST /optimization/run      – Submit and immediately run an optimization scenario.
GET  /optimization/         – List all scenarios.
GET  /optimization/{id}     – Get a specific scenario with results.
DELETE /optimization/{id}   – Delete a scenario.
"""

import asyncio
import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_db_name
from app.models.census import CensusArea
from app.models.optimization import (
    ModelType,
    OptimizationResult,
    OptimizationScenario,
    ScenarioStatus,
)
from app.optimization import rebalancing_solve
from app.schemas.optimization import (
    FacilityCapacityInfo,
    FacilityLocation,
    OptimizationRequest,
    OptimizationResponse,
    RebalancingRequest,
    RebalancingResponse,
    RebalancingTransferInfo,
    ScenarioSummary,
    UnassignedAreaInfo,
)
from app.services.optimization_service import (
    _load_demand,
    _load_distance_matrix,
    _run_optimization_bg,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimization", tags=["optimization"])


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
