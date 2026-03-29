"""
REST endpoints for social and economic impact analysis.

Corresponds to: scr.planificador.fCalculoImpactos

POST /impacts/calculate  – Compute impact indicators for a given scenario.
GET  /impacts/{id}       – Retrieve previously computed impacts.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_db_name
from app.models.census import CensusArea
from app.models.optimization import OptimizationResult, OptimizationScenario
from app.models.target_population import CensusAreaPopulation, TargetPopulation

import numpy as np

router = APIRouter(prefix="/impacts", tags=["impacts"])


class ImpactRequest(BaseModel):
    scenario_id: int
    service_radius: float = Field(..., ge=1.0, le=600.0, description="Service radius in minutes")
    inflation_rate: float = Field(0.0, ge=0.0, le=1.0, description="Annual inflation rate (0–1)")
    cost_per_unit: float = Field(0.0, ge=0.0, description="Construction cost per service unit")


class AreaImpact(BaseModel):
    census_area_id: int
    area_code: str
    population: float
    is_covered: bool
    nearest_facility_id: int | None
    travel_time: float | None


class ImpactResponse(BaseModel):
    scenario_id: int
    total_population: float
    covered_population: float
    coverage_pct: float
    uncovered_areas: list[AreaImpact]
    covered_areas: list[AreaImpact]
    estimated_total_cost: float
    summary: dict


@router.post("/calculate", response_model=ImpactResponse)
async def calculate_impacts(
    payload: ImpactRequest, db: AsyncSession = Depends(get_db)
):
    """
    Compute social coverage impact for a completed optimization scenario.

    For each census area in the scenario scope, determine whether it falls
    within the service radius of any selected facility and compute the
    population covered vs. uncovered.
    """
    scenario = await db.get(OptimizationScenario, payload.scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    if scenario.status.value != "completed":
        raise HTTPException(status_code=422, detail="Scenario has not completed successfully")

    # Load selected facility area IDs.
    res = await db.execute(
        select(OptimizationResult).where(
            OptimizationResult.scenario_id == payload.scenario_id
        )
    )
    selected_area_ids = {r.census_area_id for r in res.scalars().all()}

    # Load all census areas in scope.
    query = select(CensusArea)
    if scenario.scope_filters:
        f = scenario.scope_filters
        if f.get("province_codes"):
            query = query.where(CensusArea.province_code.in_(f["province_codes"]))
        if f.get("canton_codes"):
            query = query.where(CensusArea.canton_code.in_(f["canton_codes"]))

    result = await db.execute(query)
    areas = result.scalars().all()

    # Load total population (all_ages) for each area.
    tp_result = await db.execute(
        select(TargetPopulation).where(TargetPopulation.code == "all_ages")
    )
    tp = tp_result.scalar_one_or_none()
    pop_map: dict[int, float] = {}
    if tp:
        pop_rows = await db.execute(
            select(CensusAreaPopulation).where(
                CensusAreaPopulation.census_area_id.in_([a.id for a in areas]),
                CensusAreaPopulation.target_population_id == tp.id,
            )
        )
        pop_map = {r.census_area_id: r.population for r in pop_rows.scalars().all()}

    # Build coordinate arrays for distance computation.
    xy = np.array([(a.x or 0.0, a.y or 0.0) for a in areas])
    facility_mask = np.array([a.id in selected_area_ids for a in areas])
    facility_xy = xy[facility_mask]

    covered_list: list[AreaImpact] = []
    uncovered_list: list[AreaImpact] = []

    total_population = 0.0
    covered_population = 0.0

    for i, area in enumerate(areas):
        area_pop = pop_map.get(area.id, 0.0)
        total_population += area_pop

        if facility_xy.shape[0] == 0:
            travel_time = None
            is_covered = False
            nearest_fac = None
        else:
            dists = np.sqrt(np.sum((facility_xy - xy[i]) ** 2, axis=1))
            nearest_idx = int(np.argmin(dists))
            travel_time = float(dists[nearest_idx])
            is_covered = travel_time <= payload.service_radius
            fac_areas = [a for a in areas if a.id in selected_area_ids]
            nearest_fac = fac_areas[nearest_idx].id if nearest_idx < len(fac_areas) else None

        impact = AreaImpact(
            census_area_id=area.id,
            area_code=area.area_code,
            population=area_pop,
            is_covered=is_covered,
            nearest_facility_id=nearest_fac,
            travel_time=round(travel_time, 2) if travel_time is not None else None,
        )

        if is_covered:
            covered_population += area_pop
            covered_list.append(impact)
        else:
            uncovered_list.append(impact)

    coverage_pct = (covered_population / total_population * 100) if total_population > 0 else 0.0
    estimated_cost = payload.cost_per_unit * len(selected_area_ids) * (1 + payload.inflation_rate)

    return ImpactResponse(
        scenario_id=payload.scenario_id,
        total_population=round(total_population, 2),
        covered_population=round(covered_population, 2),
        coverage_pct=round(coverage_pct, 2),
        covered_areas=covered_list,
        uncovered_areas=uncovered_list,
        estimated_total_cost=round(estimated_cost, 2),
        summary={
            "num_facilities": len(selected_area_ids),
            "num_covered_areas": len(covered_list),
            "num_uncovered_areas": len(uncovered_list),
            "coverage_pct": round(coverage_pct, 2),
            "estimated_cost": round(estimated_cost, 2),
        },
    )
