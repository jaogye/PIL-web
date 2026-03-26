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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session_factory
from app.dependencies import get_db, get_db_name
from app.models.census import CensusArea
from app.models.facility import Facility, FacilityStatus, FacilityType
from app.optimization.sparse_matrix import MAX_DIST, SparseDistanceMatrix
from app.models.optimization import (
    ModelType,
    OptimizationResult,
    OptimizationScenario,
    ScenarioStatus,
)
from app.optimization import (
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
)

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
        p_facilities=payload.p_facilities,
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
            if not k.startswith("_")
        }
        locations = [
            FacilityLocation.model_validate(loc)
            for loc in scenario.result_stats["_locations"]
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
    demand = np.array([a.demand for a in areas], dtype=np.float64)

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

            if len(areas) < payload.p_facilities:
                raise ValueError(
                    f"Not enough census areas ({len(areas)}) for p={payload.p_facilities}"
                )

            # 2. Build demand vector.
            demand = np.array([a.demand for a in areas], dtype=np.float64)

            # 3. Load distance matrix.
            _t = time.perf_counter()
            distance_matrix = await _load_distance_matrix(db, [a.id for a in areas])
            _timing["3_load_distance_matrix"] = time.perf_counter() - _t

            # 4. Resolve pre-selected indices.
            pre_selected: list[int] = []
            pre_selected_set: set[int] = set()
            existing_capacity: dict[int, float] = {}

            if payload.fixed_census_area_ids is not None:
                # Reoptimization: user specified exact fixed facilities by census_area_id.
                area_id_to_idx = {a.id: i for i, a in enumerate(areas)}
                for cid in payload.fixed_census_area_ids:
                    if cid in area_id_to_idx:
                        idx = area_id_to_idx[cid]
                        pre_selected.append(idx)
                pre_selected_set = set(pre_selected)
                # existing_capacity left empty → fixed facilities use max_capacity (or inf).

            elif payload.mode == OptimizationMode.COMPLETE_EXISTING and payload.facility_type:
                try:
                    fac_type = FacilityType(payload.facility_type)
                except ValueError:
                    fac_type = None
                if fac_type:
                    area_id_to_idx = {a.id: i for i, a in enumerate(areas)}
                    fac_result = await db.execute(
                        select(Facility).where(
                            Facility.status == FacilityStatus.EXISTING,
                            Facility.facility_type == fac_type,
                            Facility.census_area_id.in_(list(area_id_to_idx.keys())),
                        )
                    )
                    existing_facs = fac_result.scalars().all()
                    for f in existing_facs:
                        if f.census_area_id in area_id_to_idx:
                            idx = area_id_to_idx[f.census_area_id]
                            pre_selected.append(idx)
                            existing_capacity[idx] = float(f.capacity) if f.capacity else math.inf
                    pre_selected_set = set(pre_selected)

            # 5. Load scenario from DB.
            scenario = await db.get(OptimizationScenario, scenario_id)

            # 6. Run solver (CPU-bound – off-loads to thread pool).
            _t = time.perf_counter()
            facility_indices, stats = await asyncio.to_thread(
                _run_solver, payload, demand, distance_matrix, pre_selected
            )
            _timing["6_solver"] = time.perf_counter() - _t

            # 7. Build per-facility capacity map.
            max_cap = float(payload.max_capacity) if payload.max_capacity else math.inf
            facility_capacities: dict[int, float] = {
                fac: existing_capacity.get(fac, math.inf)
                if fac in pre_selected_set
                else max_cap
                for fac in facility_indices
            }

            # 8. Capacity-constrained assignment (CPU-bound).
            _t = time.perf_counter()
            final_facility_indices, assignments = await asyncio.to_thread(
                _capacity_assignment,
                distance_matrix,
                demand,
                facility_indices,
                facility_capacities,
                payload.min_capacity,
                pre_selected_set,
            )
            _timing["8_capacity_assignment"] = time.perf_counter() - _t

            # 9. Invert assignments.
            final_set = set(final_facility_indices)
            facility_to_served: dict[int, list[tuple[int, float]]] = defaultdict(list)
            for area_idx, fac_assignments in assignments.items():
                for fac_idx, amount in fac_assignments:
                    if fac_idx in final_set:
                        facility_to_served[fac_idx].append((area_idx, amount))

            # 10. Persist results.
            orm_results: list[tuple[int, object, list[tuple[int, float]]]] = []
            for idx in final_facility_indices:
                area = areas[idx]
                served = facility_to_served.get(idx, [])
                covered_dem = float(sum(amt for _, amt in served))

                max_tt = None
                if served:
                    served_idx_set = {i for i, _ in served}
                    rows_c, vals_c = distance_matrix.col_neighbors(idx)
                    dists = [
                        int(v)
                        for r, v in zip(rows_c.tolist(), vals_c.tolist())
                        if r in served_idx_set
                    ]
                    max_tt = float(max(dists)) if dists else None

                served_ids = [[areas[i].id, round(amt, 4)] for i, amt in served]

                orm_result = OptimizationResult(
                    scenario_id=scenario_id,
                    census_area_id=area.id,
                    covered_demand=covered_dem,
                    assigned_areas=len(served),
                    max_travel_time=max_tt,
                    served_area_ids=served_ids,
                )
                db.add(orm_result)
                orm_results.append((idx, area, served))

            # 10.5. Recompute travel-time stats from actual assignments.
            _total_assigned = 0.0
            _weighted_tt = 0.0
            _tt_demand = 0.0
            _actual_max_tt = 0.0
            for _idx, _area_obj, _served in orm_results:
                _rows, _vals = distance_matrix.col_neighbors(_idx)
                _dl: dict[int, float] = dict(
                    zip(_rows.tolist(), [float(v) for v in _vals.tolist()])
                )
                for _area_i, _amt in _served:
                    _total_assigned += _amt
                    _tt = _dl.get(_area_i)
                    if _tt is not None and _tt < MAX_DIST:
                        _weighted_tt += _amt * _tt
                        _tt_demand += _amt
                        if _tt > _actual_max_tt:
                            _actual_max_tt = _tt

            _total_dem = float(np.sum(demand))
            stats["avg_travel_time_minutes"] = (
                round(_weighted_tt / _tt_demand, 2) if _tt_demand > 0 else 0.0
            )
            stats["max_travel_time_minutes"] = round(_actual_max_tt, 2)
            stats["coverage_pct"] = (
                round(_total_assigned / _total_dem * 100, 2) if _total_dem > 0 else 0.0
            )

            # 11. Build full location objects (with served_areas for map).
            locations = []
            for idx, area, served in orm_results:
                rows_c, vals_c = distance_matrix.col_neighbors(idx)
                dist_lookup: dict[int, float] = {
                    int(r): float(v)
                    for r, v in zip(rows_c.tolist(), vals_c.tolist())
                }

                served_area_infos = []
                for area_i, amt in served:
                    sa = areas[area_i]
                    if sa.x is not None and sa.y is not None:
                        served_area_infos.append(
                            ServedAreaInfo(
                                census_area_id=sa.id,
                                area_code=sa.area_code,
                                x=sa.x,
                                y=sa.y,
                                assigned_demand=round(amt, 4),
                                travel_time=dist_lookup.get(area_i),
                            )
                        )

                locations.append(
                    FacilityLocation(
                        census_area_id=area.id,
                        area_code=area.area_code,
                        name=area.name,
                        x=area.x,
                        y=area.y,
                        covered_demand=float(sum(amt for _, amt in served)),
                        assigned_areas=len(served),
                        max_travel_time=dist_lookup.get(idx),
                        is_existing=idx in pre_selected_set,
                        served_areas=served_area_infos,
                    )
                )

            # Add meta for frontend facility-type / mode lookup.
            stats["_meta"] = {
                "facility_type": payload.facility_type or "high_school",
                "mode": payload.mode.value,
            }

            # 12. Mark scenario completed; store full locations in result_stats.
            scenario.status = ScenarioStatus.COMPLETED
            scenario.result_stats = {
                **stats,
                "_locations": [loc.model_dump() for loc in locations],
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

def _capacity_assignment(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    solver_facility_indices: list[int],
    facility_capacities: dict[int, float],
    min_capacity: float | None,
    pre_selected_set: set[int],
) -> tuple[list[int], dict[int, list[tuple[int, float]]]]:
    """
    Assign census areas to facilities with capacity constraints.

    Areas are processed nearest-first so that the closest areas have priority
    in filling a facility's capacity.  When an area's demand exceeds the
    remaining capacity of its nearest facility, the overflow goes to the next
    nearest facility with available capacity (partial assignment).

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
        assignments, fac_demand = _single_capacity_pass(dm, demand, active, facility_capacities)

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
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, float]]:
    """
    One greedy pass: assign each area to its nearest facility(ies) subject to
    capacity, processing areas in ascending order of distance to nearest facility.
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
        for fac in _sorted_facilities_for_area(dm, i, fac_set):
            if rem <= 1e-9:
                break
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
) -> list[int]:
    """
    Return facility indices sorted by distance to area_idx (ascending).
    Facilities not stored as CSR neighbors appear last at MAX_DIST.
    """
    s = int(dm.csr_ptr[area_idx])
    e = int(dm.csr_ptr[area_idx + 1])
    neighbors = dm.csr_col[s:e]
    distances = dm.csr_val[s:e]

    pairs: list[tuple[int, int]] = [
        (int(distances[k]), int(neighbors[k]))
        for k in range(len(neighbors))
        if int(neighbors[k]) in fac_set
    ]
    pairs.sort()

    seen = {fac for _, fac in pairs}
    for fac in fac_set:
        if fac not in seen:
            pairs.append((MAX_DIST, fac))

    return [fac for _, fac in pairs]


# --------------------------------------------------------------------------- #
# Solver dispatch                                                               #
# --------------------------------------------------------------------------- #

def _run_solver(
    payload: OptimizationRequest,
    demand: np.ndarray,
    distance_matrix: SparseDistanceMatrix,
    pre_selected: list[int],
) -> tuple[list[int], dict]:
    """Dispatch to the appropriate solver and return (facility_indices, stats)."""
    model = payload.model_type.value

    if model == "p_median":
        max_exchange = 0 if distance_matrix.n > 5_000 else 50
        result = p_median_solve(
            distance_matrix, demand, payload.p_facilities,
            max_exchange_iters=max_exchange,
            pre_selected=pre_selected or None,
        )
        result.coverage_stats["num_facilities"] = len(result.facility_indices)
        return result.facility_indices, result.coverage_stats

    if model == "p_center":
        result = p_center_solve(
            distance_matrix, demand, payload.p_facilities,
            pre_selected=pre_selected or None,
        )
        result.coverage_stats["num_facilities"] = len(result.facility_indices)
        return result.facility_indices, result.coverage_stats

    if model == "max_coverage":
        result = max_coverage_solve(
            distance_matrix, demand, payload.p_facilities, payload.service_radius,
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
