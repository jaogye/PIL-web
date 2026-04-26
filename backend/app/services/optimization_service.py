"""
Background task and helper functions for facility-location optimization.

This module contains all business logic that runs outside the HTTP request/response
cycle.  The route handlers in routes/optimization.py delegate the heavy lifting here.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session_factory
from app.models.census import CensusArea
from app.models.facility import Facility, FacilityStatus, FacilityType, FacilityTypeLookup
from app.models.optimization import (
    ModelType,
    OptimizationResult,
    OptimizationScenario,
    ScenarioStatus,
)
from app.models.target_population import CensusAreaPopulation, TargetPopulation
from app.optimization import (
    max_coverage_solve,
    p_center_solve,
    p_median_solve,
    capacity_assignment,
)
from app.optimization.assignment import sorted_facilities_for_area
from app.optimization.sparse_matrix import MAX_DIST, SparseDistanceMatrix
from app.schemas.optimization import (
    FacilityLocation,
    OptimizationMode,
    OptimizationRequest,
    ServedAreaInfo,
    UnassignedAreaInfo,
)
from app.services.bump_hunter_service import run_bump_hunter

logger = logging.getLogger(__name__)

_FALLBACK_SPEED_KMH = 30.0

# In-process distance-matrix cache: keeps the last _DM_CACHE_SIZE matrices.
_DM_CACHE: dict[int, SparseDistanceMatrix] = {}
_DM_CACHE_SIZE = 8


# ─────────────────────────────────────────────────────────────────────────── #
# Demand loader                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

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
    pop_map: dict[int, float] = {
        r.census_area_id: r.population for r in pop_result.scalars().all()
    }
    return np.array([pop_map.get(a.id, 0.0) for a in areas], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────── #
# Pre-selection resolver                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

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
    kept_indices: list[int] = []
    new_indices:  list[int] = []

    if is_reoptimization:
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
                        existing_capacity[idx] = cap
                    else:
                        prev = existing_capacity[idx]
                        existing_capacity[idx] = (
                            math.inf if prev == math.inf or cap == math.inf
                            else prev + cap
                        )
            pre_selected_set = set(pre_selected)

    return (
        pre_selected,
        pre_selected_set,
        existing_capacity,
        is_reoptimization,
        kept_indices,
        new_indices,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Unassigned areas builder                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def _compute_unassigned_areas(
    areas: list,
    final_set: list,
    distance_matrix: "SparseDistanceMatrix",
    demand: "np.ndarray",
    assignment_radius,
    scope_area_ids: set,
) -> list:
    """Build the list of unassigned area dicts for areas with demand not covered by any facility."""
    unassigned_areas_out = []
    for idx, a in enumerate(areas):
        if a.id in scope_area_ids or float(demand[idx]) <= 1e-9:
            continue

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


# ─────────────────────────────────────────────────────────────────────────── #
# Background optimization task                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

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

            # 2. Build demand vector.
            demand = await _load_demand(
                db, areas, payload.target_population_id, payload.facility_type
            )

            # 2b. Build coordinate and speed arrays.
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

            # 3. Load distance matrix.
            _t = time.perf_counter()
            distance_matrix = await _load_distance_matrix(db, [a.id for a in areas])
            distance_matrix.xy = xy
            distance_matrix.speeds = speeds
            _timing["3_load_distance_matrix"] = time.perf_counter() - _t

            # Bump Hunter is delegated to its own service.
            if payload.model_type.value == "bump_hunter":
                await run_bump_hunter(
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
                cap_min_f = float(payload.min_capacity or 0.0)
                cap_max_f = float(payload.max_capacity) if payload.max_capacity else math.inf
                _assignment_radius = (
                    float(payload.service_radius) if payload.service_radius else None
                )
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
                effective_p = payload.p_facilities
                if (
                    payload.mode == OptimizationMode.COMPLETE_EXISTING
                    and pre_selected
                    and payload.p_facilities is not None
                ):
                    effective_p = len(pre_selected) + payload.p_facilities

                facility_indices, stats = await asyncio.to_thread(
                    _run_solver, payload, demand, distance_matrix, pre_selected, effective_p
                )
                _timing["6_solver"] = time.perf_counter() - _t

                max_cap = float(payload.max_capacity) if payload.max_capacity else math.inf
                facility_capacities: dict[int, float] = {
                    fac: existing_capacity[fac]
                    if fac in pre_selected_set and fac in existing_capacity
                    else max_cap
                    for fac in facility_indices
                }

                _assignment_radius = (
                    float(payload.service_radius)
                    if payload.model_type == ModelType.MAX_COVERAGE
                    else None
                )
                _t = time.perf_counter()
                final_facility_indices, assignments = await asyncio.to_thread(
                    capacity_assignment,
                    distance_matrix,
                    demand,
                    facility_indices,
                    facility_capacities,
                    payload.min_capacity,
                    pre_selected_set,
                    _assignment_radius,
                )
                _timing["8_capacity_assignment"] = time.perf_counter() - _t
                stats["num_facilities"] = len(final_facility_indices)

            # 9. Deduplicate and invert assignments.
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

            # 10.5. Compute travel-time stats.
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
            stats["covered_demand"]   = round(min(_total_assigned, _total_dem))
            stats["uncovered_demand"] = round(max(_total_dem - _total_assigned, 0.0))
            stats["coverage_pct"] = (
                round(min(_total_assigned, _total_dem) / _total_dem * 100, 2)
                if _total_dem > 0 else 0.0
            )

            # 11. Build full location objects.
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

                _new_indices_set = set(new_indices)
                _is_user_added  = idx in _new_indices_set
                _is_existing    = (idx in pre_selected_set
                                   and not is_reoptimization
                                   and not _is_user_added)
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

            stats["_meta"] = {
                "facility_type": payload.facility_type or "high_school",
                "mode": payload.mode.value,
                "target_population_id": payload.target_population_id,
                "scope_filters": payload.scope_filters.model_dump() if payload.scope_filters else None,
            }

            # 11.5. Compute unassigned areas.
            served_census_ids: set[int] = set()
            for loc in locations:
                served_census_ids.add(loc.census_area_id)
                for sa in loc.served_areas:
                    served_census_ids.add(sa.census_area_id)

            unassigned_areas_out = _compute_unassigned_areas(
                areas, final_facility_indices, distance_matrix, demand,
                _assignment_radius, served_census_ids,
            )

            # 12. Mark scenario completed.
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


# ─────────────────────────────────────────────────────────────────────────── #
# Reoptimization assignment                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

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
    """Constrained nearest-assignment for reoptimization (solver is bypassed)."""
    all_indices = kept_indices + new_indices
    kept_set    = set(kept_indices)
    new_set     = set(new_indices)
    fac_set     = set(all_indices)

    remaining: dict[int, float] = {
        f: (per_facility_cap[f][1] if per_facility_cap and f in per_facility_cap else cap_max)
        for f in all_indices
    }
    fac_load: dict[int, float]  = defaultdict(float)
    assignments: dict[int, list[tuple[int, float]]] = {}

    min_dists  = dm.min_dist_to_set(all_indices)
    area_order = np.argsort(min_dists, kind="stable").tolist()

    for i in area_order:
        d = float(demand[i])
        if d <= 1e-9:
            continue
        rem = d
        area_asgn: list[tuple[int, float]] = []
        for dist, fac in sorted_facilities_for_area(dm, i, fac_set):
            if rem <= 1e-9:
                break
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


# ─────────────────────────────────────────────────────────────────────────── #
# Solver dispatch                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def _run_solver(
    payload: OptimizationRequest,
    demand: np.ndarray,
    distance_matrix: SparseDistanceMatrix,
    pre_selected: list[int],
    override_p: int | None = None,
) -> tuple[list[int], dict]:
    """Dispatch to the appropriate solver and return (facility_indices, stats)."""
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


# ─────────────────────────────────────────────────────────────────────────── #
# Distance-matrix cache and loader                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def _dm_cache_key(area_ids: list[int]) -> int:
    return hash(tuple(sorted(area_ids)))


async def _load_distance_matrix(
    db: AsyncSession, area_ids: list[int]
) -> SparseDistanceMatrix:
    """Load and cache the travel-time matrix for the given area ids.

    Falls back to k-NN Euclidean approximation when the DB table is empty.
    """
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
        _t_extract = time.perf_counter()
        from_ids = np.array([r[0] for r in data], dtype=np.int32)
        to_ids   = np.array([r[1] for r in data], dtype=np.int32)
        times    = np.array([r[2] for r in data], dtype=np.float32)
        _extract_time = time.perf_counter() - _t_extract

        _t_index = time.perf_counter()
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

        if len(_DM_CACHE) >= _DM_CACHE_SIZE:
            _DM_CACHE.pop(next(iter(_DM_CACHE)))
        _DM_CACHE[cache_key] = dm
        return dm

    # Fallback: k-NN Euclidean (dev/test when DB table is empty).
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
