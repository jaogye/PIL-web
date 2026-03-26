"""
REST endpoints for report generation.

GET /reports/scenario/{id}/excel  – Download a scenario report as .xlsx
GET /reports/scenario/{id}/json   – Download full scenario data as JSON
"""

import io
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_db_name
from app.models.census import CensusArea
from app.models.optimization import OptimizationResult, OptimizationScenario

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/scenario/{scenario_id}/excel")
async def export_scenario_excel(
    scenario_id: int, db: AsyncSession = Depends(get_db)
):
    """Export scenario results as an Excel workbook (.xlsx)."""
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        raise HTTPException(status_code=500, detail="openpyxl is not installed")

    scenario = await db.get(OptimizationScenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    res = await db.execute(
        select(OptimizationResult, CensusArea)
        .join(CensusArea, OptimizationResult.census_area_id == CensusArea.id)
        .where(OptimizationResult.scenario_id == scenario_id)
    )
    rows = res.all()

    # Pre-load all census areas for the served-areas sheet.
    # served_area_ids format: [[census_area_id, demand_amount], ...]
    all_area_ids: set[int] = set()
    for opt_res, _ in rows:
        for entry in (opt_res.served_area_ids or []):
            all_area_ids.add(int(entry[0]) if isinstance(entry, (list, tuple)) else int(entry))
    area_map: dict[int, CensusArea] = {}
    if all_area_ids:
        area_res = await db.execute(
            select(CensusArea).where(CensusArea.id.in_(list(all_area_ids)))
        )
        area_map = {a.id: a for a in area_res.scalars().all()}

    wb = openpyxl.Workbook()

    # --- Sheet 1: Scenario Summary ---
    ws_summary = wb.active
    ws_summary.title = "Summary"
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(fill_type="solid", fgColor="1F4E79")

    ws_summary.append(["LIP2 – Optimization Scenario Report"])
    ws_summary["A1"].font = Font(bold=True, size=14)
    ws_summary.append([])
    ws_summary.append(["Scenario ID", scenario.id])
    ws_summary.append(["Name", scenario.name])
    ws_summary.append(["Model", scenario.model_type.value])
    ws_summary.append(["Facilities (p)", scenario.p_facilities])
    ws_summary.append(["Service Radius (min)", scenario.service_radius or "N/A"])
    ws_summary.append(["Status", scenario.status.value])
    ws_summary.append(["Generated", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")])
    ws_summary.append([])

    if scenario.result_stats:
        ws_summary.append(["── Statistics ──"])
        for key, value in scenario.result_stats.items():
            ws_summary.append([key.replace("_", " ").title(), value])

    ws_summary.column_dimensions["A"].width = 30
    ws_summary.column_dimensions["B"].width = 25

    # --- Sheet 2: Facility Locations ---
    ws_fac = wb.create_sheet("Facility Locations")
    headers = ["#", "Area Code", "Area Name", "Province", "Canton", "Parish",
               "X (Lon)", "Y (Lat)", "Demand Covered"]
    ws_fac.append(headers)

    for cell in ws_fac[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for i, (opt_res, area) in enumerate(rows, start=1):
        ws_fac.append([
            i,
            area.area_code,
            area.name or "",
            area.province_code,
            area.canton_code,
            area.parish_code,
            area.x,
            area.y,
            round(opt_res.covered_demand or 0.0, 2),
        ])

    for col in ws_fac.columns:
        max_len = max(len(str(cell.value or "")) for cell in col) + 2
        ws_fac.column_dimensions[col[0].column_letter].width = min(max_len, 30)

    # --- Sheet 3: Served Areas ---
    ws_served = wb.create_sheet("Served Areas")
    served_headers = [
        "Facility #", "Facility Code", "Facility Name",
        "Area Code", "Area Name", "Province", "Canton", "Parish",
        "X (Lon)", "Y (Lat)", "Demand Assigned",
    ]
    ws_served.append(served_headers)
    for cell in ws_served[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for fac_num, (opt_res, fac_area) in enumerate(rows, start=1):
        for entry in (opt_res.served_area_ids or []):
            # Support both old format (plain int) and new format ([area_id, demand]).
            if isinstance(entry, (list, tuple)):
                aid, demand_amt = int(entry[0]), float(entry[1])
            else:
                aid, demand_amt = int(entry), 0.0
            sa = area_map.get(aid)
            if sa is None:
                continue
            ws_served.append([
                fac_num,
                fac_area.area_code,
                fac_area.name or "",
                sa.area_code,
                sa.name or "",
                sa.province_code,
                sa.canton_code,
                sa.parish_code,
                sa.x,
                sa.y,
                round(demand_amt, 2),
            ])

    for col in ws_served.columns:
        max_len = max(len(str(cell.value or "")) for cell in col) + 2
        ws_served.column_dimensions[col[0].column_letter].width = min(max_len, 30)

    # Serialize to bytes.
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"lip2_scenario_{scenario_id}_{datetime.utcnow().strftime('%Y%m%d')}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/scenario/{scenario_id}/json")
async def export_scenario_json(
    scenario_id: int, db: AsyncSession = Depends(get_db)
):
    """Export full scenario data as a JSON download."""
    scenario = await db.get(OptimizationScenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    res = await db.execute(
        select(OptimizationResult, CensusArea)
        .join(CensusArea, OptimizationResult.census_area_id == CensusArea.id)
        .where(OptimizationResult.scenario_id == scenario_id)
    )
    rows = res.all()

    payload = {
        "scenario": {
            "id": scenario.id,
            "name": scenario.name,
            "model_type": scenario.model_type.value,
            "p_facilities": scenario.p_facilities,
            "service_radius": scenario.service_radius,
            "status": scenario.status.value,
            "stats": scenario.result_stats,
            "created_at": scenario.created_at.isoformat() if scenario.created_at else None,
        },
        "facility_locations": [
            {
                "area_id": area.id,
                "area_code": area.area_code,
                "name": area.name,
                "province_code": area.province_code,
                "canton_code": area.canton_code,
                "parish_code": area.parish_code,
                "x": area.x,
                "y": area.y,
                "covered_demand": opt_res.covered_demand,
            }
            for opt_res, area in rows
        ],
    }

    import json

    content = json.dumps(payload, indent=2, ensure_ascii=False)
    filename = f"lip2_scenario_{scenario_id}.json"
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
