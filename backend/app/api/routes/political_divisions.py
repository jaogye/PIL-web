"""
REST endpoints for political divisions (provincia > canton > parroquia).

GET  /political-divisions/tree            – Full hierarchy tree.
POST /political-divisions/census-summary  – Count and demand for selected parroquias.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import join
from app.dependencies import get_db, get_db_name
from app.models.census import CensusArea
from app.models.political_division import PoliticalDivision
from app.models.target_population import CensusAreaPopulation, TargetPopulation

router = APIRouter(prefix="/political-divisions", tags=["political-divisions"])


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #

@router.get("/tree")
async def get_tree(db: AsyncSession = Depends(get_db)):
    """Return the full provincia > canton > parroquia hierarchy.

    Loads all rows in one query and assembles the tree in Python
    to avoid lazy-loading issues with async SQLAlchemy.
    """
    result = await db.execute(
        select(PoliticalDivision).order_by(PoliticalDivision.name)
    )
    all_divs = result.scalars().all()

    # Index nodes and build tree in Python (no ORM relationship access).
    nodes: dict[int, dict] = {}
    for d in all_divs:
        nodes[d.id] = {
            "id": d.id,
            "code": d.code,
            "name": d.name,
            "level": d.level,
            "children": [],
        }

    roots = []
    for d in all_divs:
        node = nodes[d.id]
        if d.parent_id is None:
            roots.append(node)
        elif d.parent_id in nodes:
            nodes[d.parent_id]["children"].append(node)

    return roots


class CensusSummaryRequest(BaseModel):
    parish_ids: list[int]
    target_population_id: int | None = None  # if None, returns all_ages total


async def _pop_sum(db, area_ids_filter, tp_id: int):
    """Return sum of census_areas_population for a given target_population_id."""
    result = await db.execute(
        select(
            func.count(CensusArea.id).label("count"),
            func.coalesce(func.sum(CensusAreaPopulation.population), 0.0).label("pop"),
        )
        .join(
            CensusAreaPopulation,
            (CensusAreaPopulation.census_area_id == CensusArea.id)
            & (CensusAreaPopulation.target_population_id == tp_id),
            isouter=True,
        )
        .where(area_ids_filter)
    )
    return result.one()


@router.post("/census-summary")
async def census_summary(
    payload: CensusSummaryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Return census area count, total population, and target population for a scope."""
    if not payload.parish_ids:
        return {"census_area_count": 0, "total_demand": 0.0, "target_population": 0.0}

    area_filter = CensusArea.parish_id.in_(payload.parish_ids)

    # Always fetch all_ages total.
    tp_all = await db.execute(
        select(TargetPopulation).where(TargetPopulation.code == "all_ages")
    )
    tp_all = tp_all.scalar_one_or_none()

    if tp_all:
        row_all = await _pop_sum(db, area_filter, tp_all.id)
        census_count = int(row_all.count)
        total_demand = float(row_all.pop)
    else:
        count_result = await db.execute(
            select(func.count(CensusArea.id)).where(area_filter)
        )
        census_count = int(count_result.scalar())
        total_demand = 0.0

    # Fetch target-population-specific count if requested.
    target_pop = total_demand  # default: same as total
    if payload.target_population_id and payload.target_population_id != (tp_all.id if tp_all else None):
        row_tp = await _pop_sum(db, area_filter, payload.target_population_id)
        target_pop = float(row_tp.pop)

    return {
        "census_area_count": census_count,
        "total_demand":      total_demand,
        "target_population": target_pop,
    }
