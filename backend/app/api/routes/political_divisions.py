"""
REST endpoints for political divisions (provincia > canton > parroquia).

GET  /political-divisions/tree            – Full hierarchy tree.
POST /political-divisions/census-summary  – Count and demand for selected parroquias.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_db_name
from app.models.census import CensusArea
from app.models.political_division import PoliticalDivision

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


@router.post("/census-summary")
async def census_summary(
    payload: CensusSummaryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Return count and total demand of census areas for a set of parroquia IDs."""
    if not payload.parish_ids:
        return {"census_area_count": 0, "total_demand": 0.0}

    result = await db.execute(
        select(
            func.count(CensusArea.id).label("count"),
            func.coalesce(func.sum(CensusArea.demand), 0.0).label("demand"),
        ).where(CensusArea.parish_id.in_(payload.parish_ids))
    )
    row = result.one()
    return {"census_area_count": int(row.count), "total_demand": float(row.demand)}
