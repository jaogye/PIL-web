"""
REST endpoints for target populations and facility types.

GET /target-populations/          – List all demographic target groups.
GET /target-populations/facility-types – List facility types with their default target group.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.models.facility import FacilityTypeLookup
from app.models.target_population import TargetPopulation

router = APIRouter(prefix="/target-populations", tags=["target-populations"])


class TargetPopulationOut(BaseModel):
    id: int
    code: str
    label: str
    min_age: int | None
    max_age: int | None

    class Config:
        from_attributes = True


class FacilityTypeOut(BaseModel):
    code: str
    label: str
    default_target_population_id: int | None

    class Config:
        from_attributes = True


@router.get("/", response_model=list[TargetPopulationOut])
async def list_target_populations(db: AsyncSession = Depends(get_db)):
    """Return all demographic target population groups ordered by min_age."""
    result = await db.execute(
        select(TargetPopulation).order_by(TargetPopulation.id)
    )
    return result.scalars().all()


@router.get("/facility-types", response_model=list[FacilityTypeOut])
async def list_facility_types(db: AsyncSession = Depends(get_db)):
    """Return all facility types with their default target population id."""
    result = await db.execute(select(FacilityTypeLookup).order_by(FacilityTypeLookup.code))
    return result.scalars().all()
