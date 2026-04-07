"""
REST endpoints for existing and planned infrastructure (facilities).

GET    /infrastructure/           – List facilities with optional filters.
POST   /infrastructure/           – Create a new facility.
GET    /infrastructure/{id}       – Get a single facility.
PUT    /infrastructure/{id}       – Update a facility.
DELETE /infrastructure/{id}       – Delete a facility.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_db_name
from app.models.census import CensusArea
from app.models.facility import Facility, FacilityStatus, FacilityType

router = APIRouter(prefix="/infrastructure", tags=["infrastructure"])


# ------------------------------------------------------------------ #
# Schemas (inline for brevity)                                         #
# ------------------------------------------------------------------ #

class FacilityIn(BaseModel):
    name: str | None = None
    facility_type: FacilityType = FacilityType.OTHER
    status: FacilityStatus = FacilityStatus.EXISTING
    capacity: float = 0.0
    census_area_id: int | None = None
    x: float | None = None
    y: float | None = None


class FacilityOut(BaseModel):
    id: int
    name: str | None
    facility_type: FacilityType
    status: FacilityStatus
    capacity: float
    census_area_id: int | None
    x: float | None = None
    y: float | None = None

    class Config:
        from_attributes = True


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #

@router.get("/", response_model=list[FacilityOut])
async def list_facilities(
    facility_type: FacilityType | None = Query(None),
    facility_status: FacilityStatus | None = Query(None, alias="status"),
    parish_ids: list[int] | None = Query(None),
    province_codes: list[str] | None = Query(None),
    canton_codes: list[str] | None = Query(None),
    parish_codes: list[str] | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(Facility, CensusArea.x, CensusArea.y)
        .outerjoin(CensusArea, Facility.census_area_id == CensusArea.id)
    )
    if facility_type:
        query = query.where(Facility.facility_type == facility_type)
    if facility_status:
        query = query.where(Facility.status == facility_status)
    # Geographic scope filters — parish_ids takes priority over code-based filters.
    if parish_ids:
        query = query.where(CensusArea.parish_id.in_(parish_ids))
    elif province_codes:
        query = query.where(CensusArea.province_code.in_(province_codes))
        if canton_codes:
            query = query.where(CensusArea.canton_code.in_(canton_codes))
        if parish_codes:
            query = query.where(CensusArea.parish_code.in_(parish_codes))
    elif canton_codes:
        query = query.where(CensusArea.canton_code.in_(canton_codes))
        if parish_codes:
            query = query.where(CensusArea.parish_code.in_(parish_codes))
    elif parish_codes:
        query = query.where(CensusArea.parish_code.in_(parish_codes))

    result = await db.execute(query)
    rows = result.all()
    out = []
    for fac, cx, cy in rows:
        out.append(FacilityOut(
            id=fac.id,
            name=fac.name,
            facility_type=fac.facility_type,
            status=fac.status,
            capacity=fac.capacity,
            census_area_id=fac.census_area_id,
            x=cx,
            y=cy,
        ))
    return out


@router.post("/", response_model=FacilityOut, status_code=status.HTTP_201_CREATED)
async def create_facility(payload: FacilityIn, db: AsyncSession = Depends(get_db)):
    facility = Facility(**payload.model_dump(exclude={"x", "y"}))
    db.add(facility)
    await db.flush()
    return facility


@router.get("/{facility_id}", response_model=FacilityOut)
async def get_facility(facility_id: int, db: AsyncSession = Depends(get_db)):
    facility = await db.get(Facility, facility_id)
    if not facility:
        raise HTTPException(status_code=404, detail="Facility not found")
    return facility


@router.put("/{facility_id}", response_model=FacilityOut)
async def update_facility(
    facility_id: int, payload: FacilityIn, db: AsyncSession = Depends(get_db)
):
    facility = await db.get(Facility, facility_id)
    if not facility:
        raise HTTPException(status_code=404, detail="Facility not found")

    for field, value in payload.model_dump(exclude_unset=True, exclude={"x", "y"}).items():
        setattr(facility, field, value)

    return facility


@router.delete("/{facility_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_facility(facility_id: int, db: AsyncSession = Depends(get_db)):
    facility = await db.get(Facility, facility_id)
    if not facility:
        raise HTTPException(status_code=404, detail="Facility not found")
    await db.delete(facility)
