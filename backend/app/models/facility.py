"""
ORM models for existing and planned public infrastructure facilities.

Corresponds to: scr.optimiza.ClusterItem
                scr.planificador.fInfraActual
"""

import enum

from geoalchemy2 import Geometry
from sqlalchemy import Enum, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class FacilityType(str, enum.Enum):
    SCHOOL = "school"
    HIGH_SCHOOL = "high_school"
    HEALTH_CENTER = "health_center"
    HOSPITAL = "hospital"
    NURSERY = "nursery"
    OTHER = "other"


class FacilityStatus(str, enum.Enum):
    EXISTING = "existing"
    PROPOSED = "proposed"
    OPTIMIZED = "optimized"


class Facility(Base):
    __tablename__ = "facilities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    name: Mapped[str | None] = mapped_column(String(255))

    # facility_type references the facility_types lookup table (not an ENUM).
    facility_type: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("facility_types.code"),
        nullable=False,
        default=FacilityType.OTHER.value,
    )

    status: Mapped[FacilityStatus] = mapped_column(
        Enum(FacilityStatus, name="facility_status", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=FacilityStatus.EXISTING,
    )

    # Capacity in service units (students, patients/day, etc.).
    capacity: Mapped[float] = mapped_column(Float, default=0.0)

    # Link to the census area where this facility sits.
    census_area_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("census_areas.id"), nullable=True
    )
    census_area: Mapped["CensusArea"] = relationship(lazy="select")  # type: ignore[name-defined]

    geom: Mapped[object] = mapped_column(Geometry("POINT", srid=4326), nullable=True)

    __table_args__ = (
        Index("idx_facility_geom", "geom", postgresql_using="gist"),
    )

    def __repr__(self) -> str:
        return f"<Facility id={self.id} type={self.facility_type} status={self.status}>"


class FacilityTypeLookup(Base):
    """ORM representation of the facility_types lookup table."""

    __tablename__ = "facility_types"

    code:                         Mapped[str]        = mapped_column(String(50), primary_key=True)
    label:                        Mapped[str]        = mapped_column(String(255), nullable=False)
    default_target_population_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("target_population.id"), nullable=True
    )

    def __repr__(self) -> str:
        return f"<FacilityTypeLookup code={self.code}>"
