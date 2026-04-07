"""
ORM model for census areas (AreaItem in the original Java code).

Each row represents one census track (area censal) with its geographic
location, administrative codes, and population demand.
"""

from geoalchemy2 import Geometry
from sqlalchemy import Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class CensusArea(Base):
    __tablename__ = "census_areas"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Administrative codes for province / canton / parish hierarchy.
    province_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    canton_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    parish_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    area_code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    
    name: Mapped[str | None] = mapped_column(String(255))
    
    # Service capacity of existing facilities at this area (students, patients/day, etc.).
    capacity: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Geographic coordinates (WGS84 EPSG:4326).
    geom: Mapped[object] = mapped_column(Geometry("POINT", srid=4326), nullable=True)
    
    # Raw X/Y as floats for fast in-memory operations without PostGIS.
    x: Mapped[float | None] = mapped_column(Float)
    y: Mapped[float | None] = mapped_column(Float)
    
    # Median travel speed (km/h) derived from stored distance_matrix neighbours.
    # Used to estimate travel times to areas not stored in distance_matrix.
    avg_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # FK to political_division (parroquia level).
    parish_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("political_division.id"), nullable=True
    )
    
    __table_args__ = (
        Index("idx_census_geom", "geom", postgresql_using="gist"),
        Index("idx_census_admin", "province_code", "canton_code", "parish_code"),
    )
    
    def __repr__(self) -> str:
        return f"<CensusArea code={self.area_code}>"
