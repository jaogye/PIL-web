"""ORM models for target population groups and per-area population figures."""

from sqlalchemy import Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TargetPopulation(Base):
    __tablename__ = "target_population"

    id:      Mapped[int]        = mapped_column(Integer, primary_key=True, autoincrement=True)
    code:    Mapped[str]        = mapped_column(String(50), unique=True, nullable=False)
    label:   Mapped[str]        = mapped_column(String(255), nullable=False)
    min_age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_age: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<TargetPopulation code={self.code}>"


class CensusAreaPopulation(Base):
    __tablename__ = "census_areas_population"

    id:                   Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    census_area_id:       Mapped[int]   = mapped_column(
        Integer, ForeignKey("census_areas.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_population_id: Mapped[int]   = mapped_column(
        Integer, ForeignKey("target_population.id"), nullable=False, index=True
    )
    population:           Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        UniqueConstraint("census_area_id", "target_population_id", name="uq_cap_area_tp"),
    )

    def __repr__(self) -> str:
        return f"<CensusAreaPopulation area={self.census_area_id} tp={self.target_population_id} pop={self.population}>"
