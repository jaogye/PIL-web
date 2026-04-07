"""
ORM models for optimization scenarios and their results.

A Scenario stores the input parameters for one optimization run.
OptimizationResult stores which census areas were selected as facilities.
"""

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ModelType(str, enum.Enum):
    P_MEDIAN = "p_median"
    P_CENTER = "p_center"
    MAX_COVERAGE = "max_coverage"
    BUMP_HUNTER = "bump_hunter"


class ScenarioStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizationScenario(Base):
    __tablename__ = "optimization_scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000))

    model_type: Mapped[ModelType] = mapped_column(
        Enum(ModelType, name="model_type", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    p_facilities: Mapped[int] = mapped_column(Integer, nullable=False)

    # Used only for MCLP and p-center.
    service_radius: Mapped[float | None] = mapped_column(Float)

    # Administrative scope filters (province / canton / parish codes).
    scope_filters: Mapped[dict | None] = mapped_column(JSON)

    # Additional algorithm parameters (e.g. capacity constraints).
    parameters: Mapped[dict | None] = mapped_column(JSON)

    status: Mapped[ScenarioStatus] = mapped_column(
        Enum(ScenarioStatus, name="scenario_status", values_callable=lambda x: [e.value for e in x]),
        default=ScenarioStatus.PENDING,
    )

    # Summary metrics returned by the solver.
    result_stats: Mapped[dict | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(String(2000))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    results: Mapped[list["OptimizationResult"]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Scenario id={self.id} model={self.model_type} p={self.p_facilities}>"


class OptimizationResult(Base):
    __tablename__ = "optimization_results"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    scenario_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("optimization_scenarios.id", ondelete="CASCADE"), nullable=False
    )
    scenario: Mapped[OptimizationScenario] = relationship(back_populates="results")
    
    # Census area selected as a facility location.
    census_area_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("census_areas.id"), nullable=False
    )
    
    # Demand covered by this specific facility.
    covered_demand: Mapped[float | None] = mapped_column(Float)
    
    # Number of census areas assigned to this facility.
    assigned_areas: Mapped[int | None] = mapped_column(Integer)
    
    # Maximum travel time from any assigned area to this facility.
    max_travel_time: Mapped[float | None] = mapped_column(Float)
    
    # IDs of census areas served by this facility (for Excel sheet).
    served_area_ids: Mapped[list | None] = mapped_column(JSON)
    
    def __repr__(self) -> str:
        return f"<OptimizationResult scenario={self.scenario_id} area={self.census_area_id}>"
