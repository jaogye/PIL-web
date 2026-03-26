"""Pydantic schemas for optimization requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ModelType(str, Enum):
    P_MEDIAN = "p_median"
    P_CENTER = "p_center"
    MAX_COVERAGE = "max_coverage"


class OptimizationMode(str, Enum):
    FROM_SCRATCH = "from_scratch"
    COMPLETE_EXISTING = "complete_existing"


# ------------------------------------------------------------------ #
# Request schemas                                                      #
# ------------------------------------------------------------------ #

class ScopeFilters(BaseModel):
    province_codes: list[str] | None = None
    canton_codes: list[str] | None = None
    parish_codes: list[str] | None = None
    parish_ids: list[int] | None = None  # political_division IDs (preferred)


class OptimizationRequest(BaseModel):
    name: str = Field(..., min_length=3, max_length=255)
    description: str | None = None

    model_type: ModelType
    p_facilities: int = Field(..., ge=1, le=200)

    # Required for MAX_COVERAGE and P_CENTER.
    service_radius: float | None = Field(None, ge=1.0, le=600.0)

    scope_filters: ScopeFilters | None = None
    parameters: dict[str, Any] | None = None

    # "complete_existing": pre-select existing facilities of this type.
    mode: OptimizationMode = OptimizationMode.FROM_SCRATCH
    facility_type: str | None = None

    # Capacity constraints for planned facilities.
    # min_capacity: planned facility is discarded if it serves less than this.
    # max_capacity: maximum demand a planned facility can absorb (None = infinite).
    min_capacity: float | None = Field(None, ge=0.0)
    max_capacity: float | None = Field(None, ge=0.0)

    # Reoptimization: exact list of census_area_ids to treat as fixed (pre-selected) facilities.
    # When provided, overrides the complete_existing mode lookup.
    fixed_census_area_ids: list[int] | None = None

    @field_validator("service_radius")
    @classmethod
    def radius_required_for_coverage(cls, v, info):
        model = info.data.get("model_type")
        if model == ModelType.MAX_COVERAGE and v is None:
            raise ValueError("service_radius is required for max_coverage model")
        return v


# ------------------------------------------------------------------ #
# Response schemas                                                     #
# ------------------------------------------------------------------ #

class ServedAreaInfo(BaseModel):
    """A census area served (fully or partially) by a facility."""
    census_area_id: int
    area_code: str | None = None
    x: float | None = None
    y: float | None = None
    assigned_demand: float
    travel_time: float | None = None  # travel time in minutes from this area to its facility


class FacilityLocation(BaseModel):
    census_area_id: int
    area_code: str
    name: str | None
    x: float | None
    y: float | None
    covered_demand: float | None
    assigned_areas: int | None
    max_travel_time: float | None
    # True when this facility was pre-selected as an existing facility.
    is_existing: bool = False
    # Census areas served by this facility (for map service-area visualization).
    served_areas: list[ServedAreaInfo] = []


class OptimizationResponse(BaseModel):
    scenario_id: int
    name: str
    model_type: ModelType
    p_facilities: int
    service_radius: float | None
    status: str
    facility_locations: list[FacilityLocation]
    stats: dict[str, Any]


# ------------------------------------------------------------------ #
# Rebalancing schemas                                                  #
# ------------------------------------------------------------------ #

class RebalancingRequest(BaseModel):
    """Parameters for the capacity rebalancing algorithm."""
    capacity_per_facility: float | None = Field(None, ge=1.0,
        description="Uniform capacity target for all facilities. "
                    "Defaults to average covered demand per facility.")
    min_capacity: float = Field(0.0, ge=0.0,
        description="Operational floor: each facility retains at least this capacity.")
    max_transfers: int = Field(20, ge=1, le=100,
        description="Maximum number of transfer operations to propose.")


class RebalancingTransferInfo(BaseModel):
    """A single capacity transfer with geographic coordinates."""
    from_census_area_id: int
    from_area_code: str | None = None
    from_x: float | None = None
    from_y: float | None = None
    to_census_area_id: int
    to_area_code: str | None = None
    to_x: float | None = None
    to_y: float | None = None
    amount: float
    impact: float


class FacilityCapacityInfo(BaseModel):
    census_area_id: int
    area_code: str | None = None
    covered_demand: float
    old_capacity: float
    new_capacity: float


class RebalancingResponse(BaseModel):
    scenario_id: int
    transfers: list[RebalancingTransferInfo]
    facility_capacities: list[FacilityCapacityInfo]
    unmet_demand_before: float
    unmet_demand_after: float
    improvement_pct: float
    stats: dict


class ScenarioSummary(BaseModel):
    id: int
    name: str
    model_type: ModelType
    p_facilities: int
    status: str
    created_at: str
    stats: dict[str, Any] | None

    class Config:
        from_attributes = True
