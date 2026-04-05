"""Pydantic schemas for optimization requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ModelType(str, Enum):
    P_MEDIAN = "p_median"
    P_CENTER = "p_center"
    MAX_COVERAGE = "max_coverage"
    BUMP_HUNTER = "bump_hunter"


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
    # Required for all models except bump_hunter (which has no fixed p).
    p_facilities: int | None = Field(None, ge=1, le=500)

    # Required for MAX_COVERAGE and BUMP_HUNTER.
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

    # Reoptimization: exact list of census_area_ids to treat as fixed facilities.
    # When provided, the solver is skipped; a constrained nearest-assignment is
    # run directly over these facilities.  Overrides the complete_existing mode.
    fixed_census_area_ids: list[int] | None = None

    # Subset of fixed_census_area_ids containing only facilities newly placed by
    # the user during the reoptimization step (right-click "Add Facility Here").
    # Facilities in fixed_census_area_ids but NOT in this list are treated as
    # "kept" (from the previous optimization run) and must retain at least
    # min_capacity demand after assignment.
    # For max_coverage, user-created facilities only accept areas within the
    # service radius.
    reopt_new_facility_ids: list[int] | None = None

    # Target population to use as demand (references target_population.id).
    # If None, the default for the selected facility_type is used.
    target_population_id: int | None = None

    @field_validator("p_facilities")
    @classmethod
    def p_required_for_non_bump(cls, v, info):
        model = info.data.get("model_type")
        if model not in (ModelType.BUMP_HUNTER, ModelType.MAX_COVERAGE) and v is None:
            raise ValueError("p_facilities is required for this model type")
        return v

    @field_validator("service_radius")
    @classmethod
    def radius_required_for_coverage(cls, v, info):
        model = info.data.get("model_type")
        if model in (ModelType.MAX_COVERAGE, ModelType.BUMP_HUNTER) and v is None:
            raise ValueError("service_radius is required for max_coverage and bump_hunter models")
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
    avg_speed_kmh: float | None = None  # stored vpd for this census area


class FacilityLocation(BaseModel):
    census_area_id: int
    area_code: str
    name: str | None
    x: float | None
    y: float | None
    covered_demand: float | None
    assigned_areas: int | None
    max_travel_time: float | None
    avg_speed_kmh: float | None = None  # stored vpd for the facility census area
    # True when this facility came from the infrastructure database (complete_existing mode).
    is_existing: bool = False
    # True when this facility was manually added by the user during reoptimization.
    is_user_added: bool = False
    # Census areas served by this facility (for map service-area visualization).
    served_areas: list[ServedAreaInfo] = []


class UnassignedAreaInfo(BaseModel):
    """A census area in the scope that was not assigned to any facility."""
    census_area_id: int
    area_code: str | None = None
    name: str | None = None
    x: float | None = None
    y: float | None = None
    demand: float | None = None
    nearest_facility_code: str | None = None
    nearest_facility_travel_time_min: float | None = None
    nearest_facility_distance_km: float | None = None
    travel_speed_kmh: float | None = None  # speed computed on-the-fly to nearest facility
    avg_speed_kmh: float | None = None      # stored vpd for this census area


class OptimizationResponse(BaseModel):
    scenario_id: int
    name: str
    model_type: ModelType
    p_facilities: int
    service_radius: float | None
    status: str
    facility_locations: list[FacilityLocation]
    stats: dict[str, Any]
    # Census areas in scope with demand > 0 that were not served by any facility.
    unassigned_areas: list[UnassignedAreaInfo] = []


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
