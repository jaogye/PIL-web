"""
Unit tests for the optimization algorithms.

Run with: pytest tests/test_optimization.py -v
"""

import numpy as np
import pytest

from app.optimization.p_median import solve as p_median
from app.optimization.p_center import solve as p_center
from app.optimization.max_coverage import solve as max_coverage
from app.optimization.rebalancing import solve as rebalance


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture
def small_grid():
    """
    5-node grid in a line: 0 - 1 - 2 - 3 - 4
    Distances are integer travel times (minutes).
    Demand is uniform.
    """
    n = 5
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = abs(i - j) * 10  # 10 min per step

    demand = np.ones(n) * 100.0
    return dist, demand


@pytest.fixture
def asymmetric_demand():
    """
    5-node grid where node 4 has 10x the demand of others.
    """
    n = 5
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = abs(i - j) * 10

    demand = np.array([100.0, 100.0, 100.0, 100.0, 1000.0])
    return dist, demand


# ------------------------------------------------------------------ #
# P-Median tests                                                       #
# ------------------------------------------------------------------ #

class TestPMedian:
    def test_returns_correct_number_of_facilities(self, small_grid):
        dist, demand = small_grid
        result = p_median(dist, demand, p=2)
        assert len(result.facility_indices) == 2

    def test_no_duplicate_facilities(self, small_grid):
        dist, demand = small_grid
        result = p_median(dist, demand, p=3)
        assert len(set(result.facility_indices)) == 3

    def test_assignment_length_matches_areas(self, small_grid):
        dist, demand = small_grid
        result = p_median(dist, demand, p=2)
        assert len(result.assignment) == len(demand)

    def test_total_cost_is_positive(self, small_grid):
        dist, demand = small_grid
        result = p_median(dist, demand, p=2)
        assert result.total_cost >= 0.0

    def test_more_facilities_reduce_cost(self, small_grid):
        dist, demand = small_grid
        cost_p1 = p_median(dist, demand, p=1).total_cost
        cost_p2 = p_median(dist, demand, p=2).total_cost
        cost_p3 = p_median(dist, demand, p=3).total_cost
        assert cost_p1 >= cost_p2 >= cost_p3

    def test_high_demand_area_attracts_facility(self, asymmetric_demand):
        dist, demand = asymmetric_demand
        result = p_median(dist, demand, p=1)
        # With one facility, it should be placed at node 4 (highest demand).
        assert result.facility_indices[0] == 4

    def test_p_equals_n_gives_zero_cost(self, small_grid):
        dist, demand = small_grid
        n = len(demand)
        result = p_median(dist, demand, p=n)
        assert result.total_cost == 0.0


# ------------------------------------------------------------------ #
# P-Center tests                                                       #
# ------------------------------------------------------------------ #

class TestPCenter:
    def test_returns_correct_number_of_facilities(self, small_grid):
        dist, demand = small_grid
        result = p_center(dist, demand, p=2)
        assert len(result.facility_indices) == 2

    def test_optimal_radius_is_nonnegative(self, small_grid):
        dist, demand = small_grid
        result = p_center(dist, demand, p=2)
        assert result.optimal_radius >= 0.0

    def test_more_facilities_reduce_radius(self, small_grid):
        dist, demand = small_grid
        r1 = p_center(dist, demand, p=1).optimal_radius
        r2 = p_center(dist, demand, p=2).optimal_radius
        r3 = p_center(dist, demand, p=3).optimal_radius
        assert r1 >= r2 >= r3

    def test_p_equals_n_gives_zero_radius(self, small_grid):
        dist, demand = small_grid
        n = len(demand)
        result = p_center(dist, demand, p=n)
        assert result.optimal_radius == 0.0


# ------------------------------------------------------------------ #
# Max Coverage tests                                                   #
# ------------------------------------------------------------------ #

class TestMaxCoverage:
    def test_returns_correct_number_of_facilities(self, small_grid):
        dist, demand = small_grid
        result = max_coverage(dist, demand, p=2, radius=15.0)
        assert len(result.facility_indices) <= 2

    def test_coverage_increases_with_radius(self, small_grid):
        dist, demand = small_grid
        r_small = max_coverage(dist, demand, p=2, radius=5.0).coverage_pct
        r_large = max_coverage(dist, demand, p=2, radius=30.0).coverage_pct
        assert r_large >= r_small

    def test_full_coverage_with_enough_radius(self, small_grid):
        dist, demand = small_grid
        result = max_coverage(dist, demand, p=3, radius=50.0)
        assert result.coverage_pct == 100.0

    def test_covered_demand_leq_total(self, small_grid):
        dist, demand = small_grid
        result = max_coverage(dist, demand, p=2, radius=15.0)
        assert result.covered_demand <= result.total_demand + 1e-9


# ------------------------------------------------------------------ #
# Rebalancing tests                                                    #
# ------------------------------------------------------------------ #

class TestRebalancing:
    def test_transfers_reduce_unmet_demand(self, small_grid):
        dist, demand = small_grid
        facility_indices = [0, 4]
        # Node 0 has excess capacity, node 4 has deficit.
        capacity = np.array([800.0, 50.0])

        result = rebalance(dist, demand, facility_indices, capacity)
        assert result.unmet_demand_after <= result.unmet_demand_before

    def test_no_transfers_when_balanced(self, small_grid):
        dist, demand = small_grid
        facility_indices = [0, 4]
        # Both facilities have more capacity than demand assigned.
        capacity = np.array([500.0, 500.0])

        result = rebalance(dist, demand, facility_indices, capacity)
        assert result.unmet_demand_after == 0.0
