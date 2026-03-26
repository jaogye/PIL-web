"""
Resource Rebalancing Algorithm.

Ported from Java: scr.planificador.fRebalanceo.java

Rebalancing redistributes capacity (staff, equipment, budget) across
existing facilities to improve population coverage and reduce inequities.

The algorithm identifies:
  - Over-served facilities: capacity > actual demand assigned.
  - Under-served facilities: demand > current capacity (coverage gaps).

It then proposes capacity transfers that maximise the reduction in
unmet demand while respecting minimum operational thresholds.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .sparse_matrix import SparseDistanceMatrix


@dataclass
class RebalancingTransfer:
    """A single capacity transfer between two facilities."""

    from_facility: int
    """Source facility index (has surplus capacity)."""

    to_facility: int
    """Destination facility index (has unmet demand)."""

    amount: float
    """Capacity units to transfer."""

    impact: float
    """Estimated reduction in unmet demand from this transfer."""


@dataclass
class RebalancingResult:
    """Result returned by the rebalancing solver."""

    transfers: list[RebalancingTransfer]
    """List of recommended capacity transfers."""

    new_capacity: list[float]
    """Updated capacity for each facility after all transfers."""

    unmet_demand_before: float
    unmet_demand_after: float
    improvement_pct: float

    stats: dict = field(default_factory=dict)


def solve(
    distance_matrix: SparseDistanceMatrix,
    demand: np.ndarray,
    facility_indices: list[int],
    facility_capacity: np.ndarray,
    min_capacity: float = 0.0,
    max_transfers: int = 20,
) -> RebalancingResult:
    """
    Rebalance capacity across existing facilities.

    Parameters
    ----------
    distance_matrix : SparseDistanceMatrix
        Sparse neighbor-list distance matrix.
    demand : np.ndarray, shape (n,)
        Population demand for each census area.
    facility_indices : list[int]
        Indices of existing facility locations.
    facility_capacity : np.ndarray, shape (len(facility_indices),)
        Current capacity of each facility.
    min_capacity : float
        Minimum capacity a facility must retain (operational floor).
    max_transfers : int
        Maximum number of transfer operations to propose.

    Returns
    -------
    RebalancingResult
    """
    dm = distance_matrix
    p = len(facility_indices)

    # Assign each census area to its nearest facility via sparse matrix.
    assignment = dm.assign(facility_indices)          # area_idx → facility_area_idx
    fac_to_k = {fac: k for k, fac in enumerate(facility_indices)}
    assigned_demand = np.zeros(p)
    for area_i, fac in enumerate(assignment):
        k = fac_to_k.get(fac)
        if k is not None:
            assigned_demand[k] += demand[area_i]

    capacity = facility_capacity.astype(float).copy()

    # Unmet demand: max(0, assigned_demand - capacity).
    unmet_before = np.maximum(0.0, assigned_demand - capacity)
    total_unmet_before = float(np.sum(unmet_before))

    transfers: list[RebalancingTransfer] = []

    for _ in range(max_transfers):
        surplus = capacity - assigned_demand          # positive → over-served
        unmet = np.maximum(0.0, assigned_demand - capacity)  # positive → under-served

        if np.sum(unmet) < 1e-6:
            break  # All demand is served.

        # Find the facility with the most unmet demand.
        worst = int(np.argmax(unmet))
        if unmet[worst] < 1e-6:
            break

        # Find the best donor: highest surplus, respecting min_capacity floor.
        available = np.maximum(0.0, surplus - min_capacity)
        available[worst] = 0.0  # Cannot transfer to itself.

        best_donor = int(np.argmax(available))
        if available[best_donor] < 1e-6:
            break  # No donor has surplus above the floor.

        # Transfer as much as needed (but not more than available).
        amount = min(available[best_donor], unmet[worst])
        impact = min(amount, unmet[worst])

        transfers.append(
            RebalancingTransfer(
                from_facility=facility_indices[best_donor],
                to_facility=facility_indices[worst],
                amount=round(amount, 4),
                impact=round(impact, 4),
            )
        )

        capacity[best_donor] -= amount
        capacity[worst] += amount

    unmet_after = np.maximum(0.0, assigned_demand - capacity)
    total_unmet_after = float(np.sum(unmet_after))

    improvement = (
        (total_unmet_before - total_unmet_after) / total_unmet_before * 100
        if total_unmet_before > 0
        else 0.0
    )

    return RebalancingResult(
        transfers=transfers,
        new_capacity=capacity.tolist(),
        unmet_demand_before=round(total_unmet_before, 2),
        unmet_demand_after=round(total_unmet_after, 2),
        improvement_pct=round(improvement, 2),
        stats={
            "num_transfers": len(transfers),
            "total_capacity_transferred": round(sum(t.amount for t in transfers), 4),
            "facilities_gaining": sum(1 for t in transfers if t.to_facility in facility_indices),
            "facilities_losing": len({t.from_facility for t in transfers}),
        },
    )
