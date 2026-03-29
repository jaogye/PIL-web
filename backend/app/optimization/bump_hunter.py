"""
Bump Hunter: finds census areas that are local maxima of a gravity demand-density score.

A census area is a "bump" when all k of its nearest spatial neighbours have a
gravity score ≤ its own.  The gravity score is:

    s[i] = demand[i] + Σ_{j≠i}  demand[j] / (1 + dist_minutes(j → i))

using up to k_vec nearest stored CSR pairs per source area, plus estimated
distances for unstored pairs when xy/speeds are available, n ≤ _BH_N_LIMIT,
and the area has fewer than k_vec stored neighbours.

Parameters
----------
k_neighbors : int, optional
    Neighbourhood size for local-maxima detection.
    Default: min(max(1, int(0.05 * n)), 100).
k_vec : int, optional
    Number of nearest neighbours used per source area when computing the
    gravity score.  Default: 500.  Lower values make the score more local.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree

from .sparse_matrix import SparseDistanceMatrix, MAX_DIST

# Above this threshold, skip the O(n²) estimated-pairs contribution to gravity scores.
_BH_N_LIMIT = 10_000

# Above this threshold, fall back to stored-pairs KNN instead of full spatial KNN.
_KNN_FULL_N_LIMIT = 2_000


@dataclass
class BumpHunterResult:
    bump_indices: list[int]   # area indices, sorted by gravity score descending
    scores: list[float]       # corresponding gravity scores
    k_neighbors: int
    k_vec: int
    stats: dict = field(default_factory=dict)


def solve(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    k_neighbors: int | None = None,
    k_vec: int = 500,
) -> BumpHunterResult:
    """Find census areas that are local maxima of the gravity demand-density score."""
    n = dm.n
    if k_neighbors is None:
        k_neighbors = min(max(1, int(0.05 * n)), 100)
    k_neighbors = min(k_neighbors, n - 1)
    k_vec = max(1, k_vec)

    scores = _gravity_scores(dm, demand, k_vec)
    knn = _k_nearest_neighbors(dm, k_neighbors)

    bumps: list[int] = []
    for i in range(n):
        if demand[i] <= 1e-9:
            continue  # areas with no demand cannot be bumps
        nbrs = knn[i]
        if not nbrs or scores[i] >= max(scores[j] for j in nbrs):
            bumps.append(i)

    # Sort descending by gravity score.
    bumps.sort(key=lambda i: -scores[i])

    return BumpHunterResult(
        bump_indices=bumps,
        scores=[float(scores[i]) for i in bumps],
        k_neighbors=k_neighbors,
        k_vec=k_vec,
        stats={
            "num_bumps": len(bumps),
            "k_neighbors": k_neighbors,
            "k_vec": k_vec,
            "total_areas": n,
            "total_demand": float(np.sum(demand)),
        },
    )


def _gravity_scores(dm: SparseDistanceMatrix, demand: np.ndarray, k_vec: int) -> np.ndarray:
    """
    Compute gravity scores for each area as a potential facility site:
        s[i] = demand[i] + Σ_j demand[j] / (1 + dist(j → i))

    Uses up to k_vec nearest stored CSR neighbours per source area.
    When k_vec ≥ all stored neighbours the fast vectorised CSC bincount path is used.
    For small scopes (n ≤ _BH_N_LIMIT) with xy/speeds available, estimated distances
    fill in for areas that have fewer than k_vec stored neighbours.
    """
    n = dm.n
    scores = demand.copy().astype(np.float64)

    # Determine whether to use the fast vectorised path (k_vec covers all stored pairs).
    max_stored = int(np.max(np.diff(dm.csr_ptr))) if n > 0 else 0

    if k_vec >= max_stored:
        # Fast path: use every stored pair via CSC bincount (O(nnz), vectorised).
        if len(dm._csc_j) > 0:
            weights = demand[dm.csc_row] / (1.0 + dm.csc_val.astype(np.float64))
            scores += np.bincount(dm._csc_j.astype(np.intp), weights=weights, minlength=n)
    else:
        # Iterative path: for each source area i, use only the k_vec nearest stored
        # neighbours (by travel time) to contribute to destination scores.
        for i in range(n):
            dem_i = float(demand[i])
            if dem_i <= 1e-9:
                continue
            s, e = int(dm.csr_ptr[i]), int(dm.csr_ptr[i + 1])
            nbrs = dm.csr_col[s:e]
            dists = dm.csr_val[s:e].astype(np.float64)
            if len(nbrs) == 0:
                continue
            if len(nbrs) > k_vec:
                top = np.argpartition(dists, k_vec)[:k_vec]
                nbrs = nbrs[top]
                dists = dists[top]
            scores[nbrs] += dem_i / (1.0 + dists)

    # Estimated pairs: only for small scopes with coordinates available.
    # For each area i with fewer than k_vec stored neighbours, estimate distances to
    # the remaining (k_vec - n_stored) nearest unstored destinations.
    if dm.xy is None or dm.speeds is None or n > _BH_N_LIMIT:
        return scores

    for i in range(n):
        dem_i = float(demand[i])
        if dem_i <= 1e-9:
            continue
        s, e = int(dm.csr_ptr[i]), int(dm.csr_ptr[i + 1])
        n_stored = e - s
        if n_stored >= k_vec:
            continue  # already using k_vec stored neighbours — skip estimation

        stored_facs = dm.csr_col[s:e]
        unstored_mask = np.ones(n, dtype=bool)
        if len(stored_facs) > 0:
            unstored_mask[stored_facs] = False
        unstored_mask[i] = False
        unstored = np.where(unstored_mask)[0].astype(np.int32)
        if len(unstored) == 0:
            continue

        xi, yi = float(dm.xy[i, 0]), float(dm.xy[i, 1])
        xj = dm.xy[unstored, 0]
        yj = dm.xy[unstored, 1]
        lat_mid = np.radians((yi + yj) / 2.0)
        dx_km = (xj - xi) * 111.0 * np.cos(lat_mid)
        dy_km = (yj - yi) * 111.0
        d_km = np.sqrt(dx_km ** 2 + dy_km ** 2)
        vi = float(dm.speeds[i])
        vj = dm.speeds[unstored]
        valid = (vi > 0.0) & (vj > 0.0)
        est_d = np.where(
            valid,
            (d_km / np.where(valid, vj, 1.0) + d_km / (vi if vi > 0.0 else 1.0)) / 2.0 * 60.0,
            float(MAX_DIST),
        )

        # Take only the (k_vec - n_stored) nearest estimated destinations.
        n_need = k_vec - n_stored
        if len(unstored) > n_need:
            top_est = np.argpartition(est_d, n_need)[:n_need]
            unstored = unstored[top_est]
            est_d = est_d[top_est]

        scores[unstored] += dem_i / (1.0 + est_d)

    return scores


def _k_nearest_neighbors(dm: SparseDistanceMatrix, k: int) -> list[list[int]]:
    """
    Return the k nearest spatial neighbours for each area.

    Uses a KDTree on WGS-84 coordinates (scaled to km for approximate isotropy)
    when xy is available and n ≤ _KNN_FULL_N_LIMIT.  Falls back to k nearest
    stored CSR pairs for larger scopes.
    """
    n = dm.n

    if dm.xy is not None and n <= _KNN_FULL_N_LIMIT:
        lat_mid = np.radians(dm.xy[:, 1].mean())
        scaled = np.column_stack([
            dm.xy[:, 0] * np.cos(lat_mid) * 111.0,  # approx km east-west
            dm.xy[:, 1] * 111.0,                     # approx km north-south
        ])
        tree = KDTree(scaled)
        _, idx = tree.query(scaled, k=min(k + 1, n))  # +1 because self is included
        return [[int(j) for j in row if j != i][:k] for i, row in enumerate(idx)]

    # Fallback: k nearest stored CSR neighbours (by travel time).
    result: list[list[int]] = []
    for i in range(n):
        s, e = int(dm.csr_ptr[i]), int(dm.csr_ptr[i + 1])
        nbrs = dm.csr_col[s:e]
        dists = dm.csr_val[s:e]
        if len(nbrs) == 0:
            result.append([])
            continue
        order = np.argsort(dists)[:k]
        result.append(nbrs[order].tolist())
    return result
