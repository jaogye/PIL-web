"""
Sparse distance matrix backed by per-area neighbor lists (CSR + CSC).

The distance_matrix table stores for each census area the ~500 nearest
neighbors.  A dense n×n array wastes memory on the (n² − n×k) missing
pairs.  This module uses two compact adjacency-list representations:

  CSR (Compressed Sparse Row)  – row i → stored neighbors of area i
  CSC (Compressed Sparse Col)  – col j → areas that listed j as neighbor

Missing pairs are handled by estimating travel time from WGS-84 coordinates
and per-area median speeds (avg_speed_kmh) using the formula:

    t_est(i,j) = (D_km/v_i + D_km/v_j) / 2 * 60  minutes

where D_km is the flat-earth Euclidean distance and v_i, v_j are the median
neighbourhood speeds of areas i and j.  This requires xy and speeds to be
set on the matrix after construction.

The unified distance_time(i, j) method returns the stored value when
available and the estimate otherwise.  col_neighbors_full(j, radius) extends
col_neighbors(j) with estimated-distance entries so that the full set of
areas within `radius` minutes is returned regardless of the k-NN cutoff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MAX_DIST: int = int(np.iinfo(np.uint16).max)  # 65 535

# Threshold above which marginal_coverage_full skips the estimated-pair
# contribution (too expensive for large scopes).
_FULL_COVERAGE_N_LIMIT = 10_000


@dataclass(slots=True)
class SparseDistanceMatrix:
    """
    CSR + CSC sparse distance matrix with optional full-pair estimation.

    Stored pairs use uint16 minutes (0–65 534).
    Missing pairs are estimated via distance_time(i, j) when xy and speeds
    are set.
    """

    n: int                  # number of areas

    # ── CSR arrays ─────────────────────────────────────────────────────
    csr_ptr: np.ndarray     # (n+1,) int32
    csr_col: np.ndarray     # (nnz,) int32
    csr_val: np.ndarray     # (nnz,) uint16

    # ── CSC arrays ─────────────────────────────────────────────────────
    csc_ptr: np.ndarray     # (n+1,) int32
    csc_row: np.ndarray     # (nnz,) int32
    csc_val: np.ndarray     # (nnz,) uint16

    # ── Precomputed index arrays ────────────────────────────────────────
    _csc_j: np.ndarray      # (nnz,) int32
    _csr_i: np.ndarray      # (nnz,) int32
    _csc_val_i64: np.ndarray  # (nnz,) int64

    # ── Optional geographic data (set after construction) ───────────────
    xy: np.ndarray | None = field(default=None)      # (n, 2) float64 lon/lat WGS-84
    speeds: np.ndarray | None = field(default=None)  # (n,)  float64 km/h

    # ── Construction ───────────────────────────────────────────────────

    @staticmethod
    def from_coo(
        n: int,
        row: np.ndarray,
        col: np.ndarray,
        val: np.ndarray,
    ) -> "SparseDistanceMatrix":
        """Build from coordinate (COO) arrays."""
        csr_ptr, csr_col, csr_val = _coo_to_csr(n, row, col, val)
        csc_ptr, csc_row, csc_val = _coo_to_csr(n, col, row, val)

        csc_j = np.repeat(np.arange(n, dtype=np.int32), np.diff(csc_ptr))
        csr_i = np.repeat(np.arange(n, dtype=np.int32), np.diff(csr_ptr))

        return SparseDistanceMatrix(
            n=n,
            csr_ptr=csr_ptr, csr_col=csr_col, csr_val=csr_val,
            csc_ptr=csc_ptr, csc_row=csc_row, csc_val=csc_val,
            _csc_j=csc_j,
            _csr_i=csr_i,
            _csc_val_i64=csc_val.astype(np.int64),
        )

    # ── Unified distance function ───────────────────────────────────────

    def distance_time(self, i: int, j: int) -> float:
        """
        Travel time in minutes from area i to facility j.

        Returns the stored value if (i, j) is in the distance matrix,
        otherwise estimates using the harmonic-mean speed formula.
        Falls back to MAX_DIST when coordinates or speeds are unavailable.
        """
        s, e = int(self.csr_ptr[i]), int(self.csr_ptr[i + 1])
        nbrs = self.csr_col[s:e]
        # Linear search over the (typically ≤500) stored neighbors of i.
        for k in range(e - s):
            if int(nbrs[k]) == j:
                return float(self.csr_val[s + k])

        # Not stored: estimate.
        if self.xy is None or self.speeds is None:
            return float(MAX_DIST)
        return _estimate(
            float(self.xy[i, 0]), float(self.xy[i, 1]),
            float(self.xy[j, 0]), float(self.xy[j, 1]),
            float(self.speeds[i]), float(self.speeds[j]),
        )

    # ── Per-column (facility) access ────────────────────────────────────

    def col_neighbors(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Stored areas that have j as a neighbor, with their distances."""
        s, e = int(self.csc_ptr[j]), int(self.csc_ptr[j + 1])
        return self.csc_row[s:e], self.csc_val[s:e]

    def col_neighbors_full(self, j: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """
        All areas within `radius` minutes of facility j.

        Combines stored pairs (from CSC) with estimated distances for areas
        not listed as neighbors of j.  Requires xy and speeds to be set;
        falls back to stored-only otherwise.

        Returns (row_indices int32, distances float64 minutes).
        """
        r_val = float(min(radius, MAX_DIST))
        
        # Stored neighbors within radius.
        s, e = int(self.csc_ptr[j]), int(self.csc_ptr[j + 1])
        stored_rows = self.csc_row[s:e]
        stored_vals = self.csc_val[s:e].astype(np.float64)
        in_zone = stored_vals <= r_val
        stored_in_rows = stored_rows[in_zone]
        stored_in_dists = stored_vals[in_zone]
        
        if self.xy is None or self.speeds is None:
            return stored_in_rows, stored_in_dists

        # Areas NOT in any stored neighbor list of j.
        unstored_mask = np.ones(self.n, dtype=bool)
        if len(stored_rows) > 0:
            unstored_mask[stored_rows] = False   # exclude ALL stored neighbors
        unstored_mask[j] = False                 # exclude self

        unstored = np.where(unstored_mask)[0].astype(np.int32)
        if len(unstored) == 0:
            return stored_in_rows, stored_in_dists

        # Vectorised estimated distances for unstored areas → facility j.
        xj, yj = float(self.xy[j, 0]), float(self.xy[j, 1])
        xu = self.xy[unstored, 0]
        yu = self.xy[unstored, 1]
        lat_mid = np.radians((yu + yj) / 2.0)
        dx_km = (xu - xj) * 111.0 * np.cos(lat_mid)
        dy_km = (yu - yj) * 111.0
        d_km = np.sqrt(dx_km ** 2 + dy_km ** 2)
        vj_val = float(self.speeds[j])
        vu = self.speeds[unstored]
        valid = (vj_val > 0.0) & (vu > 0.0)
        est = np.where(
            valid,
            (d_km / np.where(valid, vu, 1.0) + d_km / (vj_val if vj_val > 0 else 1.0))
            / 2.0 * 60.0,
            float(MAX_DIST),
        )

        within = est <= r_val
        if not np.any(within):
            return stored_in_rows, stored_in_dists

        new_rows = unstored[within]
        new_dists = est[within]

        all_rows = np.concatenate([stored_in_rows, new_rows]).astype(np.int32)
        all_dists = np.concatenate([stored_in_dists, new_dists])
        return all_rows, all_dists

    # ── min_dist operations ─────────────────────────────────────────────

    def init_min_dist(self, fac: int) -> np.ndarray:
        """uint32 min_dist array initialised from a single facility."""
        md = np.full(self.n, MAX_DIST, dtype=np.uint32)
        rows, vals = self.col_neighbors(fac)
        md[rows] = vals
        return md

    def update_min_dist(self, min_dist: np.ndarray, fac: int) -> None:
        """Update min_dist in-place when facility fac is added."""
        rows, vals = self.col_neighbors(fac)
        v32 = vals.astype(np.uint32)
        better = v32 < min_dist[rows]
        min_dist[rows[better]] = v32[better]

    def min_dist_to_set(self, facilities: list[int]) -> np.ndarray:
        """uint32 array: for each area, minimum stored distance to any facility."""
        md = np.full(self.n, MAX_DIST, dtype=np.uint32)
        for fac in facilities:
            self.update_min_dist(md, fac)
        return md

    def min_dist_to_set_full(self, facilities: list[int]) -> np.ndarray:
        """
        float64 array: minimum distance (stored or estimated) to any facility.
        Uses col_neighbors_full with MAX_DIST radius to include all areas.
        """
        md = np.full(self.n, float(MAX_DIST), dtype=np.float64)
        for fac in facilities:
            rows, dists = self.col_neighbors_full(fac, float(MAX_DIST))
            better = dists < md[rows]
            md[rows[better]] = dists[better]
        return md

    # ── Greedy-add helper (p-median) ────────────────────────────────────

    def cost_reductions(self, demand: np.ndarray, min_dist: np.ndarray) -> np.ndarray:
        """
        Vectorised cost-reduction scores for p-median greedy add.
        Uses stored pairs only (p-median algorithm is dense-distance agnostic).
        """
        impr = np.maximum(
            0,
            min_dist[self.csc_row].astype(np.int64) - self.csc_val.astype(np.int64),
        )
        weighted = demand[self.csc_row] * impr
        return np.bincount(
            self._csc_j.astype(np.intp), weights=weighted, minlength=self.n
        )

    # ── Coverage helpers ────────────────────────────────────────────────

    def marginal_coverage(self, uncovered_demand: np.ndarray, radius: float) -> np.ndarray:
        """
        Stored-pairs only coverage scores (O(nnz), vectorised).
        Used as a fast approximation when full-pair coverage is not needed.
        """
        within = self.csc_val <= np.uint16(min(int(radius), MAX_DIST))
        contrib = uncovered_demand[self.csc_row] * within
        return np.bincount(
            self._csc_j.astype(np.intp), weights=contrib, minlength=self.n
        )

    def marginal_coverage_full(self, uncovered_demand: np.ndarray, radius: float) -> np.ndarray:
        """
        Full coverage scores including estimated distances for missing pairs.

        For n > _FULL_COVERAGE_N_LIMIT the estimated contribution is skipped
        and the stored-only result is returned (too expensive otherwise).

        Complexity: O(nnz) stored  +  O(n²) estimated (only for n ≤ limit).
        """
        # Stored contribution (always).
        result = self.marginal_coverage(uncovered_demand, radius)

        if self.xy is None or self.speeds is None or self.n > _FULL_COVERAGE_N_LIMIT:
            return result

        r_val = float(min(radius, MAX_DIST))

        # Estimated contribution: for each area i, add demand to unstored
        # facilities j that are within radius.
        for i in range(self.n):
            dem = float(uncovered_demand[i])
            if dem <= 1e-9:
                continue

            s, e = int(self.csr_ptr[i]), int(self.csr_ptr[i + 1])
            stored_facs = self.csr_col[s:e]

            # Facilities NOT stored as neighbors of area i.
            unstored_mask = np.ones(self.n, dtype=bool)
            if len(stored_facs) > 0:
                unstored_mask[stored_facs] = False
            unstored_mask[i] = False
            unstored = np.where(unstored_mask)[0].astype(np.int32)
            if len(unstored) == 0:
                continue

            xi, yi = float(self.xy[i, 0]), float(self.xy[i, 1])
            xj = self.xy[unstored, 0]
            yj = self.xy[unstored, 1]
            lat_mid = np.radians((yi + yj) / 2.0)
            dx_km = (xj - xi) * 111.0 * np.cos(lat_mid)
            dy_km = (yj - yi) * 111.0
            d_km = np.sqrt(dx_km ** 2 + dy_km ** 2)
            vi_val = float(self.speeds[i])
            vj_arr = self.speeds[unstored]
            valid = (vi_val > 0.0) & (vj_arr > 0.0)
            est = np.where(
                valid,
                (d_km / np.where(valid, vj_arr, 1.0) + d_km / (vi_val if vi_val > 0 else 1.0))
                / 2.0 * 60.0,
                float(MAX_DIST),
            )
            within = est <= r_val
            result[unstored[within]] += dem

        return result

    def cover_counts(self, uncovered: np.ndarray, radius: float) -> np.ndarray:
        """Count uncovered areas within radius per candidate (stored pairs)."""
        within = self.csr_val <= np.uint16(min(int(radius), MAX_DIST))
        valid = within & uncovered[self._csr_i]
        return np.bincount(
            self.csr_col[valid].astype(np.intp), minlength=self.n
        )

    def covered_by(self, fac: int, radius: float) -> np.ndarray:
        """Indices of areas within radius of fac (stored + estimated)."""
        rows, _ = self.col_neighbors_full(fac, radius)
        return rows

    # ── Assignment ──────────────────────────────────────────────────────

    def assign(self, facilities: list[int]) -> list[int]:
        """
        Assign each area to its nearest facility using distance_time().
        Stored distances take priority; missing pairs use estimation.
        """
        md = np.full(self.n, float(MAX_DIST), dtype=np.float64)
        asgn = np.full(self.n, facilities[0], dtype=np.int32)
        for fac in facilities:
            rows, dists = self.col_neighbors_full(fac, float(MAX_DIST))
            better = dists < md[rows]
            asgn[rows[better]] = fac
            md[rows[better]] = dists[better]
        return asgn.tolist()

    # ── Unique radii (p-center binary search) ───────────────────────────

    def unique_radii(self) -> np.ndarray:
        """Sorted unique stored distance values — candidate radii for p-center."""
        return np.unique(self.csr_val)

    # ── Stats helpers ───────────────────────────────────────────────────

    def memory_bytes(self) -> int:
        """Approximate RAM used by this structure."""
        arrays = [
            self.csr_ptr, self.csr_col, self.csr_val,
            self.csc_ptr, self.csc_row, self.csc_val,
            self._csc_j, self._csr_i, self._csc_val_i64,
        ]
        total = sum(a.nbytes for a in arrays)
        if self.xy is not None:
            total += self.xy.nbytes
        if self.speeds is not None:
            total += self.speeds.nbytes
        return total


# ─────────────────────────────────────────────────────────────────────────── #
# Internal helpers                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def _coo_to_csr(
    n: int,
    row: np.ndarray,
    col: np.ndarray,
    val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert coordinate arrays to CSR format, sorted by row."""
    order = np.argsort(row, kind="stable")
    row_s = row[order]
    col_s = col[order].astype(np.int32)
    val_s = val[order].astype(np.uint16)

    counts = np.bincount(row_s.astype(np.intp), minlength=n).astype(np.int32)
    ptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(counts, out=ptr[1:])

    return ptr, col_s, val_s


def _estimate(xi: float, yi: float, xj: float, yj: float,
              vi: float, vj: float) -> float:
    """
    Estimate travel time in minutes between two WGS-84 points using the
    harmonic-mean speed formula:

        t = (D_km/v_i + D_km/v_j) / 2 * 60

    Returns MAX_DIST when either speed is non-positive.
    """
    if vi <= 0.0 or vj <= 0.0:
        return float(MAX_DIST)
    lat_mid = math.radians((yi + yj) / 2.0)
    dx_km = (xj - xi) * 111.0 * math.cos(lat_mid)
    dy_km = (yj - yi) * 111.0
    d_km = math.sqrt(dx_km * dx_km + dy_km * dy_km)
    return min((d_km / vi + d_km / vj) / 2.0 * 60.0, float(MAX_DIST))
