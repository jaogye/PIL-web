"""
Sparse distance matrix backed by per-area neighbor lists (CSR + CSC).

The distance_matrix table stores for each census area the ~500 nearest
neighbors.  A dense n×n array wastes memory on the (n² − n×k) missing
pairs.  This module uses two compact adjacency-list representations:

  CSR (Compressed Sparse Row)  – row i → stored neighbors of area i
  CSC (Compressed Sparse Col)  – col j → areas that listed j as neighbor

Memory comparison (n = 40 649, k = 500, 7.7 M stored pairs):
  Dense uint16 : n² × 2 B         =  3.3 GB
  CSR + CSC    : 7.7 M × ~10 B   ≈   77 MB

Missing pairs are treated as MAX_DIST = 65 535 (≈ 45 days of travel).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

MAX_DIST: int = int(np.iinfo(np.uint16).max)  # 65 535


@dataclass(slots=True)
class SparseDistanceMatrix:
    """
    CSR + CSC sparse distance matrix.

    CSR (row-major): for each area i, its stored neighbors and distances.
    CSC (col-major): for each area j acting as a facility candidate,
                     the areas that stored j as one of their k nearest.

    Both directions are stored so that:
      - greedy-add (column scans) uses CSC
      - coverage checks (row scans) use CSR
    """

    n: int                  # number of areas

    # ── CSR arrays ─────────────────────────────────────────────────────
    csr_ptr: np.ndarray     # (n+1,) int32  – row i slice = csr_ptr[i]:csr_ptr[i+1]
    csr_col: np.ndarray     # (nnz,) int32  – neighbor column indices
    csr_val: np.ndarray     # (nnz,) uint16 – distances

    # ── CSC arrays ─────────────────────────────────────────────────────
    csc_ptr: np.ndarray     # (n+1,) int32
    csc_row: np.ndarray     # (nnz,) int32  – row (area) indices
    csc_val: np.ndarray     # (nnz,) uint16 – distances

    # ── Precomputed index arrays (avoid recomputing inside hot loops) ──
    _csc_j: np.ndarray      # (nnz,) int32  – facility j for each CSC entry
    _csr_i: np.ndarray      # (nnz,) int32  – area i   for each CSR entry
    _csc_val_i64: np.ndarray  # (nnz,) int64  – csc_val cast once; reused in Phase 2

    # ── Construction ───────────────────────────────────────────────────

    @staticmethod
    def from_coo(
        n: int,
        row: np.ndarray,   # (nnz,) area indices
        col: np.ndarray,   # (nnz,) neighbor indices
        val: np.ndarray,   # (nnz,) uint16 distances
    ) -> "SparseDistanceMatrix":
        """Build from coordinate (COO) arrays."""
        csr_ptr, csr_col, csr_val = _coo_to_csr(n, row, col, val)
        csc_ptr, csc_row, csc_val = _coo_to_csr(n, col, row, val)   # transpose

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

    # ── Per-column (facility) access ────────────────────────────────────

    def col_neighbors(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Areas that store j as a neighbor, with their distances."""
        s, e = int(self.csc_ptr[j]), int(self.csc_ptr[j + 1])
        return self.csc_row[s:e], self.csc_val[s:e]

    # ── min_dist operations ─────────────────────────────────────────────

    def init_min_dist(self, fac: int) -> np.ndarray:
        """
        Return a uint32 min_dist array initialised from a single facility.
        Areas not stored as neighbors of fac get MAX_DIST.
        """
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
        """uint32 array: for each area, minimum distance to any facility."""
        md = np.full(self.n, MAX_DIST, dtype=np.uint32)
        for fac in facilities:
            self.update_min_dist(md, fac)
        return md

    # ── Greedy-add helper (p-median) ────────────────────────────────────

    def cost_reductions(self, demand: np.ndarray, min_dist: np.ndarray) -> np.ndarray:
        """
        Vectorised: for every candidate facility j, compute

            reduction[j] = Σ_i demand[i] · max(0, min_dist[i] − dist(i, j))

        Only stored pairs contribute; un-stored pairs have dist = MAX_DIST
        which never beats min_dist[i], so their contribution is zero.

        Complexity: O(nnz) — fully vectorised, no Python loop over areas.
        """
        # improvement per stored pair
        impr = np.maximum(
            0,
            min_dist[self.csc_row].astype(np.int64) - self.csc_val.astype(np.int64),
        )                                           # (nnz,) int64
        weighted = demand[self.csc_row] * impr      # (nnz,) float64

        # np.bincount with weights is 3–5× faster than np.add.at for accumulation.
        return np.bincount(
            self._csc_j.astype(np.intp), weights=weighted, minlength=self.n
        )

    # ── Coverage helpers (p-center, max-coverage) ───────────────────────

    def marginal_coverage(self, uncovered_demand: np.ndarray, radius: float) -> np.ndarray:
        """
        Vectorised: for every candidate facility j, compute

            marginal[j] = Σ_{i within radius of j, not yet covered} uncovered_demand[i]

        Complexity: O(nnz) — no Python loop over areas or candidates.
        """
        within  = self.csc_val <= np.uint16(min(int(radius), MAX_DIST))  # (nnz,) bool
        contrib = uncovered_demand[self.csc_row] * within                 # (nnz,) float
        return np.bincount(
            self._csc_j.astype(np.intp), weights=contrib, minlength=self.n
        )

    def cover_counts(self, uncovered: np.ndarray, radius: float) -> np.ndarray:
        """
        For each candidate facility j, count uncovered areas within radius.

        Uses CSR so we iterate (area → neighbors) in row order.
        Complexity: O(nnz).
        """
        within = self.csr_val <= np.uint16(min(int(radius), MAX_DIST))  # (nnz,) bool
        valid  = within & uncovered[self._csr_i]
        counts = np.bincount(
            self.csr_col[valid].astype(np.intp), minlength=self.n
        )
        return counts

    def covered_by(self, fac: int, radius: float) -> np.ndarray:
        """Indices of areas within radius of fac (via CSC column j)."""
        rows, vals = self.col_neighbors(fac)
        return rows[vals <= np.uint16(min(int(radius), MAX_DIST))]

    # ── Assignment ──────────────────────────────────────────────────────

    def assign(self, facilities: list[int]) -> list[int]:
        """
        Assign each area to its nearest facility (by stored distances).
        Areas unreachable via any stored pair default to facilities[0].
        """
        md = np.full(self.n, MAX_DIST, dtype=np.uint32)
        asgn = np.full(self.n, facilities[0], dtype=np.int32)
        for fac in facilities:
            rows, vals = self.col_neighbors(fac)
            v32 = vals.astype(np.uint32)
            better = v32 < md[rows]
            asgn[rows[better]] = fac
            md[rows[better]] = v32[better]
        return asgn.tolist()

    # ── Unique radii (p-center binary search) ───────────────────────────

    def unique_radii(self) -> np.ndarray:
        """Sorted unique distance values — candidate radii for p-center."""
        return np.unique(self.csr_val)

    # ── Stats helpers ───────────────────────────────────────────────────

    def memory_bytes(self) -> int:
        """Approximate RAM used by this structure."""
        arrays = [
            self.csr_ptr, self.csr_col, self.csr_val,
            self.csc_ptr, self.csc_row, self.csc_val,
            self._csc_j, self._csr_i, self._csc_val_i64,
        ]
        return sum(a.nbytes for a in arrays)


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
