# Localization Models and Heuristics – Technical Reference

This document describes the mathematical models and heuristic algorithms used in PIL to solve facility location problems. All algorithms operate on a **sparse travel-time matrix** and a **demand vector** (population per census area). No external LP/MIP solver is required.

---

## Table of Contents

1. [Core Data Structure – Sparse Distance Matrix](#1-core-data-structure--sparse-distance-matrix)
2. [P-Median](#2-p-median)
3. [P-Center](#3-p-center)
4. [Maximum Coverage (MCLP)](#4-maximum-coverage-mclp)
5. [Bump Hunter](#5-bump-hunter)
6. [Capacity Rebalancing](#6-capacity-rebalancing)
7. [Complexity Summary](#7-complexity-summary)

---

## 1. Core Data Structure – Sparse Distance Matrix

**File:** `backend/app/optimization/sparse_matrix.py`

### Motivation

For `n = 20,000` census areas, a dense `n × n` travel-time matrix requires ~400 million pairs. Storing only the `k = 500` nearest neighbours per area reduces this to `~10 M` pairs while covering all practically reachable facility–demand assignments.

### Dual CSR + CSC Representation

The matrix is stored in two complementary sparse formats simultaneously:

**Compressed Sparse Row (CSR)** – optimised for row (source-area) queries:

```
csr_ptr[i] : csr_ptr[i+1]  →  range of neighbour entries for area i
csr_col[ptr[i] : ptr[i+1]] →  indices of neighbours
csr_val[ptr[i] : ptr[i+1]] →  travel times (uint16, minutes × scale)
```

**Compressed Sparse Column (CSC)** – optimised for column (facility-candidate) queries:

```
csc_ptr[j] : csc_ptr[j+1]  →  range of source areas listing j as neighbour
csc_row[ptr[j] : ptr[j+1]] →  source area indices
csc_val[ptr[j] : ptr[j+1]] →  travel times
```

Both formats are built once from the database at optimization start and reused across all algorithms.

### Missing-Pair Estimation

For area pairs `(i, j)` absent from the sparse storage, travel time is estimated using a harmonic-mean speed model:

```
v_harm(i,j) = 2 / (1/v_i + 1/v_j)
t_est(i,j)  = D_km(i,j) / v_harm(i,j) × 60  [minutes]
```

where:
- `D_km(i,j)` is the Haversine distance using approximate lat/lon scaling (`111 km/°` latitude, `111 × cos(lat) km/°` longitude)
- `v_i`, `v_j` are the median routing speeds (km/h) of each area's pre-computed neighbourhood, stored in `avg_speed_kmh`

This estimation fills the "long tail" of distant pairs without a routing query.

### Key Methods

| Method | Complexity | Description |
|--------|-----------|-------------|
| `distance_time(i, j)` | O(k) | Stored or estimated travel time |
| `col_neighbors(j)` | O(k) | Areas listing j as stored neighbour (CSC) |
| `col_neighbors_full(j, r)` | O(n) | All areas within radius r (stored + estimated) |
| `cost_reductions(demand, min_dist)` | O(nnz) | Greedy-add marginal gain for all candidates |
| `marginal_coverage(uncov, r)` | O(nnz) | Uncovered demand gained per candidate |
| `assign(facilities)` | O(nnz) | Assign each area to nearest facility |
| `min_dist_to_set(facilities)` | O(nnz) | Minimum distance per area to facility set |
| `unique_radii()` | O(nnz log nnz) | Sorted unique stored distances |

---

## 2. P-Median

**File:** `backend/app/optimization/p_median.py`

### Mathematical Formulation

**Variables:**
- `x_j ∈ {0, 1}` — 1 if facility j is opened
- `y_ij ∈ {0, 1}` — 1 if area i is served by facility j

**Objective (minimise):**
```
Σ_i Σ_j  demand[i] × d(i,j) × y_ij
```

**Constraints:**
```
Σ_j x_j = p                   (exactly p facilities)
Σ_j y_ij = 1   ∀ i            (each area served by exactly one facility)
y_ij ≤ x_j     ∀ i, j         (can only assign to open facilities)
```

### Algorithm: Greedy-Add + Vectorised 1-opt Exchange

#### Phase 1 – Greedy Add  `O(p × nnz)`

```
S ← { area with highest demand }
while |S| < p:
    scores ← cost_reductions(demand, min_dist_to_S)   # O(nnz), vectorised
    add argmax(scores) to S
    update min_dist_to_S
```

`cost_reductions()` uses `np.bincount()` over the CSC array to score all n candidates in a single pass — no Python inner loop over candidates.

#### Phase 2 – Vectorised 1-opt Exchange  (skipped for n > 10,000)

**Pre-computation (once per outer iteration):**
- `nearest_at_csc[k]`  — current nearest facility for the source area of CSC entry k
- `md_at_csc_base[k]`  — current minimum distance for that source area
- `demand_at_csc[k]`   — demand of that source area

These arrays are computed once per iteration to avoid ~900 redundant per-candidate allocations.

**Exchange loop:**
```
for each facility f_l ∈ S:
    # Cost without f_l: O(n) using precomputed top-2 nearest
    cost_without ← cost_if_removed(f_l)

    # Best replacement: O(nnz) using cost_reductions on residual assignment
    f_new ← argmax cost_reductions(demand, min_dist_without_fl)

    if cost_without - gain(f_new) < current_cost:
        S ← (S \ {f_l}) ∪ {f_new}
        current_cost ← improved cost
        restart
```

First-improving swap is accepted immediately. The loop restarts until no improvement is found.

**Overall complexity:** `O(p × (n + nnz))` per outer iteration.

---

## 3. P-Center

**File:** `backend/app/optimization/p_center.py`

### Mathematical Formulation

**Objective (minimise):**
```
max_i  min_j∈S  d(i, j)
```

**Subject to:**
```
|S| = p
```

Equivalently, find the smallest radius `r*` such that `p` facilities can cover all areas within `r*` (minimum dominating set of a disk graph).

### Algorithm: L-Layered Search + Greedy Dominating Set Oracle

Based on: *Kramer, Iori, Vidal – "A Matheuristic for the P-Center Problem" (2018)*.

#### Phase 0 – Candidate Radii

The optimal radius `r*` must be one of the `O(nnz)` unique stored distances (decomposition principle). These are extracted and sorted once:

```
R_candidates ← sorted unique values of csr_val
```

#### Phase 1 – Upper Bound via Furthest Insertion  `O(p × nnz)`

A 2-approximation of the p-center objective:

```
S ← { area with highest demand }
while |S| < p:
    add argmax min_dist_to_S to S   # furthest uncovered area
```

This narrows the search interval `[i_low, i_up]` before the layered search.

#### Phase 2 – L-Layered Search

Replaces binary search `O(log N)` with a recursive partitioning that reduces oracle calls from `O(log N)` to `O(L)` at the cost of `O(L × N^(1/L))` total subproblem evaluations (cheap size estimates).

```
layered_search(i_low, i_up, L):
    if i_up - i_low ≤ 1:  return i_up
    step ← ceil((i_up - i_low)^(1/L))
    sample indices i_low, i_low+step, i_low+2*step, ..., i_up
    find first feasible index via oracle
    recurse with L-1 layers on the narrowed interval
```

**Default:** `L = 3` → at most 3 feasibility solves, `O(N^(1/3))` total subproblems.

#### Phase 3 – Feasibility Oracle: Greedy Dominating Set  `O(n log n)`

Given a radius `r`, check whether `p` facilities suffice to cover all areas:

```
Pre-check: if any area has no stored neighbour within r → infeasible (O(n))

Greedy set cover:
    uncovered ← all areas
    while uncovered ≠ ∅ and |S| < p:
        j ← argmax |{i ∈ uncovered : d(i,j) ≤ r}|
        S ← S ∪ {j}
        uncovered ← uncovered \ {i : d(i,j) ≤ r}
    return feasible iff uncovered = ∅
```

#### Phase 4 – Fill to p Facilities

If the oracle returns fewer than `p` facilities (feasible with fewer), furthest insertion fills the remaining slots. The actual achieved radius is then recomputed exactly.

---

## 4. Maximum Coverage (MCLP)

**File:** `backend/app/optimization/max_coverage.py`

### Mathematical Formulation

**Capacitated Maximum Coverage Location Problem with Closest Assignment Constraints (CMCLP-CAC)**

**Variables:**
- `x_j ∈ {0, 1}` — 1 if facility j is opened
- `y_ij ∈ {0, 1}` — 1 if area i is covered by facility j

**Objective (maximise):**
```
Σ_i  demand[i] × (Σ_j y_ij)
```

**Constraints:**
```
Σ_j x_j = p                                  (exactly p open facilities)
y_ij ≤ x_j               ∀ i, j              (assign only to open facilities)
d(i, j) ≤ R              ∀ (i,j): y_ij = 1  (service radius constraint)
Σ_i demand[i] y_ij ≤ cap_max  ∀ j           (facility capacity ceiling)
Σ_i demand[i] y_ij ≥ cap_min  ∀ j           (minimum operational floor)
y_ij = 1 only for j = nearest open facility  (closest assignment)
```

### Algorithm: GRASP with Proportional-Exponential Construction

GRASP (Greedy Randomized Adaptive Search Procedure) runs `N_GRASP_ITER = 5` independent trials; the best result is kept.

#### Phase 0 – Pre-existing Facilities

Fixed facilities (from mode `complete_existing` or explicit `fixed_census_area_ids`) are selected first. Their covered demand is excluded from the construction budget.

#### Phase 1 – Zone Precomputation

For each candidate facility `j`, compute all areas within radius `R` once:

```
zone[j] ← sorted list of (area, distance) with distance ≤ R
          (stored neighbours + estimated pairs)
```

This caches `col_neighbors_full(j, R)` to avoid repeated O(n) calls in the local search hot loop.

#### Phase 2 – GRASP Construction (per trial)

**Proportional-Exponential Greedy (`pgreedy_exp`):**

```
while |S| < p:
    score[j] ← uncovered_demand_within_R(j)   # O(nnz) via marginal_coverage()
    rank[j]  ← argsort(-score)
    prob[j]  ∝ exp(−α × rank[j])   with α = 0.05
    j_new    ← random choice weighted by prob
    S ← S ∪ {j_new}
    update uncovered set
```

The exponential bias (`α = 0.05`) strongly favours top-ranked candidates but allows controlled exploration, avoiding the pure greedy local optima.

#### Phase 3 – First-Improvement Local Search (per trial)

```
repeat:
    for each open non-fixed facility f_l ∈ S:
        # Build neighbourhood: sample 20 areas from zone[f_l]
        # Collect all CSR neighbours of those areas as candidates
        # Score candidates by uncovered marginal gain
        # Evaluate top 25 via CAC re-assignment (see Phase 4)
        if best_candidate improves coverage:
            S ← (S \ {f_l}) ∪ {best_candidate}
            break  # first-improvement: restart
until no improvement or max_rounds = 20 reached
```

#### Phase 4 – Closest Assignment Constraint (CAC) Re-assignment

After any change to `S`:

```
for each area i (sorted by distance to nearest open facility):
    j_nearest ← nearest open facility within R
    if j_nearest exists and load[j_nearest] + demand[i] ≤ cap_max:
        assign i → j_nearest
        load[j_nearest] += demand[i]
    else:
        i is uncovered
```

This enforces both the radius constraint and capacity ceiling simultaneously.

#### Phase 5 – Drop Facilities Below Minimum

```
for each facility f ∈ S with load[f] < cap_min:
    S ← S \ {f}
```

Facilities that attract less demand than the operational floor are removed.

---

## 5. Bump Hunter

**File:** `backend/app/optimization/bump_hunter.py`

### Purpose

Bump Hunter is an **exploratory** model that does not place `p` facilities. Instead it detects census areas that are local maxima of a gravity-weighted demand density — high-demand clusters that represent natural facility placement candidates.

### Algorithm

#### Step 1 – Gravity Score

For each area `i`, compute:

```
s[i] = demand[i]  +  Σ_{j ≠ i, j ∈ kNN(i)}  demand[j] / (1 + d(i, j))
```

where `kNN(i)` are the `k_vec` (default 500) nearest stored neighbours of `i`. For small instances (`n ≤ 10,000`) with XY coordinates and speed data available, estimated pairs are also included.

The gravity term gives high weight to nearby high-demand areas and low weight to distant ones (inverse-distance decay).

#### Step 2 – Local Maxima Detection

Area `i` is a "bump" if:

```
s[i] ≥ s[j]  ∀ j ∈ spatial_kNN(i, k_spatial)
```

where `spatial_kNN` uses:
- **KDTree** (`scipy.spatial.KDTree`) for fast spatial `k`-nearest-neighbour lookup when `n ≤ 2,000`
- **CSR stored neighbours** as a fallback for larger instances

#### Output

Areas are returned sorted by `s[i]` descending. The top results are shown on the map as suggested facility locations without committing to a specific count.

---

## 6. Capacity Rebalancing

**File:** `backend/app/optimization/rebalancing.py`

### Purpose

Given a fixed set of open facilities with known capacities, redistribute capacity across the network to reduce unmet demand — without relocating or building new facilities.

### Algorithm: Greedy Transfer Heuristic

```
# Initial assignment
for each area i:
    assign i → nearest open facility

# Classify facilities
surplus[f] = max(0, capacity[f] - load[f])
deficit[f] = max(0, load[f]    - capacity[f])

# Greedy transfer loop (max_transfers iterations)
while any deficit > 0:
    worst  ← argmax deficit
    donor  ← argmax (surplus[f] - min_capacity_floor)

    if no valid donor: break

    transfer_amount ← min(surplus[donor], deficit[worst])
    capacity[donor] -= transfer_amount
    capacity[worst] += transfer_amount

    record transfer: (donor, worst, transfer_amount, unmet_impact)

    update surplus / deficit arrays
```

Each iteration records a `(from_facility, to_facility, amount, impact)` tuple. The `impact` is the reduction in unmet demand achieved by the transfer, shown in the UI as orange arrows on the map.

---

## 7. Complexity Summary

| Model | Phase | Complexity | Bottleneck |
|-------|-------|-----------|-----------|
| **P-Median** | Greedy Add | O(p × nnz) | `cost_reductions()` via `np.bincount` |
| **P-Median** | 1-opt Exchange | O(p × (n + nnz)) per outer iter | Precomputed top-2 + `cost_reductions` |
| **P-Center** | Candidate radii | O(nnz log nnz) | Sort unique stored distances |
| **P-Center** | Furthest Insertion | O(p × nnz) | `min_dist_to_set` per step |
| **P-Center** | L-Layered Search | O(L × N^(1/L)) oracle calls | Feasibility oracle × L calls |
| **P-Center** | Greedy Dom. Set | O(n log n) per oracle call | Priority queue over uncovered |
| **Max Coverage** | Zone precompute | O(p × nnz) once | `col_neighbors_full` per candidate |
| **Max Coverage** | GRASP construction | O(N_ITER × p × nnz) | `marginal_coverage()` per step |
| **Max Coverage** | Local search | O(rounds × p × 25 × n) | CAC re-assignment per candidate eval |
| **Bump Hunter** | Gravity score | O(n × k_vec) | Sparse dot product over kNN |
| **Bump Hunter** | Local maxima | O(n log n) | KDTree spatial query |
| **Rebalancing** | Assignment | O(nnz) | `assign()` single pass |
| **Rebalancing** | Transfer loop | O(max_transfers × p) | Argmax over deficit/surplus arrays |

**Notation:**
- `n` — number of census areas
- `p` — number of facilities to place
- `nnz` — number of stored pairs in the sparse matrix (`≈ n × k`, k ≈ 500)
- `N_ITER` — number of GRASP iterations (default 5)
- `L` — number of layers in L-Layered Search (default 3)

---

## References

- Kramer, R., Iori, M., & Vidal, T. (2018). *Mathematical models and search algorithms for the capacitated p-center problem.* INFORMS Journal on Computing, 32(2).
- Church, R., & ReVelle, C. (1974). *The maximal covering location problem.* Papers of the Regional Science Association.
- ReVelle, C., & Swain, R. (1970). *Central facilities location.* Geographical Analysis, 2(1).
- Feo, T. A., & Resende, M. G. C. (1995). *Greedy randomized adaptive search procedures.* Journal of Global Optimization, 6(2).
