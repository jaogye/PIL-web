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
- `y_ij ∈ {0, 1}` — 1 if area i is served by facility j

**Objective (maximise):**
```
Σ_i  demand[i] × (Σ_j y_ij)
```

**Constraints:**
```
y_ij ≤ x_j               ∀ i, j              (assign only to open facilities)
d(i, j) ≤ R              ∀ (i,j): y_ij = 1  (service radius constraint)
Σ_i demand[i] y_ij ≤ cap_max  ∀ j           (facility capacity ceiling)
Σ_i demand[i] y_ij ≥ cap_min  ∀ j           (minimum operational floor)
y_ij = 1 only for j = nearest open facility  (closest assignment)
```

Note: the number of open facilities is not fixed to `p` a priori. The algorithm places as many facilities as are needed to exhaust demand, subject to the capacity constraints. The `p` parameter is accepted for API compatibility but does not constrain placement.

### Algorithm: Momentum-Based Alternating Greedy + Redundancy Pruning

#### Phase 0 – Pre-existing Facilities  `O(k × n_pre)`

Fixed facilities (from `pre_selected`) are registered first and their demand is absorbed using nearest-first assignment up to `cap_max`, reducing `remaining[i]` before Phase 1 begins.

#### Phase 1 – Momentum Computation  `O(nnz)`

For each candidate area `i`, a **momentum indicator** measures its accessibility to uncovered demand weighted by routing speed:

```
momentum[i] = Σ_j  remaining[j] · speed(i, j)
```

where `speed(i, j)` is the estimated road speed in km/h between areas `i` and `j`:

```
speed(i, j) = D_km(i, j) / (travel_time_min(i, j) / 60)
```

Momentum is high for areas surrounded by dense uncovered population served by fast roads (dense urban cores). Momentum is low for isolated peripheral areas.

**Precomputation:** CSR speeds `speed(i → j)` and CSC speeds `speed(i ← j)` are computed once from the stored pairs. Momentum uses the CSR structure (O(nnz) bincount). After each facility placement, momentum is updated incrementally in O(k) using the CSC structure — no full recomputation.

If xy coordinates are unavailable or momentum is degenerate (all-zero or NaN), the algorithm falls back to a plain demand-coverage greedy.

#### Phase 1 – Alternating MAX/MIN Placement Loop

Facilities are opened in alternating turns:

| Turn | Selection | Purpose |
|------|-----------|---------|
| **MAX** | `argmax momentum` among valid candidates | Covers the densest uncovered cluster (efficiency) |
| **MIN** | `argmin momentum` among valid candidates | Covers the most isolated peripheral area (equity) |

The turn cycle is `equity_ratio` MAX turns followed by 1 MIN turn (default `equity_ratio = 1` → strict 1-MAX / 1-MIN alternation).

**Fill rule** — nearest-first up to `cap_min`:

```
For facility j selected on any turn:
    zone ← areas within radius R, sorted by travel time ascending
    load ← 0,  taken ← []
    for each area i in zone (in order):
        if remaining[i] ≤ 0: skip
        if load + demand[i] > cap_max: skip
        load += demand[i]
        taken.append(i)
        if load ≥ cap_min: break   ← stop once minimum is met
    if load < cap_min: turn FAILS (candidate rejected)
```

A candidate is rejected if its zone does not contain enough unassigned demand to reach `cap_min`. Two consecutive failed turns terminate Phase 1.

**Valid candidate filter:** a candidate is only eligible if its marginal coverage score (total remaining demand within radius, capped at `cap_max`) is at least `cap_min`.

#### Phase 2B – Redundancy Pruning  `O(F² × k)` amortised

After Phase 1, the solution may contain spatially clustered facilities where some are dominated by their neighbours. Phase 2B iteratively removes redundant facilities:

```
while any facility was removed in last pass:
    candidates ← new facilities sorted by load ascending (lightest first)
    for each candidate fac:
        served ← areas assigned to fac
        if served is empty:
            remove fac immediately
        else:
            for each area a in served:
                find best alternative facility g ≠ fac within radius R
                    with fac_remaining[g] ≥ demand[a]
            if all areas can be feasibly reassigned:
                commit reassignment; remove fac
                break  ← restart inner loop
```

A facility is removed only when **all** its served areas can be absorbed by other open facilities without exceeding `cap_max`. The lightest-first order ensures a progressive, homogeneous spatial distribution.

**Implementation note:** when removing a facility at position `old_pos` in `selected`, the `assigned_to` index array is updated in two steps that must capture both masks *before* any modification to avoid a cascading-removal bug:

```python
mask_stranded = assigned_to == old_pos   # truly lost areas (not reassigned)
mask_shift    = assigned_to > old_pos    # need position correction after pop
assigned_to[mask_shift] -= 1
assigned_to[mask_stranded] = -1
```

Inverting this order (decrement first, then zero) incorrectly zeroes reassigned areas when the receiving facility was at `old_pos + 1`, causing it to appear empty and be pruned in cascade.

#### Defensive cap_min Filter

After Phase 2B, any facility whose final load fell below `cap_min` (possible only if Phase 2B redistribution left it with too little demand) is dropped.

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
| **Max Coverage** | Momentum precompute | O(nnz) | `np.bincount` over CSR/CSC speeds |
| **Max Coverage** | Momentum update | O(k) per placement | Incremental CSC subtraction |
| **Max Coverage** | Placement loop | O(F × nnz) | `marginal_coverage()` + fill per turn |
| **Max Coverage** | Phase 2B pruning | O(F² × k) amortised | Distance lookup per area per candidate |
| **Bump Hunter** | Gravity score | O(n × k_vec) | Sparse dot product over kNN |
| **Bump Hunter** | Local maxima | O(n log n) | KDTree spatial query |
| **Rebalancing** | Assignment | O(nnz) | `assign()` single pass |
| **Rebalancing** | Transfer loop | O(max_transfers × p) | Argmax over deficit/surplus arrays |

**Notation:**
- `n` — number of census areas
- `p` — number of facilities (API parameter; not a hard constraint in MCLP)
- `F` — number of facilities actually placed by Phase 1 (≈ total_demand / cap_min)
- `k` — stored neighbours per area (≈ 500)
- `nnz` — total stored pairs (`≈ n × k`)
- `L` — number of layers in L-Layered Search (default 3)

---

## References

- Kramer, R., Iori, M., & Vidal, T. (2018). *Mathematical models and search algorithms for the capacitated p-center problem.* INFORMS Journal on Computing, 32(2).
- Church, R., & ReVelle, C. (1974). *The maximal covering location problem.* Papers of the Regional Science Association.
- ReVelle, C., & Swain, R. (1970). *Central facilities location.* Geographical Analysis, 2(1).
- Drezner, Z., & Hamacher, H. W. (Eds.) (2002). *Facility Location: Applications and Theory.* Springer. (Momentum-based and equity-aware heuristics for capacitated coverage.)
- Downs, B. T., & Camm, J. D. (1996). *An exact algorithm for the maximal covering location problem.* Naval Research Logistics, 43(3).
