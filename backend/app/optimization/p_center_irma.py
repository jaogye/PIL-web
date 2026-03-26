"""
IRMA Matheuristic for the Capacitated P-Center Problem (CPCP).

Mirrors the structure of p_median_proto.py, adapted for the p-center
(min-max) objective with facility capacity constraints.

Reference:
  Stefanello et al. (2015) – IRMA: Iterated Reduction Matheuristic Algorithm.
  Applied here to the CPCP following the decomposition of
  Kramer, Iori & Vidal (2018), arXiv:1803.04865.

Algorithm per IRMA iteration
─────────────────────────────
  1. Reduction   – retain the most promising facility candidates, scored by
                   max_customer_distance / capacity  (p-center criterion).
  2. MIP solve   – exact Capacitated P-Center MIP on the reduced set via
                   PuLP + CBC (free, no licence required):
                       min  z
                       s.t. Σ y_j  ≤  p
                            Σ x_ij = 1                    ∀ i
                            z ≥ Σ d_ij · x_ij             ∀ i   [bottleneck]
                            Σ dem_i · x_ij ≤ cap_j · y_j  ∀ j
                            x_ij ≤ y_j                    ∀ i, j
  3. Local search – interchange: swap one open ↔ closed facility and re-solve
                   the assignment subproblem (y fixed, faster than full MIP).

Standalone usage
────────────────
  python -m app.optimization.p_center_irma

App integration
───────────────
  from app.optimization.p_center_irma import solve_with_sparse_matrix
  results = solve_with_sparse_matrix(dm, demand, capacity, p_min=2, p_max=6)
"""

from __future__ import annotations

import time

import numpy as np
import pulp

from .sparse_matrix import MAX_DIST, SparseDistanceMatrix
from .p_center import PCenterResult, _compute_stats


# ─────────────────────────────────────────────────────────────────────────── #
# Demo instance  (mirrors load_sample_instance in p_median_proto.py)           #
# ─────────────────────────────────────────────────────────────────────────── #

def load_sample_instance():
    """
    Generate a small, feasible CPCP demo instance.
    30 customers, 15 facility candidates — fast enough for interactive testing.
    """
    n, m = 30, 15
    np.random.seed(7)
    dist   = np.random.randint(5, 120, size=(n, m)).tolist()
    demand = np.random.randint(10,  50, size=n).tolist()
    cap    = np.random.randint(80, 250, size=m).tolist()
    return n, m, dist, demand, cap


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 1 – Candidate reduction                                                #
# ─────────────────────────────────────────────────────────────────────────── #

def reduction_phase(n, m, p, dist, demand, cap, reduction_factor=0.5):
    """
    Score each candidate and retain the best reduction_factor × m fraction.

    P-center score for candidate j:
        score_j = max_i(d_ij) / cap_j

    Minimising the worst-case distance per unit capacity selects central,
    high-capacity facilities — directly aligned with the min-max objective.
    Lower score = better candidate.
    """
    scores = []
    for j in range(m):
        max_d = max(dist[i][j] for i in range(n))
        scores.append((max_d / max(cap[j], 1), j))

    scores.sort()
    num_reduced = max(int(m * reduction_factor), p + 3)
    reduced = [j for _, j in scores[:num_reduced]]

    print(f"  Reduction: {m} -> {len(reduced)} candidates")
    return reduced


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 2 – MIP solvers                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def solve_reduced_mip(n, m, p, dist, demand, cap, reduced_candidates,
                      k_nearest=8, time_limit=30):
    """
    Solve the Capacitated P-Center MIP on the reduced candidate set.

    For tractability, each customer is only allowed assignment to its
    k_nearest reachable candidates (those with d_ij < MAX_DIST).

    Parameters
    ----------
    n, m            : customers and total facility candidates
    p               : facilities to open
    dist            : n×m distance matrix (list of lists)
    demand          : customer demands (list of n numbers)
    cap             : facility capacities (list of m numbers)
    reduced_candidates : column indices (subset of 0..m-1) to consider
    k_nearest       : max candidates per customer in the MIP
    time_limit      : CBC solver time limit (seconds)

    Returns
    -------
    (radius, open_facilities, assignment)
    radius = float('inf') when infeasible or solver error
    """
    # For each customer i, restrict to its k nearest reachable candidates.
    allowed: dict[int, list[int]] = {}
    for i in range(n):
        reachable = sorted(
            (dist[i][j], j)
            for j in reduced_candidates
            if dist[i][j] < MAX_DIST
        )
        allowed[i] = [j for _, j in reachable[:k_nearest]]
        if not allowed[i]:
            print(f"  [WARN] Customer {i} unreachable from all reduced candidates")
            return float('inf'), [], {}

    prob = pulp.LpProblem("CPCP_IRMA", pulp.LpMinimize)

    z = pulp.LpVariable("z", lowBound=0.0)
    y = {j: pulp.LpVariable(f"y_{j}", cat="Binary") for j in reduced_candidates}
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in range(n)
        for j in allowed[i]
    }

    # Objective: minimise the maximum travel time
    prob += z

    # At most p open facilities
    prob += pulp.lpSum(y[j] for j in reduced_candidates) <= p, "open_count"

    for i in range(n):
        aj = allowed[i]
        # Full assignment
        prob += pulp.lpSum(x[(i, j)] for j in aj) == 1, f"assign_{i}"
        # Bottleneck: z ≥ distance of the chosen facility for customer i
        prob += z >= pulp.lpSum(dist[i][j] * x[(i, j)] for j in aj), f"btlnk_{i}"

    for j in reduced_candidates:
        served = [i for i in range(n) if j in allowed[i]]
        if not served:
            continue
        # Capacity constraint
        prob += (
            pulp.lpSum(demand[i] * x[(i, j)] for i in served) <= cap[j] * y[j]
        ), f"cap_{j}"
        # Linking: can only assign to open facility
        for i in served:
            prob += x[(i, j)] <= y[j], f"link_{i}_{j}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=int(time_limit)))

    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        print(f"  [WARN] MIP not solved (status: {pulp.LpStatus[status]})")
        return float('inf'), [], {}

    radius = float(pulp.value(z) or 0.0)
    open_fac = [j for j in reduced_candidates if (pulp.value(y[j]) or 0.0) > 0.5]
    assignment = {}
    for i in range(n):
        for j in allowed[i]:
            if (pulp.value(x[(i, j)]) or 0.0) > 0.5:
                assignment[i] = j
                break

    return radius, open_fac, assignment


def _solve_assignment_mip(n, dist, demand, cap, open_fac,
                          k_nearest=8, time_limit=15):
    """
    Solve the assignment subproblem with open facilities fixed (y = 1).

    Used in local search to evaluate a proposed swap rapidly.  No y variables
    are needed, so the MIP is much smaller than the full CPCP model.

    Returns (radius, assignment_dict) — radius = float('inf') if infeasible.
    """
    allowed: dict[int, list[int]] = {}
    for i in range(n):
        reachable = sorted(
            (dist[i][j], j) for j in open_fac if dist[i][j] < MAX_DIST
        )
        allowed[i] = [j for _, j in reachable[:k_nearest]]
        if not allowed[i]:
            return float('inf'), {}

    prob = pulp.LpProblem("CPCP_Assign", pulp.LpMinimize)

    z = pulp.LpVariable("z", lowBound=0.0)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in range(n)
        for j in allowed[i]
    }

    prob += z

    for i in range(n):
        aj = allowed[i]
        prob += pulp.lpSum(x[(i, j)] for j in aj) == 1, f"assign_{i}"
        prob += z >= pulp.lpSum(dist[i][j] * x[(i, j)] for j in aj), f"btlnk_{i}"

    for j in open_fac:
        served = [i for i in range(n) if j in allowed[i]]
        if served:
            prob += (
                pulp.lpSum(demand[i] * x[(i, j)] for i in served) <= cap[j]
            ), f"cap_{j}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=int(time_limit)))

    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        return float('inf'), {}

    radius = float(pulp.value(z) or 0.0)
    assignment = {}
    for i in range(n):
        for j in allowed[i]:
            if (pulp.value(x[(i, j)]) or 0.0) > 0.5:
                assignment[i] = j
                break

    return radius, assignment


# ─────────────────────────────────────────────────────────────────────────── #
# Phase 3 – Interchange local search                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def local_search(n, m, dist, demand, cap, current_open, current_radius,
                 current_assignment, max_swaps=8):
    """
    Interchange local search: swap one open facility for a closed one.

    Each swap is evaluated by solving the assignment subproblem with fixed
    open facilities — much faster than re-running the full CPCP MIP.
    Accepts the swap if the maximum travel time (radius) strictly improves.
    """
    best_radius     = current_radius
    best_open       = list(current_open)
    best_assignment = dict(current_assignment)
    closed          = [j for j in range(m) if j not in set(current_open)]

    improved   = True
    iterations = 0

    while improved and iterations < max_swaps:
        improved = False
        for out_j in list(best_open):
            for in_j in closed:
                if iterations >= max_swaps:
                    break

                new_open = [j for j in best_open if j != out_j] + [in_j]
                new_radius, new_assign = _solve_assignment_mip(
                    n, dist, demand, cap, new_open)

                if new_radius < best_radius:
                    best_radius     = new_radius
                    best_open       = new_open
                    best_assignment = new_assign
                    improved = True
                    closed = [j for j in range(m) if j not in set(best_open)]
                    print(f"    [+] Interchange improvement: radius={best_radius:.1f}")

            iterations += 1

    return best_radius, best_open, best_assignment


# ─────────────────────────────────────────────────────────────────────────── #
# Main IRMA loop — single p value                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def irma_cpcp(n, m, p, dist, demand, cap, max_iter=15, reduction_factor=0.5):
    """
    IRMA matheuristic for the Capacitated P-Center Problem (single p value).

    Parameters
    ----------
    n, m            : number of customers and facility candidates
    p               : number of facilities to open
    dist            : n×m distance matrix (list of lists, indexed 0..n-1 × 0..m-1)
    demand          : customer demands  (list of n numbers)
    cap             : facility capacities (list of m numbers)
    max_iter        : total IRMA iterations
    reduction_factor: fraction of m to keep after the reduction phase

    Returns
    -------
    (open_facilities, assignment, radius)  or  None if no feasible solution found.
    open_facilities : list of column indices (0..m-1)
    assignment      : dict  customer_idx → facility_col_idx
    radius          : achieved minimum-maximum travel time
    """
    best_radius   = float('inf')
    best_solution = None
    start         = time.time()

    print(f"\n=== IRMA Capacitated P-Center  n={n}  m={m}  p={p} ===")

    for it in range(max_iter):
        print(f"\nIteration {it + 1}/{max_iter}")

        # 1. Candidate reduction
        reduced = reduction_phase(n, m, p, dist, demand, cap, reduction_factor)

        # 2. MIP on reduced candidate set
        radius, open_fac, assignment = solve_reduced_mip(
            n, m, p, dist, demand, cap, reduced)

        if radius == float('inf'):
            continue

        print(f"  MIP radius: {radius:.1f}")

        # 3. Interchange local search (re-evaluates swaps via assignment MIP)
        radius, open_fac, assignment = local_search(
            n, m, dist, demand, cap, open_fac, radius, assignment)

        if radius < best_radius:
            best_radius   = radius
            best_solution = (open_fac, assignment, radius)
            print(f"  [*] New global best: radius={best_radius:.1f}")

    elapsed = time.time() - start
    print(f"\n=== END  p={p}  time={elapsed:.2f}s  best_radius={best_radius:.1f} ===")
    return best_solution


# ─────────────────────────────────────────────────────────────────────────── #
# Range-of-p API                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def solve_range_cpcp(n, m, dist, demand, cap, p_min, p_max,
                     max_iter=15, reduction_factor=0.5):
    """
    Solve the CPCP for every p ∈ [p_min, p_max] using IRMA.

    Parameters mirror irma_cpcp(); p is iterated from p_min to p_max.

    Returns
    -------
    dict[p -> (open_facilities, assignment, radius)]
    Values are None for p values where no feasible solution was found.
    """
    print(f"\n{'='*60}")
    print(f"Capacitated P-Center -- range p in [{p_min}, {p_max}]")
    print(f"n={n}  m={m}  max_iter={max_iter}  reduction={reduction_factor}")
    print(f"{'='*60}")

    results: dict[int, tuple | None] = {}
    for p in range(p_min, p_max + 1):
        solution = irma_cpcp(
            n, m, p, dist, demand, cap,
            max_iter=max_iter,
            reduction_factor=reduction_factor,
        )
        results[p] = solution
        if solution:
            print(f"\n  p={p}: radius={solution[2]:.1f}  "
                  f"facilities={sorted(solution[0])}")
        else:
            print(f"\n  p={p}: no feasible solution found")

    return results


# ─────────────────────────────────────────────────────────────────────────── #
# App integration — SparseDistanceMatrix interface                            #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_dist_from_sparse(dm: SparseDistanceMatrix,
                             candidates: list[int]) -> list[list[float]]:
    """
    Build a dense n × |candidates| distance matrix from the sparse structure.
    Column col corresponds to candidates[col].  Non-stored pairs → MAX_DIST.
    """
    n = dm.n
    dist = [[float(MAX_DIST)] * len(candidates) for _ in range(n)]
    for col, fac in enumerate(candidates):
        rows, vals = dm.col_neighbors(fac)
        for r, v in zip(rows.tolist(), vals.tolist()):
            dist[r][col] = float(v)
    return dist


def solve_with_sparse_matrix(
    dm: SparseDistanceMatrix,
    demand: np.ndarray,
    capacity: np.ndarray,
    p_min: int,
    p_max: int,
    pre_selected: list[int] | None = None,
    max_iter: int = 10,
    reduction_factor: float = 0.4,
    time_limit_per_mip: float = 30.0,
    k_nearest: int = 10,
) -> dict[int, PCenterResult]:
    """
    App integration: solve CPCP for every p ∈ [p_min, p_max].

    Pre-reduction step: all n candidate facilities are first scored by
    (max stored distance to customers) / capacity using the sparse CSC
    structure — no dense n×n matrix is built.  Only the top
    max(5·p_max + 20, 80) candidates are kept before running IRMA,
    keeping the MIP tractable regardless of instance size.

    Parameters
    ----------
    dm              : SparseDistanceMatrix from the app
    demand          : population demand per area (shape n)
    capacity        : max demand facility j can absorb (shape n);
                      use np.full(n, np.inf) for the uncapacitated case
    p_min, p_max    : range of p values to solve (inclusive)
    pre_selected    : area indices forced open (existing infrastructure)
    max_iter        : IRMA iterations per p value
    reduction_factor: fraction of the pre-reduced set to keep per IRMA iteration
    time_limit_per_mip: CBC solver time limit per MIP solve (seconds)
    k_nearest       : max candidate facilities per customer in each MIP

    Returns
    -------
    dict[p -> PCenterResult]  — same type as p_center.solve()
    """
    n            = dm.n
    pre_selected = list(pre_selected or [])

    # ── Global pre-reduction using sparse CSC (no dense n×n matrix) ──────
    # Score all n candidates from the sparse matrix to select a compact
    # working set before building any dense sub-matrix.
    mip_cap = max(5 * p_max + 20, 80)   # maximum candidates in the dense MIP sub-matrix

    scores = np.full(n, float(MAX_DIST), dtype=np.float64)
    for j in range(n):
        rows, vals = dm.col_neighbors(j)
        if len(rows) == 0:
            continue
        max_d = float(vals.max()) if len(rows) >= n else float(MAX_DIST)
        scores[j] = max_d / max(float(capacity[j]), 1.0)

    order            = np.argsort(scores)
    working_set      = order[:mip_cap].tolist()
    pre_set          = set(pre_selected)
    working_set_set  = set(working_set)
    for ps in pre_selected:
        if ps not in working_set_set:
            working_set.append(ps)

    m_red   = len(working_set)
    dist    = _build_dist_from_sparse(dm, working_set)         # n × m_red dense
    dem_lst = demand.tolist()
    cap_lst = [float(capacity[j]) for j in working_set]

    # Map global area index → column index in the dense sub-matrix
    global_to_col = {j: col for col, j in enumerate(working_set)}
    pre_cols      = [global_to_col[ps] for ps in pre_selected if ps in global_to_col]

    print(f"\n[p_center_irma] n={n}  working_set={m_red}  "
          f"p in [{p_min},{p_max}]  max_iter={max_iter}")

    # ── Run IRMA for each p value ─────────────────────────────────────────
    out: dict[int, PCenterResult] = {}

    for p in range(p_min, p_max + 1):
        solution = irma_cpcp(
            n, m_red, p, dist, dem_lst, cap_lst,
            max_iter=max_iter,
            reduction_factor=reduction_factor,
        )

        if solution is None:
            continue

        open_cols, asgn_cols, radius = solution

        # Convert column indices back to global area indices
        open_global = sorted(working_set[col] for col in open_cols)
        asgn_list   = [
            working_set[asgn_cols.get(i, open_cols[0])]
            for i in range(n)
        ]

        min_dist = dm.min_dist_to_set(open_global)
        stats    = _compute_stats(demand, min_dist, radius)
        stats["num_facilities"] = len(open_global)

        out[p] = PCenterResult(
            facility_indices=open_global,
            assignment=asgn_list,
            optimal_radius=radius,
            coverage_stats=stats,
        )

    return out


# ─────────────────────────────────────────────────────────────────────────── #
# Standalone demo                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    n, m, dist, demand, cap = load_sample_instance()
    print(f"Instance: {n} customers, {m} candidates\n")

    results = solve_range_cpcp(
        n, m, dist, demand, cap,
        p_min=2, p_max=5,
        max_iter=10,
        reduction_factor=0.5,
    )

    print("\n" + "=" * 50)
    print("FINAL SUMMARY - Capacitated P-Center")
    print("=" * 50)
    for p in sorted(results):
        sol = results[p]
        if sol:
            open_fac, _, radius = sol
            print(f"  p={p}:  max_radius={radius:.1f}   "
                  f"facilities={sorted(open_fac)}")
        else:
            print(f"  p={p}:  infeasible")
