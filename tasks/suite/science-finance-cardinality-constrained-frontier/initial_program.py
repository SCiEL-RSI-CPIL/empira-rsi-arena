"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import cvxpy as cp
import numpy as np


def _restricted(problem, support, target):
    mu, cov = problem["mean"][support], problem["cov"][np.ix_(support, support)]
    w = cp.Variable(len(support))
    constraints = [cp.sum(w) == 1, mu @ w == target, w >= problem["min_weight"], w <= problem["max_weight"]]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), constraints)
    prob.solve(solver="CLARABEL")
    return None if prob.status != "optimal" else w.value


def build_frontier(problem):
    """Seed: keep the K largest positions of the unconstrained frontier portfolio at each target and re-optimize them
    with the holding bounds; fall back to the K assets with the highest mean."""
    mu, cov, K = problem["mean"], problem["cov"], problem["K"]
    n = len(mu)
    w = cp.Variable(n)
    target = cp.Parameter()
    unconstrained = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), [cp.sum(w) == 1, w >= 0, mu @ w == target])
    rows = []
    for goal in problem["target_returns"]:
        target.value = goal
        unconstrained.solve(solver="CLARABEL")
        for support in (np.sort(np.argsort(-w.value)[:K]), np.sort(np.argsort(-mu)[:K])):
            x = _restricted(problem, support, goal)
            if x is not None:
                row = np.zeros(n)
                row[support] = x
                rows.append(row)
                break
        else:
            raise RuntimeError("no feasible support for target %r" % goal)
    return np.array(rows)
# EVOLVE-BLOCK-END
