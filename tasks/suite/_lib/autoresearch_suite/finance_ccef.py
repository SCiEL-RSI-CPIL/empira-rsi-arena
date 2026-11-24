"""Cardinality-constrained mean-variance frontier (Chang, Meade, Beasley and Sharaiha 2000).

OR-Library markets port1-port5: for each market the candidate supplies 51
portfolios of exactly K = 10 assets, each held asset between 1% and 100%,
fully invested, whose expected returns equal 51 fixed targets spread from the
return of the unconstrained minimum-variance portfolio to the highest return
attainable under the constraints. Each portfolio is scored with Chang et al.'s
percentage error against the published unconstrained efficient frontier (UEF):
the smaller of the percentage standard-deviation error at its return and the
percentage return error at its standard deviation, both by linear
interpolation on the UEF. The score is the mean percentage error over the 51
targets, averaged over the five markets.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import finance_data, finance_sandbox
from .finance_stats import finite_vector
from .runtime import CandidateError

ENTRY = "build_frontier"
MARKETS = {1: "Hang Seng", 2: "DAX 100", 3: "FTSE 100", 4: "S&P 100", 5: "Nikkei 225"}
K, MIN_WEIGHT, MAX_WEIGHT, POINTS = 10, 0.01, 1.0, 51
TOLERANCE = 1e-8  # absolute tolerance on the return target, the budget and the holding bounds


@lru_cache(maxsize=None)
def market(index):
    """(mean, covariance, UEF returns ascending, UEF standard deviations) of one OR-Library market."""
    _, columns, values = finance_data.table(f"orlib_port{index}.csv")
    mean, sd, corr = values[:, 0], values[:, 1], values[:, 2:]
    cov = corr * np.outer(sd, sd)
    _, _, frontier = finance_data.table(f"orlib_portef{index}.csv")
    order = np.argsort(frontier[:, 0], kind="stable")
    uef_r, uef_s = frontier[order, 0], np.sqrt(frontier[order, 1])
    for array in (mean, cov, uef_r, uef_s):
        array.setflags(write=False)
    return mean, cov, uef_r, uef_s


def max_constrained_return(mean):
    """Highest expected return with exactly K assets and the minimum holding: 1 - (K-1)*eps on the best asset."""
    top = np.sort(mean)[::-1][:K]
    return float((1.0 - (K - 1) * MIN_WEIGHT) * top[0] + MIN_WEIGHT * top[1:].sum())


def targets(index):
    mean, _, uef_r, _ = market(index)
    return np.linspace(float(uef_r[0]), max_constrained_return(mean), POINTS)


def percentage_error(s, r, uef_r, uef_s):
    """Chang et al. (2000): min of 100|s - s*(r)|/s*(r) and 100|r - r*(s)|/r*(s) where each interpolation exists."""
    errors = []
    if uef_r[0] <= r <= uef_r[-1]:
        s_star = float(np.interp(r, uef_r, uef_s))
        errors.append(100.0 * abs(s - s_star) / s_star)
    order = np.argsort(uef_s, kind="stable")
    if uef_s[order[0]] <= s <= uef_s[order[-1]]:
        r_star = float(np.interp(s, uef_s[order], uef_r[order]))
        errors.append(100.0 * abs(r - r_star) / r_star)
    if not errors:
        raise CandidateError("portfolio lies outside the range of the efficient frontier on both axes")
    return min(errors)


def validate(weights, mean, goals, name):
    w = finite_vector(weights, f"{name} portfolios", (POINTS, len(mean)))
    for k, (row, goal) in enumerate(zip(w, goals)):
        held = np.abs(row) > 1e-12
        if int(held.sum()) != K:
            raise CandidateError(f"{name} portfolio {k}: must hold exactly {K} assets, holds {int(held.sum())}")
        if (row[held] < MIN_WEIGHT - TOLERANCE).any() or (row[held] > MAX_WEIGHT + TOLERANCE).any():
            raise CandidateError(f"{name} portfolio {k}: held weights must lie in [{MIN_WEIGHT}, {MAX_WEIGHT}]")
        if abs(float(row.sum()) - 1.0) > TOLERANCE:
            raise CandidateError(f"{name} portfolio {k}: weights must sum to 1")
        if abs(float(row @ mean) - goal) > TOLERANCE:
            raise CandidateError(f"{name} portfolio {k}: expected return {row @ mean:.10f} differs from the target {goal:.10f}")
    return np.where(np.abs(w) > 1e-12, w, 0.0)


def frontiers():
    """Scoring loop: one candidate call per market; the unconstrained frontier it is scored against stays here."""
    weights = {}
    for index, name in MARKETS.items():
        mean, cov, _, _ = market(index)
        goals = targets(index)
        problem = {"name": name, "mean": mean.copy(), "cov": cov.copy(), "K": K, "min_weight": MIN_WEIGHT,
                   "max_weight": MAX_WEIGHT, "target_returns": goals.copy()}
        weights[index] = validate((yield finance_sandbox.step(ENTRY, problem)), mean, goals, name)
    return weights


def frontier_score(candidate):
    """Mean percentage error over targets, averaged over markets (minimize; 0 would lie on the UEF).

    ``candidate`` is the candidate module (its build_frontier runs in the sandbox) or a trusted callable.
    """
    per_market = {}
    weights = finance_sandbox.run(candidate, frontiers(), ENTRY)
    for index, name in MARKETS.items():
        mean, cov, uef_r, uef_s = market(index)
        w = weights[index]
        risk = np.sqrt(np.maximum(np.einsum("ki,ij,kj->k", w, cov, w), 0.0))
        errors = [percentage_error(s, r, uef_r, uef_s) for s, r in zip(risk, w @ mean)]
        per_market[name] = float(np.mean(errors))
    return float(np.mean(list(per_market.values()))), {"mean_percentage_error_by_market": per_market,
                                                       "assets": {n: int(len(market(i)[0])) for i, n in MARKETS.items()},
                                                       "targets_per_market": POINTS, "K": K}
