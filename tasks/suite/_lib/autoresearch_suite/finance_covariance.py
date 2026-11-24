"""Covariance estimation for the global minimum-variance portfolio (Ledoit and Wolf).

Every 21 trading days the candidate estimates the 49 x 49 covariance matrix of
daily industry returns from the previous 126 days only (N/T = 0.39). The
evaluator forms the global minimum-variance weights w = S^-1 1 / (1' S^-1 1)
from the estimate, holds them for the next 21 days and scores the annualized
standard deviation of the realized out-of-sample portfolio returns, the
criterion of Ledoit and Wolf (2017).
"""
from __future__ import annotations

import math

import numpy as np

from . import finance_data, finance_sandbox
from .runtime import CandidateError

TABLE = "ind49_vw_daily.csv"
ENTRY = "estimate_covariance"
WINDOW = 126
HOLD = 21


def load(end="20250930"):
    days, names, values = finance_data.window(TABLE, "19710312", end)
    returns = values / 100.0
    if not np.isfinite(returns).all() or (values <= -99.99).any():
        raise RuntimeError("49-industry daily returns have missing values")
    return list(days), list(names), returns


def validate(raw, n):
    try:
        sigma = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        raise CandidateError("covariance estimate must be numeric") from None
    if sigma.shape != (n, n) or not np.isfinite(sigma).all():
        raise CandidateError(f"covariance estimate must be a finite ({n}, {n}) array")
    scale = float(np.max(np.abs(sigma)))
    if scale <= 0 or np.max(np.abs(sigma - sigma.T)) > 1e-10 * scale:
        raise CandidateError("covariance estimate must be symmetric and nonzero")
    eigenvalues = np.linalg.eigvalsh(0.5 * (sigma + sigma.T))
    if eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
        raise CandidateError("covariance estimate must be positive definite (condition number below 1e12)")
    return 0.5 * (sigma + sigma.T)


def gmv_weights(sigma):
    x = np.linalg.solve(sigma, np.ones(len(sigma)))
    return x / x.sum()


def backtest_steps(returns):
    """Scoring loop: daily out-of-sample GMV returns and gross exposures; the holding-period returns stay here."""
    T, n = returns.shape
    realized, gross = [], []
    for start in range(WINDOW, T, HOLD):
        sigma = validate((yield finance_sandbox.step(ENTRY, returns[start - WINDOW:start].copy())), n)
        w = gmv_weights(sigma)
        realized.append(returns[start:start + HOLD] @ w)
        gross.append(float(np.abs(w).sum()))
    return np.concatenate(realized), np.array(gross)


def backtest(estimator, returns):
    """Daily out-of-sample GMV returns and gross exposures for a candidate module (sandboxed) or a trusted estimator."""
    return finance_sandbox.run(estimator, backtest_steps(returns), ENTRY)


def annualized_sd(daily):
    return 100.0 * float(np.std(daily, ddof=1)) * math.sqrt(252.0)


def sample_covariance(window):
    return np.cov(window, rowvar=False)


def covariance_score(candidate, end="20250930"):
    """Annualized out-of-sample standard deviation (percent) of the candidate's GMV portfolio (minimize).

    ``candidate`` is the candidate module (its estimate_covariance runs in the sandbox) or a trusted callable.
    """
    days, names, returns = load(end)
    realized, gross = backtest(candidate, returns)
    sample, sample_gross = backtest(sample_covariance, returns)
    equal = returns[WINDOW:].mean(axis=1)  # daily-rebalanced 1/N over the same days
    score = annualized_sd(realized)
    return score, {"assets": len(names), "first_day": days[WINDOW], "last_day": days[-1], "days": int(len(realized)),
                   "rebalances": int(len(gross)), "window": WINDOW, "holding_days": HOLD,
                   "mean_gross_exposure": float(gross.mean()), "max_gross_exposure": float(gross.max()),
                   "sample_covariance_sd": annualized_sd(sample), "equal_weight_sd": annualized_sd(equal),
                   "relative_to_sample_covariance": score / annualized_sd(sample)}
