"""Daily volatility forecasting for US equity portfolios (Hansen and Lunde 2005; Patton 2011).

The candidate builds a forecaster from daily returns of the US market and 49
industry portfolios up to 1999-12-31. The evaluator then walks through every
trading day from 2000-01-03 to 2025-09-30: it asks for the next day's variance
of all 50 series and only afterwards reveals that day's returns. Forecasts are
scored with the QLIKE loss log(h) + r^2/h against squared returns, a loss that
ranks forecasts consistently even with this noisy variance proxy (Patton 2011).
"""
from __future__ import annotations

import numpy as np

from . import finance_data
from .finance_stats import diebold_mariano, finite_vector
from .runtime import CandidateError

START, TRAIN_END, END = "19710312", "19991231", "20250930"


def load():
    days, names, industries = finance_data.window("ind49_vw_daily.csv", START, END)
    f_days, f_names, factors = finance_data.window("ff_factors_daily.csv", START, END)
    if days != f_days:
        raise RuntimeError("industry and factor days are not aligned")
    market = (factors[:, f_names.index("Mkt-RF")] + factors[:, f_names.index("RF")]) / 100.0
    returns = np.column_stack([market, industries / 100.0])
    if not np.isfinite(returns).all():
        raise RuntimeError("daily returns have missing values")
    return list(days), ["Market"] + list(names), returns


def qlike(variance, returns):
    return np.log(variance) + returns ** 2 / variance


class _RiskMetrics:
    """EWMA benchmark with lambda = 0.94 (J.P. Morgan RiskMetrics)."""

    def __init__(self, training, names=None):
        self.h = np.mean(training[-250:] ** 2, axis=0)

    def forecast(self):
        return self.h

    def update(self, r):
        self.h = 0.94 * self.h + 0.06 * r ** 2


def run(make_forecaster, days, names, returns, split):
    model = make_forecaster(returns[:split].copy(), list(names))
    if not (callable(getattr(model, "forecast", None)) and callable(getattr(model, "update", None))):
        raise CandidateError("make_forecaster must return an object with forecast() and update(returns)")
    n = returns.shape[1]
    forecasts = np.empty((len(days) - split, n))
    for k, t in enumerate(range(split, len(days))):
        forecasts[k] = finite_vector(model.forecast(), f"variance forecast for {days[t]}", (n,))
        if (forecasts[k] < 1e-10).any() or (forecasts[k] > 1.0).any():
            raise CandidateError(f"variance forecasts for {days[t]} must lie in [1e-10, 1] (daily, decimal returns squared)")
        model.update(returns[t].copy())
    return forecasts


def volatility_score(make_forecaster):
    """Mean QLIKE over all evaluation days and the 50 series (minimize)."""
    if not callable(make_forecaster):
        raise CandidateError("make_forecaster must be callable")
    days, names, returns = load()
    split = next(i for i, d in enumerate(days) if d > TRAIN_END)
    actual = returns[split:]
    loss = qlike(run(make_forecaster, days, names, returns, split), actual)
    bench = qlike(run(_RiskMetrics, days, names, returns, split), actual)
    dm, p = diebold_mariano(loss.mean(axis=1), bench.mean(axis=1), lags=5)
    return float(loss.mean()), {
        "series": len(names), "training_days": split, "evaluation_days": int(len(actual)),
        "first_day": days[split], "last_day": days[-1], "market_qlike": float(loss[:, 0].mean()),
        "riskmetrics_qlike": float(bench.mean()), "qlike_minus_riskmetrics": float(loss.mean() - bench.mean()),
        "diebold_mariano_vs_riskmetrics": dm, "one_sided_p_better_than_riskmetrics": p,
        "series_beating_riskmetrics": int((loss.mean(axis=0) < bench.mean(axis=0)).sum()),
    }
