"""Out-of-sample equity-premium prediction (Welch and Goyal 2008; Campbell and Thompson 2008).

At every forecast origin t the candidate receives the monthly Welch-Goyal
predictor data observed by the end of month t (inflation one month later, as
in Welch and Goyal, because the CPI is published with a lag) and forecasts the
log equity premium of month t+1. The score is the Campbell-Thompson
out-of-sample R^2 (percent) against the prevailing historical mean.
"""
from __future__ import annotations

import numpy as np

from . import finance_data, finance_sandbox, finance_stats

TABLE = "goyal_welch_monthly.csv"
ENTRY = "forecast_premium"
FIRST_RETURN = 192601  # first month with CRSP S&P 500 returns


def load():
    keys, columns, values = finance_data.table(TABLE)
    months = np.array([int(k) for k in keys])
    data = {name: values[:, j] for j, name in enumerate(columns)}
    with np.errstate(invalid="ignore"):
        premium = np.log1p(data["CRSP_SPvw"]) - np.log1p(data["Rfree"])
    if not np.isfinite(premium[months >= FIRST_RETURN]).all():
        raise RuntimeError("equity premium has gaps after 1926-01")
    return months, data, premium


def history(months, data, premium, t):
    """Information set at the end of month index t (copies, so the candidate cannot alter the data)."""
    view = {name: column[:t + 1].copy() for name, column in data.items()}
    view["infl"][-1] = np.nan  # CPI for month t is published during month t+1
    view["yyyymm"] = months[:t + 1].copy()
    view["equity_premium"] = premium[:t + 1].copy()
    return view


def prevailing_mean_forecast(history_view):
    """The Welch-Goyal benchmark: mean log equity premium from 1926-01 to the origin."""
    premium = history_view["equity_premium"]
    return float(np.mean(premium[history_view["yyyymm"] >= FIRST_RETURN]))


def forecasts(months, data, premium, targets):
    """Scoring loop: one candidate call per origin. The realised premium never leaves this process."""
    actual, predicted, bench = [], [], []
    for i in targets:
        view = history(months, data, premium, i - 1)
        bench.append(prevailing_mean_forecast(view))
        value = yield finance_sandbox.step(ENTRY, view)
        predicted.append(finance_stats.finite_scalar(value, f"forecast for {months[i]}", bound=0.5))
        actual.append(premium[i])
    return np.array(actual), np.array(predicted), np.array(bench)


def premium_score(candidate, *, start=196501, end=202512):
    """OOS R^2 (percent) of next-month log equity-premium forecasts over [start, end] (maximize).

    ``candidate`` is the candidate module (its forecast_premium runs in the sandbox) or a trusted callable.
    """
    months, data, premium = load()
    targets = [i for i in range(1, len(months)) if start <= months[i] <= end]
    if not targets or months[-1] < end:
        raise RuntimeError("forecast period exceeds the data")
    actual, predicted, bench = finance_sandbox.run(candidate, forecasts(months, data, premium, targets), ENTRY)
    score = finance_stats.oos_r2(actual, predicted, bench)
    cw, cw_p = finance_stats.clark_west(actual, predicted, bench)
    target_months = months[targets]
    split = target_months < 200001
    details = {
        "first_target_month": int(target_months[0]), "last_target_month": int(target_months[-1]), "forecasts": len(actual),
        "oos_r2_percent": score, "msfe": float(np.mean((actual - predicted) ** 2)), "msfe_prevailing_mean": float(np.mean((actual - bench) ** 2)),
        "clark_west_statistic": cw, "clark_west_one_sided_p": cw_p,
        "oos_r2_percent_before_2000": finance_stats.oos_r2(actual[split], predicted[split], bench[split]) if split.any() else None,
        "oos_r2_percent_from_2000": finance_stats.oos_r2(actual[~split], predicted[~split], bench[~split]) if (~split).any() else None,
        "share_negative_forecasts": float(np.mean(predicted < 0)), "mean_forecast": float(np.mean(predicted)),
    }
    return score, details
