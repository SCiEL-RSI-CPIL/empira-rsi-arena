"""Forecasting the Treasury yield curve (Diebold and Li 2006) on the Gurkaynak-Sack-Wright curves.

At every month-end origin from 1990-01 to 2024-07 the candidate sees the
end-of-month zero-coupon yields (1 to 10 years) observed so far and forecasts
the curve 1, 6 and 12 months ahead. The score is the root-mean-square forecast
error relative to the random walk (no-change forecast), averaged over the three
horizons and six maturities (1, 2, 3, 5, 7 and 10 years); below 1 beats it.
"""
from __future__ import annotations

import numpy as np

from . import finance_data, finance_sandbox
from .finance_stats import diebold_mariano, finite_vector

TABLE = "gsw_yields_monthly.csv"
ENTRY = "forecast_yields"
HORIZONS = (1, 6, 12)
SCORED = (1, 2, 3, 5, 7, 10)  # maturities in years that enter the score
FIRST_ORIGIN, LAST_ORIGIN = 199001, 202407


def load():
    keys, columns, values = finance_data.table(TABLE)
    months = np.array([int(k) for k in keys])
    yields = values[:, [columns.index(f"SVENY{m:02d}") for m in range(1, 11)]]
    if not np.isfinite(yields).all():
        raise RuntimeError("yield table has gaps")
    return months, yields


def forecast_errors(months, yields, origins, columns):
    """Scoring loop: one candidate call per origin. Later yields stay in this process."""
    errors = np.empty((len(HORIZONS), len(origins), len(columns)))
    naive = np.empty_like(errors)
    for k, t in enumerate(origins):
        history = {"yyyymm": months[:t + 1].copy(), "maturities": np.arange(1.0, 11.0), "yields": yields[:t + 1].copy()}
        value = yield finance_sandbox.step(ENTRY, history, list(HORIZONS))
        forecast = finite_vector(value, f"forecast at {months[t]}", (len(HORIZONS), 10), bound=30.0)
        for i, h in enumerate(HORIZONS):
            errors[i, k] = forecast[i, columns] - yields[t + h, columns]
            naive[i, k] = yields[t, columns] - yields[t + h, columns]
    return errors, naive


def yield_score(candidate):
    """Mean over horizons and scored maturities of RMSE(candidate) / RMSE(random walk) (minimize).

    ``candidate`` is the candidate module (its forecast_yields runs in the sandbox) or a trusted callable.
    """
    months, yields = load()
    origins = [t for t, m in enumerate(months) if FIRST_ORIGIN <= m <= LAST_ORIGIN]
    if origins[-1] + max(HORIZONS) >= len(months):
        raise RuntimeError("forecast targets exceed the data")
    columns = [m - 1 for m in SCORED]
    errors, naive = finance_sandbox.run(candidate, forecast_errors(months, yields, origins, columns), ENTRY)
    rmse = np.sqrt(np.mean(errors ** 2, axis=1))
    rmse_naive = np.sqrt(np.mean(naive ** 2, axis=1))
    ratio = rmse / rmse_naive
    details = {"origins": len(origins), "first_origin": int(months[origins[0]]), "last_origin": int(months[origins[-1]]),
               "horizons_months": list(HORIZONS), "scored_maturities_years": list(SCORED),
               "rmse_ratio_by_horizon": [float(r.mean()) for r in ratio],
               "rmse_percentage_points": rmse.round(6).tolist(), "random_walk_rmse_percentage_points": rmse_naive.round(6).tolist()}
    for i, h in enumerate(HORIZONS):  # Diebold-Mariano on squared errors averaged over maturities (overlapping errors: h-1 lags)
        stat, p = diebold_mariano((errors[i] ** 2).mean(axis=1), (naive[i] ** 2).mean(axis=1), lags=max(h - 1, 0))
        details[f"diebold_mariano_h{h}"] = stat
        details[f"one_sided_p_better_than_random_walk_h{h}"] = p
    return float(ratio.mean()), details
