"""Macroeconomic forecasting with many predictors (Stock and Watson 2002; FRED-MD, McCracken and Ng 2016).

At each month-end origin from 1990-01 to 2023-08 the candidate receives the
FRED-MD panel (126 series in levels, 2024-09 vintage) up to the origin and
forecasts a target 1, 6 and 12 months ahead: annualized industrial-production
growth, annualized CPI inflation or the change in the unemployment rate. The
score is the RMSE relative to a direct autoregression with BIC lag selection,
the standard benchmark, averaged over the three horizons. Forecasts whose
target window includes March-December 2020 are excluded from the RMSE so that
the pandemic shock does not dominate the comparison.
"""
from __future__ import annotations

import numpy as np

from . import finance_data, finance_sandbox
from .finance_stats import diebold_mariano, finite_vector

# The file name keeps the date of the archived snapshot (2024-10-29); its bytes are the official 2024-09 vintage,
# whose last month is 2024-08.
TABLE = "fredmd_2024_10.csv"
ENTRY = "forecast_target"
KINDS = {"INDPRO": "growth", "CPIAUCSL": "growth", "UNRATE": "difference"}
HORIZONS = (1, 6, 12)
FIRST_ORIGIN, LAST_ORIGIN, ESTIMATION_START = 199001, 202308, 196001
MAX_LAG = 6
EXCLUDED = (202003, 202012)  # pandemic months: target windows touching them are not scored


def load():
    keys, names, values = finance_data.table(TABLE)
    if keys[0] != "tcode":
        raise RuntimeError("FRED-MD table must start with the transformation codes")
    return np.array([int(k) for k in keys[1:]]), list(names), values[0].astype(int), values[1:]


def monthly(levels, kind):
    """Monthly transformation of the target level series: 1200*dlog for growth, first difference otherwise."""
    return np.r_[np.nan, (1200.0 * np.diff(np.log(levels))) if kind == "growth" else np.diff(levels)]


def ahead(levels, kind, t, h):
    """Target h months after month index t: annualized average growth or the level change."""
    if kind == "growth":
        return 1200.0 / h * np.log(levels[t + h] / levels[t])
    return levels[t + h] - levels[t]


def ar_bic(history, horizons):
    """Direct AR forecasts of the target with lag order 0..6 chosen by BIC on a common sample."""
    months, kind = history["yyyymm"], history["target_kind"]
    levels = history["panel"][:, history["series"].index(history["target"])]
    z = monthly(levels, kind)
    t = len(levels) - 1
    first = int(np.searchsorted(months, ESTIMATION_START)) + MAX_LAG
    out = []
    for h in horizons:
        rows = np.arange(first, t - h + 1)
        y = np.array([ahead(levels, kind, s, h) for s in rows])
        lags = np.column_stack([z[rows - j] for j in range(MAX_LAG)])
        now = np.array([z[t - j] for j in range(MAX_LAG)])
        best = None
        for p in range(MAX_LAG + 1):
            X = np.column_stack([np.ones(len(rows)), lags[:, :p]])
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            ssr = float(np.sum((y - X @ coef) ** 2))
            bic = len(y) * np.log(ssr / len(y)) + (p + 1) * np.log(len(y))
            if best is None or bic < best[0]:
                best = (bic, float(coef @ np.r_[1.0, now[:p]]))
        out.append(best[1])
    return np.array(out)


def forecasts(months, names, tcodes, panel, target, origins):
    """Scoring loop: one candidate call per origin. Later months and the realised targets stay in this process."""
    kind = KINDS[target]
    levels = panel[:, names.index(target)]
    predicted, bench, actual = (np.empty((len(origins), len(HORIZONS))) for _ in range(3))
    for k, t in enumerate(origins):
        history = {"yyyymm": months[:t + 1].copy(), "series": list(names), "tcodes": tcodes.copy(),
                   "panel": panel[:t + 1].copy(), "target": target, "target_kind": kind}
        bench[k] = ar_bic(history, HORIZONS)
        value = yield finance_sandbox.step(ENTRY, history, list(HORIZONS))
        predicted[k] = finite_vector(value, f"forecast at {months[t]}", (len(HORIZONS),), bound=1e3)
        actual[k] = [ahead(levels, kind, t, h) for h in HORIZONS]
    return predicted, bench, actual


def macro_score(candidate, target):
    """Mean over horizons of RMSE(candidate) / RMSE(direct AR-BIC) for one FRED-MD target (minimize).

    ``candidate`` is the candidate module (its forecast_target runs in the sandbox) or a trusted callable.
    """
    months, names, tcodes, panel = load()
    origins = [t for t, m in enumerate(months) if FIRST_ORIGIN <= m <= LAST_ORIGIN]
    if origins[-1] + max(HORIZONS) >= len(months):
        raise RuntimeError("forecast targets exceed the data")
    predicted, bench, actual = finance_sandbox.run(candidate, forecasts(months, names, tcodes, panel, target, origins), ENTRY)
    scored = np.array([[not any(EXCLUDED[0] <= months[t + j] <= EXCLUDED[1] for j in range(1, h + 1)) for h in HORIZONS]
                       for t in origins])
    rmse = np.array([np.sqrt(np.mean((predicted[m, i] - actual[m, i]) ** 2)) for i, m in enumerate(scored.T)])
    rmse_ar = np.array([np.sqrt(np.mean((bench[m, i] - actual[m, i]) ** 2)) for i, m in enumerate(scored.T)])
    details = {"target": target, "origins": len(origins), "first_origin": int(months[origins[0]]),
               "last_origin": int(months[origins[-1]]), "scored_forecasts_by_horizon": scored.sum(axis=0).tolist(),
               "rmse": rmse.tolist(), "ar_bic_rmse": rmse_ar.tolist(), "rmse_ratio_by_horizon": (rmse / rmse_ar).tolist()}
    for i, h in enumerate(HORIZONS):
        m = scored[:, i]
        stat, p = diebold_mariano((predicted[m, i] - actual[m, i]) ** 2, (bench[m, i] - actual[m, i]) ** 2, lags=max(h - 1, 0))
        details[f"one_sided_p_better_than_ar_h{h}"] = p
    return float(np.mean(rmse / rmse_ar)), details
