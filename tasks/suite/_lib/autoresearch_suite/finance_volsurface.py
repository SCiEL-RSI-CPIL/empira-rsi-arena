"""Arbitrage-free SVI implied-volatility surface (Gatheral and Jacquier 2014) on DAX quotes.

The candidate returns raw SVI parameters (a, b, rho, m, sigma) for each of the
eight maturity slices of the DAX surface of 2002-07-05. Total implied variance
w(k) = a + b*(rho*(k - m) + sqrt((k - m)^2 + sigma^2)) in log-forward moneyness
k = ln(K/F). The evaluator requires static no-arbitrage: positive variance,
Durrleman's butterfly condition g(k) >= 0 and the wing bound b(1+|rho|) <= 2 on
every slice, and no calendar spread arbitrage (w increasing in maturity for
every k). The score is the root-mean-square implied-volatility error in
volatility points over the 104 quotes.
"""
from __future__ import annotations

import math

import numpy as np

from . import finance_data, finance_sandbox
from .finance_stats import finite_vector
from .runtime import CandidateError

TABLE = "dax_2002_surface.csv"
ENTRY = "fit_surface"
TOLERANCE = 1e-10       # allowed negative violation of g(k) and of calendar spreads (total variance units)
K_RANGE = 6.0           # dense checks on |k| <= 6 plus refinement near every SVI vertex
MIN_SIGMA = 1e-4


def load():
    """Quotes grouped by maturity: maturities (years), forwards, strikes, implied vols (slice x strike)."""
    _, columns, values = finance_data.table(TABLE)
    col = {name: values[:, j] for j, name in enumerate(columns)}
    days = np.unique(col["maturity_days"])
    strikes = np.unique(col["strike"])
    vols = np.empty((len(days), len(strikes)))
    rates = np.empty(len(days))
    for i, d in enumerate(days):
        rows = col["maturity_days"] == d
        order = np.argsort(col["strike"][rows])
        vols[i] = col["implied_vol"][rows][order]
        rates[i] = col["zero_rate"][rows][0]
    spot = float(col["spot"][0])
    maturities = days / 365.0
    forwards = spot * np.exp(rates * maturities)  # DAX is a total-return index: no dividend yield
    return {"spot": spot, "maturities": maturities, "maturity_days": days, "zero_rates": rates, "forwards": forwards,
            "strikes": strikes, "implied_vol": vols,
            "log_moneyness": np.log(strikes[None, :] / forwards[:, None]),
            "total_variance": vols ** 2 * maturities[:, None]}


def svi(params, k):
    a, b, rho, m, sigma = params
    root = np.sqrt((k - m) ** 2 + sigma ** 2)
    w = a + b * (rho * (k - m) + root)
    w1 = b * (rho + (k - m) / root)
    w2 = b * sigma ** 2 / root ** 3
    return w, w1, w2


def durrleman(params, k):
    """Gatheral-Jacquier butterfly function g(k); non-negative everywhere means no butterfly arbitrage."""
    w, w1, w2 = svi(params, k)
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1 ** 2 * (1.0 / w + 0.25) + 0.5 * w2


def check_grid(params_list):
    """Uniform grid on [-K_RANGE, K_RANGE] refined within 25 sigma of every vertex (steps of sigma/40)."""
    parts = [np.linspace(-K_RANGE, K_RANGE, 24001)]
    for _, _, _, m, sigma in params_list:
        parts.append(m + sigma * np.linspace(-25.0, 25.0, 2001))
    grid = np.unique(np.concatenate(parts))
    return grid[np.abs(grid) <= K_RANGE]


def validate(raw, slices):
    params = finite_vector(raw, "SVI parameters", (slices, 5))
    for i, (a, b, rho, m, sigma) in enumerate(params):
        if b < 0 or not abs(rho) < 1 or sigma < MIN_SIGMA:
            raise CandidateError(f"slice {i}: need b >= 0, |rho| < 1 and sigma >= {MIN_SIGMA}")
        if a + b * sigma * math.sqrt(1.0 - rho * rho) <= 0:
            raise CandidateError(f"slice {i}: minimum total variance is not positive")
        if b * (1.0 + abs(rho)) > 2.0:
            raise CandidateError(f"slice {i}: wing slope b(1+|rho|) = {b * (1 + abs(rho)):.4f} exceeds 2 (Lee bound)")
    grid = check_grid(params)
    worst_g = []
    for i, p in enumerate(params):
        values = durrleman(p, grid)
        g = float(np.min(values))
        worst_g.append(g)
        if g < -TOLERANCE:
            raise CandidateError(f"slice {i}: butterfly arbitrage, min g(k) = {g:.3e} at k = {grid[np.argmin(values)]:.4f}")
    worst_calendar = []
    for i in range(slices - 1):
        gap = svi(params[i + 1], grid)[0] - svi(params[i], grid)[0]
        worst_calendar.append(float(np.min(gap)))
        if worst_calendar[-1] < -TOLERANCE:
            raise CandidateError(f"slices {i}->{i + 1}: calendar arbitrage, min w difference {worst_calendar[-1]:.3e} "
                                 f"at k = {grid[np.argmin(gap)]:.4f}")
        for side in (1.0, -1.0):  # wings: later slices need slopes at least as steep
            slope_now = params[i][1] * (1.0 + side * params[i][2])
            slope_next = params[i + 1][1] * (1.0 + side * params[i + 1][2])
            if slope_next < slope_now - 1e-12:
                raise CandidateError(f"slices {i}->{i + 1}: wing slope decreases with maturity (calendar arbitrage for large |k|)")
    return params, min(worst_g), min(worst_calendar)


def fitted(data):
    """Scoring loop: one candidate call with the quotes; the arbitrage checks run in this process."""
    quotes = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in data.items()}
    return validate((yield finance_sandbox.step(ENTRY, quotes)), len(data["maturities"]))


def surface_score(candidate):
    """RMSE of implied volatility in volatility points over all quotes (minimize).

    ``candidate`` is the candidate module (its fit_surface runs in the sandbox) or a trusted callable.
    """
    data = load()
    params, min_g, min_calendar = finance_sandbox.run(candidate, fitted(data), ENTRY)
    errors = np.empty_like(data["implied_vol"])
    for i, p in enumerate(params):
        w = svi(p, data["log_moneyness"][i])[0]
        errors[i] = np.sqrt(w / data["maturities"][i]) - data["implied_vol"][i]
    rmse = 100.0 * float(np.sqrt(np.mean(errors ** 2)))
    details = {"rmse_vol_points": rmse, "max_abs_error_vol_points": 100.0 * float(np.max(np.abs(errors))),
               "slice_rmse_vol_points": [100.0 * float(np.sqrt(np.mean(e ** 2))) for e in errors],
               "min_durrleman_g": min_g, "min_calendar_gap": min_calendar, "quotes": int(errors.size)}
    return rmse, details
