"""Out-of-sample portfolio choice against the 1/N rule (DeMiguel, Garlappi and Uppal 2009).

Rolling-window protocol of DGU Section 2: at every month the candidate sees only
the previous M monthly excess returns and returns positions x; the evaluator
forms DGU's relative weights w = x / |1'x|, applies them to the next month's
returns and charges proportional costs c * sum|w_t - w_{t-}| on the trade from
the drifted weights w_{t-} (DGU eq. 16). The score is the monthly Sharpe ratio
of the net out-of-sample excess returns; 1/N is recomputed on the same months.
"""
from __future__ import annotations

import math

import numpy as np

from . import finance_data, finance_sandbox
from .runtime import CandidateError

FACTORS = "ff_factors_monthly.csv"
ENTRY = "choose_weights"
DATASETS = {
    # DGU dataset 2: ten industry portfolios plus the US equity market (N = 11).
    "industry10_market": {"table": "ind10_vw_monthly.csv", "market": True, "drop": ()},
    # DGU dataset 5 ("FF-1-factor"): 25 size/book-to-market portfolios without the
    # five largest-size portfolios, plus the market (N = 21).
    "ff20_market": {"table": "ff25_vw_monthly.csv", "market": True, "drop": ("BIG ", "ME5 ")},
    # Forty-nine industry portfolios, complete from 1969-07 (N = 49); not a DGU dataset.
    "industry49": {"table": "ind49_vw_monthly.csv", "market": False, "drop": ()},
}


def load(dataset, start, end):
    """Months, asset names, monthly excess returns (T x N, decimal) and the risk-free rate."""
    spec = DATASETS[dataset]
    months, columns, values = finance_data.window(spec["table"], start, end)
    f_months, f_columns, factors = finance_data.window(FACTORS, start, end)
    if months != f_months:
        raise RuntimeError("portfolio and factor tables are not aligned")
    rf = factors[:, f_columns.index("RF")] / 100.0
    keep = [j for j, name in enumerate(columns) if not name.startswith(spec["drop"])] if spec["drop"] else list(range(len(columns)))
    excess = values[:, keep] / 100.0 - rf[:, None]
    names = [columns[j] for j in keep]
    if spec["market"]:
        excess = np.column_stack([excess, factors[:, f_columns.index("Mkt-RF")] / 100.0])
        names.append("Mkt")
    if not np.isfinite(excess).all() or (values[:, keep] <= -99.99).any():
        raise RuntimeError(f"{dataset} has missing returns between {start} and {end}")
    return list(months), names, excess, rf


def relative_weights(positions, n, max_gross):
    """Validate candidate positions and normalise them as in DGU eq. (1)."""
    try:
        x = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as error:
        raise CandidateError(f"positions are not numeric: {error}") from None
    if x.shape != (n,):
        raise CandidateError(f"positions must have shape ({n},), got {x.shape}")
    if not np.isfinite(x).all():
        raise CandidateError("positions must be finite")
    total = float(x.sum())
    # Relative to the positions' own size: the weights are scale free, so a small common scale is not a zero sum.
    if abs(total) <= 1e-9 * float(np.abs(x).sum()):
        raise CandidateError("positions sum to (almost) zero; relative weights x/|sum(x)| are undefined")
    w = x / abs(total)
    gross = float(np.abs(w).sum())
    if gross > max_gross + 1e-9:
        raise CandidateError(f"gross exposure {gross:.3f} exceeds the limit {max_gross}")
    return w


def backtest_steps(months, excess, rf, *, window, cost, max_gross, previous_context=True):
    """Scoring loop of the rolling-window protocol; returns per-month arrays and diagnostics.

    Each call receives the estimation window and the drifted weights; the month's returns stay in this process.
    """
    T, n = excess.shape
    if T <= window + 1:
        raise RuntimeError("sample too short for the estimation window")
    net, gross_ret, turnover, exposure = [], [], [], []
    drifted = None
    for t in range(window, T):
        context = {"cost_rate": cost, "window": window, "max_gross": max_gross,
                   "risk_free": rf[t - window:t].copy(),
                   "previous_weights": None if drifted is None or not previous_context else drifted.copy()}
        w = relative_weights((yield finance_sandbox.step(ENTRY, excess[t - window:t].copy(), context)), n, max_gross)
        trade = 0.0 if drifted is None else float(np.abs(w - drifted).sum())
        total_assets = excess[t] + rf[t]
        port = float(rf[t] + w @ excess[t])  # positions financed at the risk-free rate
        wealth = (1.0 - cost * trade) * (1.0 + port)
        if not wealth > 0.0:
            raise CandidateError(f"strategy lost all wealth in month {months[t]} (net gross return {wealth - 1:.3f})")
        net.append(wealth - 1.0 - rf[t])
        gross_ret.append(port - rf[t])
        turnover.append(trade)
        exposure.append(float(np.abs(w).sum()))
        drifted = w * (1.0 + total_assets) / (1.0 + port)
    return {"net": np.array(net), "gross": np.array(gross_ret), "turnover": np.array(turnover),
            "exposure": np.array(exposure), "months": months[window:]}


def backtest(policy, months, excess, rf, **settings):
    """Run the rolling-window protocol for a candidate module (sandboxed) or a trusted callable."""
    return finance_sandbox.run(policy, backtest_steps(months, excess, rf, **settings), ENTRY)


def sharpe(returns):
    sd = float(np.std(returns, ddof=1))
    if not sd > 0:
        raise CandidateError("out-of-sample excess returns have zero variance")
    return float(np.mean(returns)) / sd


def jobson_korkie_memmel(a, b):
    """z statistic and one-sided p-value for SR(a) - SR(b) (Memmel 2003 correction), as in DGU."""
    T = len(a)
    mu_a, mu_b = float(np.mean(a)), float(np.mean(b))
    cov = np.cov(a, b, ddof=1)
    sa, sb, sab = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1]), float(cov[0, 1])
    theta = (2 * sa**2 * sb**2 - 2 * sa * sb * sab + 0.5 * mu_a**2 * sb**2 + 0.5 * mu_b**2 * sa**2
             - mu_a * mu_b / (sa * sb) * sab**2) / T
    if not theta > 0:
        return 0.0, 0.5
    z = (sb * mu_a - sa * mu_b) / math.sqrt(theta)
    return z, 0.5 * math.erfc(abs(z) / math.sqrt(2.0))


def equal_weight(history, context):
    return np.ones(history.shape[1])


def portfolio_choice_score(candidate, dataset, *, start, end, window=120, cost=0.005, max_gross=20.0):
    """Monthly net-of-cost out-of-sample Sharpe ratio of the candidate policy (maximize).

    ``candidate`` is the candidate module (its choose_weights runs in the sandbox) or a trusted callable.
    """
    months, names, excess, rf = load(dataset, start, end)
    run = backtest(candidate, months, excess, rf, window=window, cost=cost, max_gross=max_gross)
    base = backtest(equal_weight, months, excess, rf, window=window, cost=cost, max_gross=max_gross)
    score = sharpe(run["net"])
    z, p = jobson_korkie_memmel(run["net"], base["net"])
    details = {
        "dataset": dataset, "assets": names, "first_month": run["months"][0], "last_month": run["months"][-1],
        "months": len(run["net"]), "window": window, "cost_rate": cost,
        "net_sharpe": score, "gross_sharpe": sharpe(run["gross"]),
        "net_mean": float(np.mean(run["net"])), "net_std": float(np.std(run["net"], ddof=1)),
        "net_ceq_gamma1": float(np.mean(run["net"]) - 0.5 * np.var(run["net"], ddof=1)),
        "mean_turnover": float(np.mean(run["turnover"][1:])), "max_gross_exposure": float(run["exposure"].max()),
        "equal_weight_net_sharpe": sharpe(base["net"]), "equal_weight_gross_sharpe": sharpe(base["gross"]),
        "sharpe_minus_equal_weight": score - sharpe(base["net"]),
        "jobson_korkie_memmel_z": z, "one_sided_p_value_vs_equal_weight": p,
    }
    return score, details
