"""Index tracking with a cardinality limit and transaction costs (Beasley, Meade and Chang 2003).

OR-Library markets indtrack1-5 give weekly prices of an index and its
constituents for weeks 0..290. At week 145 the candidate invests in at most
K = 10 stocks (each held weight between 1% and 100%) using prices up to that
week only; every 13 weeks afterwards it may rebalance, paying 1% of the traded
value. The portfolio holds whole positions in shares between rebalances (its
weights drift with prices). The score is the out-of-sample tracking error, the
root-mean-square difference between weekly log returns of the portfolio (net
of costs) and of the index over weeks 146..290, in percent, averaged over the
five markets.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import finance_data
from .finance_stats import finite_vector
from .runtime import CandidateError

MARKETS = {1: "Hang Seng", 2: "DAX 100", 3: "FTSE 100", 4: "S&P 100", 5: "Nikkei 225"}
K, MIN_WEIGHT, MAX_WEIGHT, COST = 10, 0.01, 1.0, 0.01
START, END, EVERY = 145, 290, 13


@lru_cache(maxsize=None)
def market(index):
    """(index prices (T+1,), stock prices (T+1, N)) for weeks 0..290."""
    _, columns, values = finance_data.table(f"orlib_indtrack{index}.csv")
    if len(values) != END + 1 or columns[0] != "index":
        raise RuntimeError(f"indtrack{index} has an unexpected layout")
    return values[:, 0], values[:, 1:]


def validate(raw, n, week):
    w = finite_vector(raw, f"weights at week {week}", (n,))
    w = np.where(np.abs(w) <= 1e-12, 0.0, w)  # numerical zeros count as not held
    held = w > 0
    if (w < 0).any() or int(held.sum()) > K or int(held.sum()) == 0:
        raise CandidateError(f"week {week}: need nonnegative weights on 1 to {K} stocks")
    if (w[held] < MIN_WEIGHT - 1e-9).any() or (w[held] > MAX_WEIGHT + 1e-9).any() or abs(float(w.sum()) - 1.0) > 1e-9:
        raise CandidateError(f"week {week}: held weights must lie in [{MIN_WEIGHT}, {MAX_WEIGHT}] and sum to 1")
    return w


def trade(shares, prices, weights, value, charge):
    """New shares after rebalancing to target weights, paying COST on traded value (value after costs solves a 1-D fixed point)."""
    current = shares * prices
    if not charge:
        return weights * value / prices, value, 0.0
    lo, hi = value * (1.0 - 2.0 * COST), value
    for _ in range(100):  # bisection on V' = V - c * sum|w V' - current|
        mid = 0.5 * (lo + hi)
        if mid > value - COST * np.abs(weights * mid - current).sum():
            hi = mid
        else:
            lo = mid
    after = 0.5 * (lo + hi)
    return weights * after / prices, after, float(np.abs(weights * after - current).sum())


def track(rebalance, index):
    """Weekly log returns of portfolio and index over the out-of-sample weeks, plus traded value."""
    level, prices = market(index)
    n = prices.shape[1]
    shares = np.zeros(n)
    portfolio, traded = [], 0.0
    for week in range(START, END):
        base = 1.0 if week == START else float(shares @ prices[week])  # value before any trade this week
        if (week - START) % EVERY == 0:
            state = {"week": week, "index_prices": level[:week + 1].copy(), "stock_prices": prices[:week + 1].copy(),
                     "holdings": shares.copy(), "portfolio_value": base, "K": K, "min_weight": MIN_WEIGHT,
                     "max_weight": MAX_WEIGHT, "cost_rate": COST, "rebalance_every": EVERY}
            raw = rebalance(state)
            if raw is None and week == START:
                raise CandidateError("an initial portfolio is required at week 145")
            if raw is not None:  # the initial purchase is free; later trades pay COST and lower this week's return
                shares, _, amount = trade(shares, prices[week], validate(raw, n, week), base, week > START)
                traded += amount
        portfolio.append(np.log(float(shares @ prices[week + 1]) / base))
    benchmark = np.diff(np.log(level[START:]))
    return np.array(portfolio), benchmark, traded


def tracking_score(rebalance):
    """Mean over markets of the out-of-sample tracking error (percent per week; minimize)."""
    if not callable(rebalance):
        raise CandidateError("rebalance must be callable")
    rows = {}
    for index, name in MARKETS.items():
        r, b, traded = track(rebalance, index)
        rows[name] = {"tracking_error_percent": 100.0 * float(np.sqrt(np.mean((r - b) ** 2))),
                      "excess_return_percent_per_year": 100.0 * 52.0 * float(np.mean(r - b)),
                      "traded_value_after_start": traded}
    score = float(np.mean([v["tracking_error_percent"] for v in rows.values()]))
    return score, {"markets": rows, "weeks": END - START, "K": K, "cost_rate": COST}
