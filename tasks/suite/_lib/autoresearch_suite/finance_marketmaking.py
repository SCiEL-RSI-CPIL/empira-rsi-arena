"""Market making with inventory risk (Avellaneda and Stoikov 2008; Gueant, Lehalle and Fernandez-Tapia 2013).

Discrete-time simulator of the Avellaneda-Stoikov model: in each step of
length dt the market maker quotes depths (db, da) around the mid price; the
bid is hit with probability 1 - exp(-A exp(-k db) dt) and the ask is lifted
with probability 1 - exp(-A exp(-k da) dt), independently; each fill earns its
depth and moves inventory by one unit within |q| <= Q. The objective is the
risk-adjusted P&L of Gueant (2017): spread captured minus the running penalty
(gamma/2) sigma^2 q^2 dt (the mark-to-market term q dS has zero mean for every
non-anticipating policy and is omitted to reduce variance). It is estimated on
fixed random numbers and divided by the value of the exact optimal policy of
this discrete model on the same random numbers, computed by dynamic
programming.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from . import finance_sandbox
from .runtime import CandidateError

ENTRY = "quote"
REGIMES = [  # Avellaneda-Stoikov's example (sigma 2, A 140, k 1.5, T 1) with two risk aversions, and a thinner market
    {"name": "as-low-risk-aversion", "sigma": 2.0, "A": 140.0, "k": 1.5, "gamma": 0.05, "T": 1.0, "steps": 1000, "Q": 30, "seed": 501},
    {"name": "as-high-risk-aversion", "sigma": 2.0, "A": 140.0, "k": 1.5, "gamma": 0.5, "T": 1.0, "steps": 1000, "Q": 30, "seed": 502},
    {"name": "thin-market", "sigma": 1.5, "A": 50.0, "k": 0.75, "gamma": 0.2, "T": 1.0, "steps": 1000, "Q": 30, "seed": 503},
]
PATHS = 4000


def fill_probability(depth, regime):
    dt = regime["T"] / regime["steps"]
    return -np.expm1(-regime["A"] * np.exp(-regime["k"] * depth) * dt)


def _best_depth(constant, regime):
    """argmax over d of p(d) * (d + constant), solved by Newton steps on the first-order condition (vectorized)."""
    k, rate = regime["k"], regime["A"] * regime["T"] / regime["steps"]
    d = 1.0 / k - constant
    for _ in range(50):
        lam = rate * np.exp(-k * d)
        p, dp = -np.expm1(-lam), -k * lam * np.exp(-lam)
        ddp = k * k * lam * np.exp(-lam) * (1.0 - lam)
        f, df = dp * (d + constant) + p, ddp * (d + constant) + 2.0 * dp
        d = d - np.clip(f / np.where(df < 0, df, -1e-12), -1.0, 1.0)
    return d


@lru_cache(maxsize=None)
def optimal_policy(index):
    """Exact optimal depths of the discrete model by backward induction: tables (steps, 2Q+1) for bid and ask."""
    r = REGIMES[index]
    q = np.arange(-r["Q"], r["Q"] + 1)
    dt = r["T"] / r["steps"]
    theta = np.zeros(len(q))
    bids, asks = np.zeros((r["steps"], len(q))), np.zeros((r["steps"], len(q)))
    for n in range(r["steps"] - 1, -1, -1):
        up = np.r_[theta[1:] - theta[:-1], 0.0]   # theta(q+1) - theta(q)
        down = np.r_[0.0, theta[:-1] - theta[1:]]  # theta(q-1) - theta(q)
        pa = np.zeros(len(q))
        for _ in range(6):  # coordinate ascent on the coupled bid/ask problem
            db = _best_depth((1.0 - pa) * up - pa * down, r)
            pb = np.where(q < r["Q"], fill_probability(db, r), 0.0)
            da = _best_depth((1.0 - pb) * down - pb * up, r)
            pa = np.where(q > -r["Q"], fill_probability(da, r), 0.0)
        gain = pb * (db + (1.0 - pa) * up - pa * down) + pa * (da + down)
        theta = theta + gain - 0.5 * r["gamma"] * r["sigma"] ** 2 * q ** 2 * dt
        bids[n], asks[n] = db, da
    return bids, asks, float(theta[r["Q"]])


def simulate_steps(index):
    """Scoring loop for one regime: per-path risk-adjusted P&L on the regime's fixed random numbers.

    Each call receives the time, inventories and mid prices; the random numbers that decide the fills stay here.
    """
    r = REGIMES[index]
    rng = np.random.default_rng(r["seed"])
    dt, Q = r["T"] / r["steps"], r["Q"]
    params = {key: r[key] for key in ("sigma", "A", "k", "gamma", "T", "steps", "Q")}
    q = np.zeros(PATHS, dtype=np.int64)
    mid = np.full(PATHS, 100.0)
    spread, penalty, fills = np.zeros(PATHS), np.zeros(PATHS), 0
    for n in range(r["steps"]):
        u_bid, u_ask, z = rng.random(PATHS), rng.random(PATHS), rng.standard_normal(PATHS)
        out = yield finance_sandbox.step(ENTRY, n * dt, q.copy(), mid.copy(), dict(params))
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise CandidateError("quote must return (bid_depths, ask_depths)")
        db, da = (np.asarray(x, dtype=float) for x in out)
        for name, depth in (("bid", db), ("ask", da)):
            if depth.shape != (PATHS,) or np.isnan(depth).any() or (depth < -3.0 / r["k"]).any() or np.isneginf(depth).any():
                raise CandidateError(f"{name} depths must be {PATHS} values >= -3/k (inf = no quote)")
        hit_bid = (u_bid < np.where(q < Q, fill_probability(db, r), 0.0)) & np.isfinite(db)
        hit_ask = (u_ask < np.where(q > -Q, fill_probability(da, r), 0.0)) & np.isfinite(da)
        penalty += 0.5 * r["gamma"] * r["sigma"] ** 2 * q ** 2 * dt
        spread += np.where(hit_bid, db, 0.0) + np.where(hit_ask, da, 0.0)
        fills += int(hit_bid.sum() + hit_ask.sum())
        q = q + hit_bid - hit_ask
        mid = mid + r["sigma"] * math.sqrt(dt) * z
    return spread - penalty, {"mean_abs_terminal_inventory": float(np.abs(q).mean()), "fills_per_path": fills / PATHS,
                              "mean_spread_capture": float(spread.mean()), "mean_inventory_penalty": float(penalty.mean())}


def simulate(quote, index):
    """Per-path risk-adjusted P&L of a candidate module (sandboxed) or a trusted quoting function."""
    return finance_sandbox.run(quote, simulate_steps(index), ENTRY)


def table_policy(index):
    """The dynamic-programming policy as a quoting function (used for normalization)."""
    bids, asks, _ = optimal_policy(index)
    r = REGIMES[index]

    def quote(t, q, mid, params):
        n = min(int(round(t * r["steps"] / r["T"])), r["steps"] - 1)
        return bids[n, q + r["Q"]], asks[n, q + r["Q"]]
    return quote


def glft_approximation(t, q, mid, params):
    """Closed-form approximation (model B): 1/k +- (2q +- 1)/2 * sqrt(gamma sigma^2 e / (2 A k))."""
    k, slope = params["k"], math.sqrt(params["gamma"] * params["sigma"] ** 2 * math.e / (2.0 * params["A"] * params["k"]))
    return 1.0 / k + (2 * q + 1) / 2.0 * slope, 1.0 / k - (2 * q - 1) / 2.0 * slope


def regimes():
    """Scoring loop over the regimes, in one session."""
    results = []
    for index in range(len(REGIMES)):
        results.append((yield from simulate_steps(index)))
    return results


def market_making_score(candidate):
    """Mean over regimes of mean risk-adjusted P&L / that of the exact optimal policy (maximize; about 1 at the optimum).

    ``candidate`` is the candidate module (its quote runs in the sandbox) or a trusted callable.
    """
    rows, ratios = [], []
    simulated = finance_sandbox.run(candidate, regimes(), ENTRY)
    for index, regime in enumerate(REGIMES):
        values, stats = simulated[index]
        reference, _ = simulate(table_policy(index), index)
        ratios.append(float(values.mean() / reference.mean()))
        rows.append({"regime": regime["name"], "mean_value": float(values.mean()), "optimal_policy_value": float(reference.mean()),
                     "dynamic_programming_value": optimal_policy(index)[2], "ratio": ratios[-1], **stats})
    return float(np.mean(ratios)), {"regimes": rows, "paths": PATHS}
