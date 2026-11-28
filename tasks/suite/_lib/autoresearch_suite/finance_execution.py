"""Optimal execution of a sell program (Almgren and Chriss 2000; Almgren 2003).

Discrete Almgren-Chriss model: X shares are sold in N periods of length tau.
Selling n_k shares in period k costs the temporary impact
epsilon*|n_k| + tau*eta_k*|n_k/tau|^(1+beta) and, through linear permanent
impact gamma, the path-independent part gamma*X^2/2 minus gamma*sum(n_k^2)/2.
Holdings x_k = X - (n_1 + ... + n_k) are exposed to price risk
tau*sigma_k^2*x_k^2. A candidate schedule is scored by
U = E[shortfall] + lambda*Var[shortfall] against the exact optimum, which is
the Almgren-Chriss closed form for linear impact and the unique convex
optimum (solved with CVXPY/Clarabel) otherwise.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from . import finance_sandbox
from .finance_stats import finite_vector
from .runtime import CandidateError

ENTRY = "trade_schedule"
AC = {"shares": 1e6, "price": 50.0, "horizon_days": 5.0, "sigma": 0.95, "epsilon": 0.0625, "gamma": 2.5e-7, "eta": 2.5e-6}
DAILY_VOLUME = 5e6  # AC Table 1: median daily volume


def _power_eta(beta):
    """Power-law coefficient with the same impact as AC's linear eta at 1% of daily volume."""
    rate = 0.01 * DAILY_VOLUME
    return AC["eta"] * rate / rate ** beta


def _intraday(n, kind):
    """Stylized U-shaped intraday profile on n equal bins (volume share or variance weight), mean 1."""
    s = (np.arange(n) + 0.5) / n
    weight = 1.0 + (2.0 if kind == "volume" else 1.5) * (2.0 * s - 1.0) ** 2
    return weight / weight.mean()


def instance(spec):
    """Expand one instance specification into the arrays given to the candidate."""
    n = spec["periods"]
    horizon = spec.get("horizon_days", AC["horizon_days"])
    beta = spec.get("beta", 1.0)
    eta = np.full(n, AC["eta"] if beta == 1.0 else _power_eta(beta))
    sigma = np.full(n, spec.get("sigma", AC["sigma"]))
    if spec.get("intraday"):
        eta = eta / _intraday(n, "volume") ** beta  # less liquidity, larger impact per share in thin bins
        sigma = sigma * np.sqrt(_intraday(n, "variance"))
    return {"shares": spec.get("shares", AC["shares"]), "periods": n, "horizon_days": horizon, "tau": horizon / n,
            "price": AC["price"], "sigma": sigma, "eta": eta, "beta": beta, "epsilon": AC["epsilon"],
            "gamma": AC["gamma"] if beta == 1.0 else 0.0, "risk_aversion": spec["risk_aversion"], "name": spec["name"]}


INSTANCES = [
    {"name": "ac-table1", "periods": 5, "risk_aversion": 1e-6},
    {"name": "ac-urgent", "periods": 5, "risk_aversion": 1e-5},
    {"name": "ac-20", "periods": 20, "risk_aversion": 2e-6},
    {"name": "ac-50-patient", "periods": 50, "risk_aversion": 1e-7},
    {"name": "ac-50-urgent", "periods": 50, "risk_aversion": 1e-5},
    {"name": "ac-volatile", "periods": 50, "risk_aversion": 5e-7, "sigma": 2.0},
    {"name": "power-0.6-20", "periods": 20, "risk_aversion": 1e-6, "beta": 0.6},
    {"name": "power-0.6-50-urgent", "periods": 50, "risk_aversion": 1e-5, "beta": 0.6},
    {"name": "square-root-20", "periods": 20, "risk_aversion": 3e-6, "beta": 0.5},
    {"name": "large-order", "periods": 50, "risk_aversion": 1e-6, "beta": 0.6, "shares": 5e6, "horizon_days": 10.0},
    {"name": "intraday-linear", "periods": 13, "risk_aversion": 1e-6, "horizon_days": 1.0, "shares": 5e5, "intraday": True},
    {"name": "intraday-power", "periods": 26, "risk_aversion": 1e-5, "horizon_days": 1.0, "shares": 5e5, "intraday": True, "beta": 0.6},
]


def objective(problem, trades):
    """Exact mean-variance objective U = E + lambda*V and its parts for a trade vector (no validation)."""
    n, tau, X = np.asarray(trades, float), problem["tau"], problem["shares"]
    holdings = X - np.cumsum(n)
    expected = (0.5 * problem["gamma"] * X ** 2 - 0.5 * problem["gamma"] * float(np.sum(n ** 2))
                + problem["epsilon"] * float(np.sum(np.abs(n)))
                + float(np.sum(tau * problem["eta"] * np.abs(n / tau) ** (1.0 + problem["beta"]))))
    variance = tau * float(np.sum(problem["sigma"] ** 2 * holdings ** 2))
    return expected + problem["risk_aversion"] * variance, expected, variance


def almgren_chriss(problem):
    """Closed-form optimal trades for constant linear impact (AC 2000, eq. for x_j with kappa from cosh)."""
    tau, T, X = problem["tau"], problem["horizon_days"], problem["shares"]
    eta_tilde = problem["eta"][0] - 0.5 * problem["gamma"] * tau
    kappa_tilde2 = problem["risk_aversion"] * problem["sigma"][0] ** 2 / eta_tilde
    kappa = math.acosh(1.0 + 0.5 * kappa_tilde2 * tau ** 2) / tau
    times = np.arange(problem["periods"] + 1) * tau
    holdings = X * np.sinh(kappa * (T - times)) / math.sinh(kappa * T)
    return -np.diff(holdings), kappa


@lru_cache(maxsize=None)
def optimum(index):
    """Exact optimal objective of INSTANCES[index] (closed form when it applies, else convex solve)."""
    problem = instance(INSTANCES[index])
    constant = problem["beta"] == 1.0 and np.ptp(problem["eta"]) == 0 and np.ptp(problem["sigma"]) == 0
    if constant:
        trades, _ = almgren_chriss(problem)
        return objective(problem, trades)[0], "closed_form"
    import cvxpy as cp
    N, X, tau, beta = problem["periods"], problem["shares"], problem["tau"], problem["beta"]
    u = cp.Variable(N, nonneg=True)  # fraction of X sold per period
    holdings = 1.0 - cp.cumsum(u)
    scale = X  # objective in dollars divided by X
    if beta == 1.0:
        impact = cp.sum(cp.multiply(problem["eta"] / tau - 0.5 * problem["gamma"], cp.square(u))) * X
    else:
        impact = cp.sum(cp.multiply(tau * problem["eta"] * (X / tau) ** (1.0 + beta) / scale, cp.power(u, 1.0 + beta)))
    risk = problem["risk_aversion"] * tau * X * cp.sum(cp.multiply(problem["sigma"][:-1] ** 2, cp.square(holdings[:-1])))
    cp.Problem(cp.Minimize(impact + risk), [cp.sum(u) == 1.0]).solve(solver="CLARABEL")
    trades = X * np.clip(u.value, 0.0, None)
    trades *= X / trades.sum()
    return objective(problem, trades)[0], "convex_solver"


def validate(problem, trades):
    """Accept a schedule within the tolerances and return the one that is scored: nonnegative and summing to X.

    The tolerances exist for rounding. Scoring the schedule as returned would reward selling 1e-6 less than the
    order with an objective below the optimum, so the accepted schedule is rescaled to the order size.
    """
    X, N = problem["shares"], problem["periods"]
    n = finite_vector(trades, f"trades for {problem['name']}", (N,))
    if (n < -1e-9 * X).any():
        raise CandidateError(f"{problem['name']}: a sell program cannot buy (negative trade)")
    if abs(float(n.sum()) - X) > 1e-6 * X:
        raise CandidateError(f"{problem['name']}: trades sum to {n.sum():.6g}, not the order size {X:.6g}")
    n = np.clip(n, 0.0, None)
    total = float(n.sum())
    return n if total == X else n * (X / total)


def schedules():
    """Scoring loop: one candidate call per instance; the optimum stays in this process."""
    rows = []
    for spec in INSTANCES:
        problem = instance(spec)
        given = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in problem.items()}
        rows.append((problem, validate(problem, (yield finance_sandbox.step(ENTRY, given)))))
    return rows


def execution_score(candidate):
    """Mean percent excess of U over the exact optimum across the fixed instances (minimize; 0 is optimal).

    ``candidate`` is the candidate module (its trade_schedule runs in the sandbox) or a trusted callable.
    """
    rows, excess = [], []
    for index, (spec, (problem, n)) in enumerate(zip(INSTANCES, finance_sandbox.run(candidate, schedules(), ENTRY))):
        value, expected, variance = objective(problem, n)
        best, method = optimum(index)
        gap = 100.0 * (value - best) / best
        excess.append(gap)
        rows.append({"instance": spec["name"], "objective": value, "optimum": best, "excess_percent": gap,
                     "expected_shortfall": expected, "shortfall_std": math.sqrt(variance), "reference": method})
    return float(np.mean(excess)), {"instances": rows, "max_excess_percent": float(np.max(excess))}
