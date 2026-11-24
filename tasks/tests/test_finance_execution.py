"""Optimal execution: objective formula, closed form, reference optimum and candidate validation."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:  # optional: the test that needs an independent solver is skipped
    minimize = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_execution as fe  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import needs  # noqa: E402


def twap(problem):
    return np.full(problem["periods"], problem["shares"] / problem["periods"])


class ExecutionTests(unittest.TestCase):
    def test_closed_form_reproduces_almgren_chriss_example(self):
        problem = fe.instance(fe.INSTANCES[0])
        trades, kappa = fe.almgren_chriss(problem)
        self.assertAlmostEqual(kappa, 0.6, delta=0.01)  # AC: kappa ~ 0.6/day, kappa*T ~ 3
        self.assertAlmostEqual(kappa * problem["horizon_days"], 3.0, delta=0.05)
        self.assertAlmostEqual(trades.sum(), problem["shares"])
        self.assertTrue((np.diff(trades) < 0).all())  # front-loaded sales

    def test_objective_matches_simulated_shortfall(self):
        problem = fe.instance({"name": "check", "periods": 5, "risk_aversion": 1e-6})
        trades = np.array([4e5, 2.5e5, 1.5e5, 1.2e5, 0.8e5])
        rng = np.random.default_rng(7)
        paths, tau, X = 400_000, problem["tau"], problem["shares"]
        shocks = rng.standard_normal((paths, 5)) * problem["sigma"][0] * np.sqrt(tau)
        price = np.full(paths, problem["price"])
        proceeds = np.zeros(paths)
        for k, n in enumerate(trades):
            rate = n / tau
            proceeds += n * (price - problem["epsilon"] - problem["eta"][k] * rate)  # executed at S_{k-1} - h(v_k)
            price = price + shocks[:, k] - problem["gamma"] * n
        shortfall = X * problem["price"] - proceeds
        _, expected, variance = fe.objective(problem, trades)
        self.assertAlmostEqual(shortfall.mean() / expected, 1.0, delta=4 * shortfall.std() / np.sqrt(paths) / expected)
        self.assertAlmostEqual(shortfall.var() / variance, 1.0, delta=0.01)

    @unittest.skipUnless(minimize, "scipy is not installed")
    @needs("cvxpy")
    def test_reference_optima_match_an_independent_solver(self):
        for index, spec in enumerate(fe.INSTANCES):
            problem = fe.instance(spec)
            N, X = problem["periods"], problem["shares"]
            result = minimize(lambda u: fe.objective(problem, X * np.abs(u))[0] / X, np.full(N, 1 / N), method="SLSQP",
                              bounds=[(0, 1)] * N, constraints=[{"type": "eq", "fun": lambda u: u.sum() - 1}],
                              options={"ftol": 1e-15, "maxiter": 2000})
            best, method = fe.optimum(index)
            self.assertLess(abs(result.fun * X - best) / best, 1e-8, spec["name"])
            if method == "closed_form":
                self.assertEqual(spec.get("beta", 1.0), 1.0)

    @needs("cvxpy")
    def test_scores_seed_closed_form_and_invalid_schedules(self):
        score, details = fe.execution_score(twap)
        self.assertAlmostEqual(score, 221.43, delta=0.05)
        self.assertEqual(len(details["instances"]), 12)

        def perfect(problem):
            index = [s["name"] for s in fe.INSTANCES].index(problem["name"])
            if fe.optimum(index)[1] == "closed_form":
                return fe.almgren_chriss(problem)[0]
            return twap(problem)
        partial, _ = fe.execution_score(perfect)
        self.assertLess(partial, score)
        bad = [lambda p: -twap(p), lambda p: twap(p) * 1.01, lambda p: twap(p)[:-1], lambda p: twap(p) * np.nan,
               lambda p: np.r_[2 * p["shares"], -p["shares"], np.zeros(p["periods"] - 2)]]
        for candidate in bad:
            with self.assertRaises(CandidateError):
                fe.execution_score(candidate)

    @needs("cvxpy")
    def test_candidate_cannot_change_the_instances(self):
        def vandal(problem):
            problem["eta"][:] = 0.0
            problem["sigma"][:] = 0.0
            return twap(problem)
        self.assertAlmostEqual(fe.execution_score(vandal)[0], fe.execution_score(twap)[0])


if __name__ == "__main__":
    unittest.main()
