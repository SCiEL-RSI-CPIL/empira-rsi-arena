"""Cardinality-constrained frontier: data reconstruction, percentage error, validation and scoring."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_ccef as fc  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import have_data, needs, seed_source  # noqa: E402

TABLES = [f"orlib_port{i}.csv" for i in range(1, 6)] + [f"orlib_portef{i}.csv" for i in range(1, 6)]


@unittest.skipUnless(have_data(*TABLES), "run scripts/prepare_finance.py first")
class FrontierTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("cvxpy"), "cvxpy is not installed")
    def test_parsed_statistics_reproduce_the_published_frontier(self):
        import cvxpy as cp
        for index in fc.MARKETS:
            mean, cov, uef_r, uef_s = fc.market(index)
            self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)
            w, target = cp.Variable(len(mean)), cp.Parameter()
            problem = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), [cp.sum(w) == 1, w >= 0, mean @ w == target])
            for k in range(0, 2000, 250):
                target.value = uef_r[k]
                problem.solve(solver="CLARABEL")
                self.assertLess(abs(problem.value - uef_s[k] ** 2) / uef_s[k] ** 2, 1e-4)

    def test_percentage_error(self):
        _, _, uef_r, uef_s = fc.market(1)
        for k in (0, 700, 1999):
            self.assertAlmostEqual(fc.percentage_error(uef_s[k], uef_r[k], uef_r, uef_s), 0.0, places=9)
        s, r = uef_s[800] * 1.02, uef_r[800]
        std_error = 2.0
        self.assertLessEqual(fc.percentage_error(s, r, uef_r, uef_s), std_error + 1e-9)
        with self.assertRaises(CandidateError):
            fc.percentage_error(10.0, 10.0, uef_r, uef_s)

    def test_targets_and_validation(self):
        mean, _, uef_r, _ = fc.market(1)
        goals = fc.targets(1)
        self.assertEqual(len(goals), 51)
        self.assertAlmostEqual(goals[0], uef_r[0])
        top = np.argsort(-mean)[:10]
        best = np.zeros(len(mean))
        best[top] = 0.01
        best[top[0]] = 0.91
        self.assertAlmostEqual(best @ mean, goals[-1])
        rows = np.repeat(best[None, :], 51, axis=0)
        with self.assertRaises(CandidateError):  # right portfolio for the last target only
            fc.validate(rows, mean, goals, "Hang Seng")
        nine = best.copy()
        nine[top[-1]] = 0.0
        nine[top[0]] += 0.01
        for bad in (np.repeat(nine[None, :], 51, axis=0), rows[:50], rows * np.nan):
            with self.assertRaises(CandidateError):
                fc.validate(bad, mean, goals, "Hang Seng")

    @needs("cvxpy")
    def test_seed_scores_and_is_valid(self):
        namespace = {}
        exec(seed_source("science-finance-cardinality-constrained-frontier"), namespace)  # the package's seed
        score, details = fc.frontier_score(namespace["build_frontier"])
        self.assertAlmostEqual(score, 1.535, delta=0.01)
        self.assertEqual(set(details["mean_percentage_error_by_market"]), set(fc.MARKETS.values()))

    def test_last_target_has_a_unique_feasible_portfolio(self):
        mean, cov, uef_r, uef_s = fc.market(2)
        top = np.argsort(-mean)[:10]
        w = np.zeros(len(mean))
        w[top] = 0.01
        w[top[0]] = 0.91
        self.assertLess(abs(w @ mean - fc.targets(2)[-1]), 1e-15)
        self.assertGreater(fc.percentage_error(float(np.sqrt(w @ cov @ w)), float(w @ mean), uef_r, uef_s), 0.0)


if __name__ == "__main__":
    unittest.main()
