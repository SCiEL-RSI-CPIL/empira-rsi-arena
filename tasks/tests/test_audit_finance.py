"""Finance and economics evaluators: regressions of the evaluator audit.

Covers the corrected behaviour (execution schedules scored at the order size, scale-free portfolio positions,
typed output validation, the ignored return value of a volatility model's update) and two facts the statements
rely on (the FRED-MD vintage and the proven kidney optima, re-derived by linear-programming duality).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "suite/_lib")]
from autoresearch_suite import (finance_data, finance_execution, finance_kidney, finance_macro, finance_portfolio,
                                finance_stats, finance_volatility, runtime)
from autoresearch_suite.runtime import CandidateError

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
HAVE_CVXPY = importlib.util.find_spec("cvxpy") is not None
# SHA-256 of the official FRED-MD file 2024-09.csv (St. Louis Fed, monthly vintages; data to 2024-08).
FRED_MD_2024_09_SHA256 = "b7b33919fef3dad5ba02cc331006477d6d631ff448c04817866f16e23a6bf9a8"
# A valid forecaster: what update() returns is not part of the contract.
RETURNS_ITSELF = '''
import numpy as np


class Ewma:
    def __init__(self, training):
        self.h = np.mean(training ** 2, axis=0)

    def forecast(self):
        return self.h

    def update(self, returns):
        self.h = 0.9 * self.h + 0.1 * returns ** 2
        return self


def make_forecaster(training_returns, names):
    return Ewma(training_returns)
'''


def have_data(*names):
    """True when the prepared tables exist (python3 scripts/prepare_finance.py)."""
    return all(finance_data.path(name).is_file() for name in names)


class OutputValidatorTests(unittest.TestCase):
    def test_text_and_booleans_are_not_numbers(self):
        for bad in (["1", "2", "3"], [b"1", b"2", b"3"], [True, False, True], np.ones(3, dtype=bool), "123"):
            with self.subTest(value=bad), self.assertRaisesRegex(CandidateError, "numeric|shape"):
                finance_stats.finite_vector(bad, "forecast", (3,))

    def test_an_integer_beyond_the_float_range_is_the_candidates_error(self):
        with self.assertRaisesRegex(CandidateError, "real number"):
            finance_stats.finite_scalar(10 ** 400, "forecast", bound=0.5)
        with self.assertRaisesRegex(CandidateError, "numeric"):
            finance_stats.finite_vector([10 ** 400, 1, 2], "forecast", (3,))

    def test_numbers_in_any_container_are_still_accepted(self):
        for good in ([1, 2, 3], (1.0, 2.5, 3), np.arange(3, dtype=np.int8), np.ones(3, dtype=np.float32), [np.float64(1), 2, 3.0]):
            with self.subTest(value=good):
                self.assertEqual(finance_stats.finite_vector(good, "forecast", (3,)).dtype, np.float64)
        self.assertEqual(finance_stats.finite_scalar(np.float32(0.25), "forecast", bound=0.5), 0.25)
        self.assertEqual(finance_stats.finite_scalar(7, "forecast"), 7.0)


class ExecutionToleranceTests(unittest.TestCase):
    def problems(self):
        return [finance_execution.instance(spec) for spec in finance_execution.INSTANCES]

    def test_an_accepted_schedule_is_scored_at_the_order_size(self):
        for problem in self.problems():
            X, N = problem["shares"], problem["periods"]
            for factor in (1.0 - 0.9e-6, 1.0 + 0.9e-6):
                scored = finance_execution.validate(problem, np.full(N, X / N) * factor)
                self.assertAlmostEqual(float(scored.sum()) / X, 1.0, places=14)
                np.testing.assert_allclose(scored, X / N, rtol=1e-13)
            dust = np.r_[X + 1e-4, -1e-4, np.zeros(N - 2)]  # a negative entry inside the tolerance
            scored = finance_execution.validate(problem, dust)
            self.assertTrue((scored >= 0).all())
            self.assertAlmostEqual(float(scored.sum()) / X, 1.0, places=14)

    def test_an_exact_schedule_is_not_changed(self):
        for problem in self.problems():
            X, N = problem["shares"], problem["periods"]
            trades = np.linspace(2.0, 1.0, N)
            trades *= X / trades.sum()
            if float(trades.sum()) == X:
                np.testing.assert_array_equal(finance_execution.validate(problem, trades), trades)

    def test_selling_less_than_the_order_does_not_beat_the_optimum(self):
        for index in range(6):  # the six closed-form instances: no solver needed
            problem = finance_execution.instance(finance_execution.INSTANCES[index])
            trades, _ = finance_execution.almgren_chriss(problem)
            best = finance_execution.objective(problem, trades)[0]
            under = finance_execution.validate(problem, trades * (1.0 - 0.999e-6))
            self.assertGreaterEqual(finance_execution.objective(problem, under)[0], best * (1.0 - 1e-12))

    def test_violations_beyond_the_tolerances_are_still_rejected(self):
        problem = self.problems()[0]
        X, N = problem["shares"], problem["periods"]
        for bad in (np.full(N, X / N) * (1.0 - 1.1e-6), np.full(N, X / N) * 1.01, np.r_[X + 1.0, -1.0, np.zeros(N - 2)],
                    np.full(N - 1, X / (N - 1)), np.full(N, np.nan), [True] * N):
            with self.subTest(value=bad), self.assertRaises(CandidateError):
                finance_execution.validate(problem, bad)

    @unittest.skipUnless(HAVE_CVXPY, "the reference optimum of the non-linear instances needs cvxpy")
    def test_tolerance_slack_scores_like_the_schedule_itself(self):
        def near_optimal(problem):
            index = [spec["name"] for spec in finance_execution.INSTANCES].index(problem["name"])
            if finance_execution.optimum(index)[1] == "closed_form":
                return finance_execution.almgren_chriss(problem)[0]
            return np.full(problem["periods"], problem["shares"] / problem["periods"])
        exact = finance_execution.execution_score(near_optimal)[0]
        slack = finance_execution.execution_score(lambda problem: near_optimal(problem) * (1.0 - 0.999e-6))[0]
        self.assertAlmostEqual(slack, exact, delta=1e-8)


class PortfolioScaleTests(unittest.TestCase):
    def test_a_common_scale_does_not_make_positions_invalid(self):
        for scale in (1e-200, 1e-30, 1e-12, 1e-6, 1.0, 1e200):
            with self.subTest(scale=scale):
                weights = finance_portfolio.relative_weights(np.full(11, scale), 11, 20.0)
                np.testing.assert_allclose(weights, 1.0 / 11, rtol=1e-14)
        weights = finance_portfolio.relative_weights(1e-15 * np.array([3.0, -1.0]), 2, 20.0)
        np.testing.assert_allclose(weights, [1.5, -0.5], rtol=1e-14)

    def test_cancelling_positions_are_still_rejected(self):
        for bad in (np.zeros(11), np.r_[1.0, -1.0, np.zeros(9)], np.r_[1.0, -1.0 + 1e-12, np.zeros(9)],
                    1e-20 * np.r_[1.0, -1.0, np.zeros(9)]):
            with self.subTest(value=bad), self.assertRaisesRegex(CandidateError, "zero"):
                finance_portfolio.relative_weights(bad, 11, 20.0)
        with self.assertRaisesRegex(CandidateError, "gross exposure"):
            finance_portfolio.relative_weights(np.r_[10.6, -9.6, np.zeros(9)], 11, 20.0)


class VolatilityUpdateValueTests(unittest.TestCase):
    def test_what_update_returns_does_not_decide_validity(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "candidate.py"
        path.write_text(RETURNS_ITSELF)
        module = runtime.load_program(path)
        rng = np.random.default_rng(3)
        returns = 0.01 * rng.standard_normal((60, 4))
        days, names = [f"day{t:02d}" for t in range(60)], ["a", "b", "c", "d"]
        trusted = finance_volatility.run(module.make_forecaster, days, names, returns, 40)
        sandboxed = finance_volatility.run(module, days, names, returns, 40)
        self.assertEqual(sandboxed.shape, (20, 4))
        np.testing.assert_array_equal(sandboxed, trusted)


class FredMdVintageTests(unittest.TestCase):
    def test_the_pinned_file_is_the_2024_09_vintage(self):
        lock = json.loads((ROOT / "catalog/finance-data-lock.json").read_text())
        source = next(item for item in lock["sources"] if item["id"] == "fred-md-2024-10")
        self.assertEqual(source["sha256"], FRED_MD_2024_09_SHA256)

    @unittest.skipUnless(have_data(finance_macro.TABLE), "run scripts/prepare_finance.py first")
    def test_the_panel_ends_in_august_2024(self):
        months, names, tcodes, panel = finance_macro.load()
        self.assertEqual((int(months[0]), int(months[-1]), len(names)), (195901, 202408, 126))
        self.assertEqual(int(months[[t for t, m in enumerate(months) if m <= finance_macro.LAST_ORIGIN][-1] + 12]), 202408)


@unittest.skipUnless(HAVE_SCIPY and have_data(finance_kidney.TABLE), "needs scipy and scripts/prepare_finance.py")
class KidneyOptimaTests(unittest.TestCase):
    """The recorded optimum equals the linear-programming bound of the cycle packing, and a clearing attains it."""

    def packing(self, name):
        from scipy.sparse import coo_matrix
        pairs, edges = finance_kidney.pool(name)
        adjacent = np.zeros((pairs, pairs), dtype=bool)
        adjacent[edges[:, 0], edges[:, 1]] = True
        cycles = [(int(a), int(b)) for a, b in zip(*np.nonzero(np.triu(adjacent & adjacent.T, 1)))]
        for a in range(pairs):  # three-cycles a -> b -> c -> a with a the smallest index
            after, before = np.nonzero(adjacent[a])[0], np.nonzero(adjacent[:, a])[0]
            after, before = after[after > a], before[before > a]
            if len(after) and len(before):
                cycles += [(a, int(after[i]), int(before[j])) for i, j in zip(*np.nonzero(adjacent[np.ix_(after, before)]))
                           if after[i] != before[j]]
        rows = [v for cycle in cycles for v in cycle]
        cols = [j for j, cycle in enumerate(cycles) for _ in cycle]
        matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(pairs, len(cycles))).tocsc()
        return cycles, matrix, np.array([len(cycle) for cycle in cycles], dtype=float)

    def test_small_pools_by_duality(self):
        from scipy.optimize import Bounds, LinearConstraint, linprog, milp
        for name in ("00036-00000153", "00036-00000154"):
            with self.subTest(pool=name):
                cycles, matrix, size = self.packing(name)
                relaxed = linprog(-size, A_ub=matrix, b_ub=np.ones(matrix.shape[0]), bounds=(0, 1), method="highs-ds")
                self.assertEqual(int(np.floor(-relaxed.fun + 1e-6)), finance_kidney.OPTIMA[name])
                # A clearing that attains the bound uses only columns of zero reduced cost (complementary slackness).
                reduced = size + matrix.T @ relaxed.ineqlin.marginals
                keep = np.nonzero(reduced > -1e-7)[0]
                result = milp(-size[keep], constraints=LinearConstraint(matrix[:, keep], 0, 1), integrality=np.ones(len(keep)),
                              bounds=Bounds(0, 1), options={"mip_rel_gap": 0})
                chosen = [list(cycles[keep[j]]) for j in np.nonzero(result.x > 0.5)[0]]
                self.assertEqual(finance_kidney.check_clearing(name, chosen)[0], finance_kidney.OPTIMA[name])


if __name__ == "__main__":
    unittest.main()
