"""Portfolio choice against 1/N: backtest arithmetic, candidate validation and the DGU reference check."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_portfolio as fp  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import have_data  # noqa: E402

MONTHLY = ("ind10_vw_monthly.csv", "ff25_vw_monthly.csv", "ind49_vw_monthly.csv", "ff_factors_monthly.csv")


class BacktestArithmeticTests(unittest.TestCase):
    def test_net_returns_follow_dgu_wealth_equation(self):
        excess = np.array([[0.01, 0.03], [0.02, -0.01], [0.04, 0.00], [-0.02, 0.05]])
        rf = np.array([0.001, 0.001, 0.002, 0.002])
        months = ["200001", "200002", "200003", "200004"]
        policy = lambda h, c: np.array([3.0, 1.0])  # noqa: E731 - constant relative weights (0.75, 0.25)
        run = fp.backtest(policy, months, excess, rf, window=2, cost=0.01, max_gross=20)
        w = np.array([0.75, 0.25])
        port2 = rf[2] + w @ excess[2]
        drifted = w * (1 + excess[2] + rf[2]) / (1 + port2)
        trade3 = np.abs(w - drifted).sum()
        port3 = rf[3] + w @ excess[3]
        expected = [port2 - rf[2], (1 - 0.01 * trade3) * (1 + port3) - 1 - rf[3]]
        np.testing.assert_allclose(run["net"], expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(run["turnover"], [0.0, trade3])

    def test_relative_weights_follow_dgu_normalisation_and_limits(self):
        np.testing.assert_allclose(fp.relative_weights([2, -4], 2, 20), [1.0, -2.0])
        for bad in ([1, 2, 3], [np.nan, 1], [1, -1], [[1, 1]], ["a", "b"], [30.0, -29.0]):
            with self.assertRaises(CandidateError):
                fp.relative_weights(bad, 2, 20)

    def test_strategy_that_loses_all_wealth_is_invalid(self):
        excess = np.array([[0.01, 0.0], [0.0, 0.01], [0.0, 0.0], [-0.5, 0.6]])
        months, rf = ["1", "2", "3", "4"], np.zeros(4)
        with self.assertRaises(CandidateError):
            fp.backtest(lambda h, c: np.array([10.0, -9.0]), months, excess, rf, window=2, cost=0.0, max_gross=20)

    def test_jobson_korkie_memmel_is_antisymmetric(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(0.01, 0.04, 400), rng.normal(0.005, 0.05, 400)
        z_ab, p_ab = fp.jobson_korkie_memmel(a, b)
        z_ba, p_ba = fp.jobson_korkie_memmel(b, a)
        self.assertAlmostEqual(z_ab, -z_ba)
        self.assertAlmostEqual(p_ab, p_ba)
        self.assertEqual(fp.jobson_korkie_memmel(a, a)[0], 0.0)


@unittest.skipUnless(have_data(*MONTHLY), "run scripts/prepare_finance.py first")
class PublishedReferenceTests(unittest.TestCase):
    def gross_equal_weight(self, dataset):
        months, names, excess, rf = fp.load(dataset, "196307", "200411")
        run = fp.backtest(fp.equal_weight, months, excess, rf, window=120, cost=0.0, max_gross=20)
        return fp.sharpe(run["gross"]), float(np.mean(run["turnover"][1:])), len(names), len(run["gross"])

    def test_reproduces_dgu_table_3_and_5_for_one_over_n(self):
        # DeMiguel, Garlappi and Uppal (2009), Table 3 (1/N Sharpe) and Table 5 (1/N turnover), 1963-07 to 2004-11.
        sharpe, turnover, n, months = self.gross_equal_weight("industry10_market")
        self.assertEqual((n, months), (11, 377))
        self.assertAlmostEqual(sharpe, 0.1353, delta=0.0005)
        self.assertAlmostEqual(turnover, 0.0216, delta=0.0006)
        sharpe, turnover, n, months = self.gross_equal_weight("ff20_market")
        self.assertEqual(n, 21)
        self.assertAlmostEqual(sharpe, 0.1623, delta=0.0015)
        self.assertAlmostEqual(turnover, 0.0162, delta=0.0005)

    def test_seed_scores_and_invalid_candidates_fail(self):
        score, details = fp.portfolio_choice_score(fp.equal_weight, "industry10_market", start="196307", end="202509")
        self.assertEqual((details["months"], details["first_month"], details["last_month"]), (627, "197307", "202509"))
        self.assertAlmostEqual(score, details["equal_weight_net_sharpe"])
        self.assertAlmostEqual(details["one_sided_p_value_vs_equal_weight"], 0.5)
        with self.assertRaises(CandidateError):
            fp.portfolio_choice_score(lambda h, c: np.ones(3), "industry10_market", start="196307", end="202509")
        with self.assertRaises(CandidateError):
            fp.portfolio_choice_score(lambda h, c: np.r_[1.0, np.full(10, np.inf)], "industry10_market", start="196307", end="202509")

    def test_candidate_cannot_mutate_evaluation_data(self):
        def vandal(history, context):
            history[:] = 1.0  # writes only to the evaluator's copy
            return np.ones(history.shape[1])
        first, _ = fp.portfolio_choice_score(vandal, "industry49", start="196907", end="202509")
        second, _ = fp.portfolio_choice_score(fp.equal_weight, "industry49", start="196907", end="202509")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
