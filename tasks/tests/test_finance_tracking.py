"""Index tracking: price layout, trading arithmetic, information set, validation and scoring."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_tracking as ft  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import have_data, needs, seed_source  # noqa: E402

TABLES = [f"orlib_indtrack{i}.csv" for i in range(1, 6)]


def first_ten(state):
    if state["week"] != 145:
        return None
    w = np.zeros(state["stock_prices"].shape[1])
    w[:10] = 0.1
    return w


class TradingTests(unittest.TestCase):
    def test_costs_solve_the_budget_equation(self):
        prices = np.array([10.0, 20.0, 5.0])
        shares = np.array([1.0, 0.5, 2.0])
        value = float(shares @ prices)
        weights = np.array([0.0, 0.5, 0.5])
        new, after, traded = ft.trade(shares, prices, weights, value, True)
        self.assertAlmostEqual(after, value - ft.COST * traded, places=12)
        self.assertAlmostEqual(float(new @ prices), after, places=12)
        self.assertEqual(ft.trade(shares, prices, weights, value, False)[1], value)


@unittest.skipUnless(have_data(*TABLES), "run scripts/prepare_finance.py first")
class TrackingTests(unittest.TestCase):
    def test_buy_and_hold_returns_and_information_set(self):
        level, prices = ft.market(1)
        weeks = []

        def spy(state):
            weeks.append(state["week"])
            self.assertEqual(len(state["index_prices"]), state["week"] + 1)
            return first_ten(state)
        r, b, traded = ft.track(spy, 1)
        self.assertEqual(weeks, list(range(145, 290, 13)))
        shares = 0.1 / prices[145, :10]
        values = prices[145:, :10] @ shares
        np.testing.assert_allclose(r, np.diff(np.log(values)), atol=1e-13)
        np.testing.assert_allclose(b, np.diff(np.log(level[145:])), atol=1e-15)
        self.assertEqual(traded, 0.0)

    def test_rebalancing_costs_enter_returns(self):
        def churn(state):
            w = np.zeros(state["stock_prices"].shape[1])
            w[(state["week"] // 13) % 2 * 10:(state["week"] // 13) % 2 * 10 + 10] = 0.1
            return w
        r, b, traded = ft.track(churn, 2)
        self.assertGreater(traded, 1.0)
        score, details = ft.tracking_score(first_ten)
        self.assertEqual(details["weeks"], 145)

    def test_validation(self):
        n = ft.market(1)[1].shape[1]
        good = np.zeros(n)
        good[:10] = 0.1
        ft.validate(good, n, 145)
        eleven = np.zeros(n)
        eleven[:11] = 1 / 11
        tiny = good.copy()
        tiny[0], tiny[1] = 0.195, 0.005
        for bad in (eleven, tiny, -good, good * 2, good[:-1], good * np.nan, np.zeros(n)):
            with self.assertRaises(CandidateError):
                ft.validate(bad, n, 145)
        with self.assertRaises(CandidateError):
            ft.tracking_score(lambda state: None)

    @needs("scipy")
    def test_seed_score(self):
        namespace = {}
        exec(seed_source("science-finance-index-tracking"), namespace)  # the package's seed
        score, details = ft.tracking_score(namespace["rebalance"])
        self.assertAlmostEqual(score, 1.087, delta=0.002)
        self.assertEqual(len(details["markets"]), 5)


if __name__ == "__main__":
    unittest.main()
