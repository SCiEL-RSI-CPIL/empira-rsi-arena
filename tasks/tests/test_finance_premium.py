"""Equity-premium prediction: forecast statistics, information sets and candidate validation."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_premium as fpm, finance_stats  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import have_data  # noqa: E402


class StatisticsTests(unittest.TestCase):
    def test_oos_r2_and_clark_west_on_known_cases(self):
        rng = np.random.default_rng(1)
        actual = rng.normal(0.005, 0.04, 500)
        bench = np.full(500, 0.005)
        self.assertEqual(finance_stats.oos_r2(actual, bench, bench), 0.0)
        self.assertEqual(finance_stats.oos_r2(actual, actual, bench), 100.0)
        self.assertEqual(finance_stats.clark_west(actual, bench, bench), (0.0, 0.5))
        signal = rng.normal(0, 0.01, 500)
        target = 0.005 + signal + rng.normal(0, 0.01, 500)
        stat, p = finance_stats.clark_west(target, 0.005 + signal, bench)
        self.assertGreater(stat, 5)
        self.assertLess(p, 1e-6)
        manual = 100 * (1 - np.sum((target - 0.005 - signal) ** 2) / np.sum((target - bench) ** 2))
        self.assertAlmostEqual(finance_stats.oos_r2(target, 0.005 + signal, bench), manual)

    def test_scalar_validation(self):
        self.assertEqual(finance_stats.finite_scalar(np.float32(0.25), "x", bound=0.5), 0.25)
        for bad in (np.nan, np.inf, "0.1", None, True, 0.7, [0.1]):
            with self.assertRaises(CandidateError):
                finance_stats.finite_scalar(bad, "x", bound=0.5)


@unittest.skipUnless(have_data("goyal_welch_monthly.csv"), "run scripts/prepare_finance.py first")
class PremiumTaskTests(unittest.TestCase):
    def test_information_set_ends_at_origin_and_withholds_inflation(self):
        months, data, premium = fpm.load()
        t = int(np.where(months == 196412)[0][0])
        view = fpm.history(months, data, premium, t)
        self.assertEqual(int(view["yyyymm"][-1]), 196412)
        self.assertTrue(all(len(v) == t + 1 for v in view.values()))
        self.assertTrue(np.isnan(view["infl"][-1]))
        self.assertTrue(np.isfinite(view["infl"][-2]))
        view["equity_premium"][:] = 0.0
        self.assertNotEqual(float(premium[t]), 0.0)

    def test_prevailing_mean_seed_scores_zero(self):
        score, details = fpm.premium_score(fpm.prevailing_mean_forecast)
        self.assertEqual(score, 0.0)
        self.assertEqual((details["forecasts"], details["first_target_month"], details["last_target_month"]), (732, 196501, 202512))

    def test_candidate_sees_each_origin_once_in_order(self):
        seen = []

        def spy(view):
            seen.append(int(view["yyyymm"][-1]))
            return 0.004
        fpm.premium_score(spy)
        self.assertEqual(seen[0], 196412)
        self.assertEqual(seen[-1], 202511)
        self.assertEqual(len(seen), 732)
        self.assertTrue(all(b > a for a, b in zip(seen, seen[1:])))

    def test_invalid_forecasts_are_rejected(self):
        for bad in (lambda v: float("nan"), lambda v: 0.9, lambda v: "high"):
            with self.assertRaises(CandidateError):
                fpm.premium_score(bad)
        with self.assertRaises(CandidateError):
            fpm.premium_score(None)


if __name__ == "__main__":
    unittest.main()
