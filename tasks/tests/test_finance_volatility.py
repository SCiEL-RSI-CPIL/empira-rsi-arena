"""Volatility forecasting: causal protocol, QLIKE scoring and candidate validation."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_stats, finance_volatility as fv  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import have_data  # noqa: E402


class Rolling:
    def __init__(self, training, names):
        self.window = list(training[-63:])

    def forecast(self):
        return np.mean(np.square(self.window), axis=0)

    def update(self, r):
        self.window = self.window[1:] + [r]


class Constant:
    def __init__(self, value):
        self.value = value

    def forecast(self):
        return self.value

    def update(self, r):
        pass


class VolatilityTests(unittest.TestCase):
    def test_qlike_is_minimized_by_the_true_variance(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.0, 0.01, 200_000)
        losses = [fv.qlike(np.full_like(r, h), r).mean() for h in (0.5e-4, 1e-4, 2e-4)]
        self.assertLess(losses[1], losses[0])
        self.assertLess(losses[1], losses[2])

    def test_diebold_mariano_sign(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(1.0, 1.0, 1000), rng.normal(1.2, 1.0, 1000)
        stat, p = finance_stats.diebold_mariano(a, b, lags=5)
        self.assertLess(stat, -3)
        self.assertLess(p, 0.01)

    @unittest.skipUnless(have_data("ind49_vw_daily.csv", "ff_factors_daily.csv"), "run scripts/prepare_finance.py first")
    def test_protocol_and_scores(self):
        days, names, returns = fv.load()
        self.assertEqual(returns.shape[1], 50)
        seen = []

        class Spy(Rolling):
            def __init__(self, training, names):
                super().__init__(training, names)
                seen.append(len(training))

            def update(self, r):
                seen.append(float(r[0]))
                super().update(r)
        score, details = fv.volatility_score(Spy)
        self.assertEqual(seen[0], 7280)
        split = 7280
        np.testing.assert_allclose(seen[1:6], returns[split:split + 5, 0])  # returns arrive in order, after forecasts
        self.assertAlmostEqual(score, -7.5627, delta=0.001)
        self.assertEqual((details["evaluation_days"], details["first_day"], details["last_day"]), (6475, "20000103", "20250930"))
        self.assertAlmostEqual(fv.volatility_score(fv._RiskMetrics)[0], details["riskmetrics_qlike"])
        for bad in (lambda t, n: Constant(np.zeros(50)), lambda t, n: Constant(np.full(49, 1e-4)),
                    lambda t, n: Constant(np.full(50, np.nan)), lambda t, n: Constant(np.full(50, 2.0)), lambda t, n: object()):
            with self.assertRaises(CandidateError):
                fv.volatility_score(bad)


if __name__ == "__main__":
    unittest.main()
