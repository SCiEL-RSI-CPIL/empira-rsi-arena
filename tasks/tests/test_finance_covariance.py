"""Covariance estimation for minimum-variance portfolios: validation, backtest arithmetic and scoring."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_covariance as fc  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import have_data  # noqa: E402


def identity_shrinkage(window):
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity."""
    x = window - window.mean(axis=0)
    t, n = x.shape
    s = x.T @ x / t
    mu = np.trace(s) / n
    d2 = np.linalg.norm(s - mu * np.eye(n), "fro") ** 2
    b2 = min(sum(np.linalg.norm(np.outer(r, r) - s, "fro") ** 2 for r in x) / t ** 2, d2)
    return (b2 / d2) * mu * np.eye(n) + (1 - b2 / d2) * s


class CovarianceTests(unittest.TestCase):
    def test_validation(self):
        good = np.diag([1.0, 2.0, 3.0])
        np.testing.assert_allclose(fc.validate(good, 3), good)
        asym = good.copy()
        asym[0, 1] = 0.1
        singular = np.ones((3, 3))
        for bad in (good[:2], asym, singular, good * np.nan, -good, "x", np.diag([1.0, 1e-14, 1.0])):
            with self.assertRaises(CandidateError):
                fc.validate(bad, 3)

    def test_gmv_weights_minimize_variance(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(6, 6))
        sigma = a @ a.T + np.eye(6)
        w = fc.gmv_weights(sigma)
        self.assertAlmostEqual(w.sum(), 1.0)
        for _ in range(20):
            other = w + rng.normal(scale=0.1, size=6)
            other += (1 - other.sum()) / 6
            self.assertGreaterEqual(other @ sigma @ other, w @ sigma @ w)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
    def test_identity_shrinkage_matches_scikit_learn(self):
        from sklearn.covariance import LedoitWolf
        window = np.random.default_rng(5).normal(size=(126, 49)) * 0.01
        np.testing.assert_allclose(identity_shrinkage(window), LedoitWolf().fit(window).covariance_, atol=1e-18)

    @unittest.skipUnless(have_data("ind49_vw_daily.csv"), "run scripts/prepare_finance.py first")
    def test_scores_on_industry_returns(self):
        score, details = fc.covariance_score(fc.sample_covariance)
        self.assertAlmostEqual(score, 11.974, delta=0.002)
        self.assertEqual((details["rebalances"], details["days"]), (649, 13629))
        self.assertAlmostEqual(details["relative_to_sample_covariance"], 1.0)
        shrunk, _ = fc.covariance_score(identity_shrinkage)
        self.assertLess(shrunk, score - 1.0)
        with self.assertRaises(CandidateError):
            fc.covariance_score(lambda w: np.zeros((49, 49)))

        def vandal(window):
            window[:] = 0.0
            return np.cov(np.random.default_rng(0).normal(size=(126, 49)), rowvar=False)
        fc.covariance_score(vandal)
        self.assertAlmostEqual(fc.covariance_score(fc.sample_covariance)[0], score)


if __name__ == "__main__":
    unittest.main()
