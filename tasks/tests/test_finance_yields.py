"""Yield-curve forecasting: data layout, information set, scoring and validation."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_yields as fy  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import have_data  # noqa: E402


def random_walk(history, horizons):
    return np.repeat(history["yields"][-1][None, :], len(horizons), axis=0)


@unittest.skipUnless(have_data("gsw_yields_monthly.csv"), "run scripts/prepare_finance.py first")
class YieldTests(unittest.TestCase):
    def test_monthly_table(self):
        months, yields = fy.load()
        self.assertEqual((int(months[0]), int(months[-1]), len(months)), (197201, 202507, 643))
        self.assertTrue((np.diff(months % 100) % 12 == 1).all())  # consecutive months
        self.assertTrue(((yields > -1) & (yields < 20)).all())

    def test_random_walk_scores_one_and_information_ends_at_origin(self):
        seen = []

        def spy(history, horizons):
            seen.append((int(history["yyyymm"][-1]), len(history["yields"])))
            return random_walk(history, horizons)
        score, details = fy.yield_score(spy)
        self.assertEqual(score, 1.0)
        self.assertEqual(seen[0][0], 199001)
        self.assertEqual(seen[-1][0], 202407)
        self.assertEqual(details["origins"], 415)
        self.assertTrue(all(b[1] == a[1] + 1 for a, b in zip(seen, seen[1:])))

    def test_perfect_and_invalid_forecasts(self):
        months, yields = fy.load()
        index = {int(m): t for t, m in enumerate(months)}

        def oracle(history, horizons):  # uses the future: only to check the arithmetic
            t = index[int(history["yyyymm"][-1])]
            return np.array([yields[t + h] for h in horizons])
        self.assertEqual(fy.yield_score(oracle)[0], 0.0)
        for bad in (lambda h, z: random_walk(h, z)[:2], lambda h, z: random_walk(h, z) * np.nan, lambda h, z: random_walk(h, z) + 50):
            with self.assertRaises(CandidateError):
                fy.yield_score(bad)


if __name__ == "__main__":
    unittest.main()
