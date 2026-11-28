"""Aftershock forecasts: magnitude bins and count windows, negative-binomial scoring, the held-out protocol and the seed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

try:
    from scipy.stats import nbinom
except ImportError:  # optional: the test that compares with scipy is skipped
    nbinom = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_aftershocks as af  # noqa: E402
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_sandbox as es  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, needs, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
AFTERSHOCK_SEED = seed_source("science-earth-global-aftershock-forecast")

SPY = '''
import numpy as np


class Spy:
    def forecast(self, history, case):
        p = np.full(17, 1e-6)
        p[0] = 1 - 16e-6
        if history["t"].max() >= case["start"]:
            p[:] = 1.0  # sums to 17: rejected as invalid
        return p


def fit_aftershock_model(history, training_cases, info):
    assert history["t"].max() < info["test_start_day"]
    assert training_cases["start"].max() < info["test_start_day"]
    return Spy()
'''


def catalog(rows):
    return {f: np.array([r[i] for r in rows], float) for i, f in enumerate(af.FIELDS)}


class RuleTests(unittest.TestCase):
    def test_bins_and_radius(self):
        self.assertEqual([af.count_bin(n) for n in (0, 9, 10, 14, 15, 199, 200, 5000)], [0, 9, 10, 10, 11, 15, 16, 16])
        self.assertAlmostEqual(float(af.radius_km(6.0)), 50.0)
        self.assertAlmostEqual(float(af.radius_km(8.0)), 2 * 10 ** 2.2, places=6)

    def test_cases_count_window_and_skip_aftershocks_of_larger_events(self):
        t0 = af.FIRST_MAIN + 100
        rows = [(t0 - 30, 10.0, 20.0, 10.0, 7.0),      # larger event 30 days earlier, 0.2 deg away
                (t0, 10.2, 20.0, 10.0, 6.5),           # skipped: inside the M7 zone
                (t0 + 200, -30.0, -70.0, 20.0, 6.2),    # kept
                (t0 + 200.5, -30.1, -70.0, 50.0, 5.1),  # first day, not counted
                (t0 + 202, -30.2, -70.1, 50.0, 5.0),    # counted in the 7-day window
                (t0 + 203, -35.0, -70.0, 50.0, 5.5),    # outside 50 km
                (t0 + 220, -30.0, -70.1, 80.0, 5.4)]    # counted in the 30-day window
        out = af.cases(catalog(rows))
        mains = sorted(set(out["main_t"]))
        self.assertIn(t0 + 200, mains)
        self.assertNotIn(t0, mains)
        kept = out["main_t"] == t0 + 200
        self.assertEqual(sorted(out["count"][kept].tolist()), [1.0, 1.0])

    @unittest.skipUnless(nbinom, "scipy is not installed")
    def test_negative_binomial_matches_scipy(self):
        mean, k = 2.5, 0.4
        n = np.arange(30.0)
        expected = nbinom.logpmf(n, k, k / (k + mean))
        np.testing.assert_allclose(af.nb_log_pmf(n, mean, k), expected, rtol=1e-10)
        probs = af.nb_bins(mean, k)
        self.assertAlmostEqual(probs.sum(), 1.0)
        self.assertAlmostEqual(probs[0], float(np.exp(expected[0])), places=6)

    def test_validation(self):
        good = np.full(17, 1 / 17)
        np.testing.assert_allclose(af.validate(0, good), good)
        for bad in (np.full(16, 1 / 16), np.full(17, 0.1), np.r_[np.zeros(1), np.full(16, 1 / 16)]):
            with self.assertRaises(CandidateError):
                af.validate(0, bad)


@unittest.skipUnless((ec.data_dir() / af.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_no_event_at_or_after_a_forecast_start_is_visible(self):
        setup, steps, test = af.protocol(af.load())
        outputs = es.run_online(module_from(SPY), "fit_aftershock_model", setup, steps, af.validate)
        self.assertEqual(len(outputs), len(test["start"]))
        self.assertTrue(np.all(np.diff(test["start"]) >= 0))

    @needs("scipy")
    def test_seed(self):
        value, details = af.score(module_from(AFTERSHOCK_SEED))
        self.assertEqual(details["forecasts"], 1443)
        self.assertLess(value, details["reference_mean_log_score"])  # Poisson ignores inter-sequence variability


if __name__ == "__main__":
    unittest.main()
