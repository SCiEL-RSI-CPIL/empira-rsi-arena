"""Temperature homogenisation: the network generator, the centred error measure and the seed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_homog as hg  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
HOMOG_SEED = seed_source("science-earth-temperature-homogenization")

IDENTITY = "def homogenize(networks, calendar):\n    return [n.copy() for n in networks]\n"


class GeneratorTests(unittest.TestCase):
    def test_network_has_inhomogeneities_and_gaps(self):
        truth, observed = hg.network(np.random.default_rng(1))
        self.assertEqual(truth.shape, (hg.STATIONS, hg.MONTHS))
        self.assertTrue(np.isnan(observed).any())
        self.assertTrue(np.all(np.isfinite(truth)))
        diff = observed - truth
        self.assertGreater(np.nanstd(diff), 0.1)
        anomalies = truth - truth.mean(axis=1, keepdims=True)
        correlation = np.corrcoef(anomalies)
        self.assertGreater(np.median(correlation[np.triu_indices(hg.STATIONS, 1)]), 0.8)  # annual cycle dominates

    def test_crmsd_ignores_constant_offsets(self):
        a = np.random.default_rng(0).normal(size=(3, 50))
        np.testing.assert_allclose(hg.crmsd(a + 5.0, a), 0.0, atol=1e-12)
        self.assertAlmostEqual(float(hg.crmsd(np.r_[np.zeros(25), np.ones(25)][None], np.zeros((1, 50)))[0]), 0.5)

    def test_annual_means_need_ten_months(self):
        series = np.arange(24, dtype=float)[None]
        mask = np.ones((1, 24), bool)
        mask[0, 12:15] = False
        years = hg.YEARS
        try:
            hg.YEARS = 2
            out = hg.annual(series, mask)
        finally:
            hg.YEARS = years
        self.assertAlmostEqual(float(out[0, 0]), 5.5)
        self.assertTrue(np.isnan(out[0, 1]))


@unittest.skipUnless((ec.data_dir() / hg.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_identity_scores_the_raw_error_and_seed_improves(self):
        value, details = hg.score(module_from(IDENTITY))
        self.assertAlmostEqual(value, details["raw_annual_crmsd"], places=4)
        seed, _ = hg.score(module_from(HOMOG_SEED))
        self.assertLess(seed, 0.6 * value)

    def test_invalid_output(self):
        with self.assertRaises(CandidateError):
            hg.score(module_from("def homogenize(networks, calendar):\n    return networks[:3]\n"))
        with self.assertRaises(CandidateError):
            hg.score(module_from("import numpy as np\n\ndef homogenize(networks, calendar):\n    return [np.full(n.shape, np.nan) for n in networks]\n"))


if __name__ == "__main__":
    unittest.main()
