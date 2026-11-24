"""Gutenberg-Richter b-value: the Aki-Utsu estimator, the catalogue generator and the seed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_mc as mc  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, needs, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
MC_SEED = seed_source("science-earth-earthquake-b-value-completeness")


class EstimatorTests(unittest.TestCase):
    def test_aki_utsu_recovers_b_on_a_complete_binned_catalogue(self):
        rng = np.random.default_rng(0)
        m = np.round(np.round((1.95 + rng.exponential(1 / (1.1 * np.log(10)), 200000)) / 0.1) * 0.1, 1)
        self.assertAlmostEqual(mc.aki_utsu(m, 2.0), 1.1, delta=0.02)
        self.assertEqual(mc.max_curvature(m), 2.0)

    @needs("scipy")
    def test_generator_parameters_are_in_range(self):
        b, mc95, m = mc.catalogue(np.random.default_rng(5))
        self.assertTrue(0.6 <= b <= 1.4)
        self.assertTrue(150 <= len(m) <= 5000)
        np.testing.assert_allclose(m * 10, np.round(m * 10), atol=1e-9)


@unittest.skipUnless((ec.data_dir() / mc.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_seed_matches_its_reference(self):
        value, details = mc.score(module_from(MC_SEED))
        self.assertAlmostEqual(value, details["maxc_plus_0.2_aki_utsu_b_mae"], places=4)
        self.assertLess(details["true_mc_aki_utsu_b_mae"], value)

    def test_invalid_output(self):
        with self.assertRaises(CandidateError):
            mc.score(module_from("import numpy as np\n\ndef estimate_b_and_mc(catalogues):\n    return np.full((len(catalogues), 2), 5.0)\n"))


if __name__ == "__main__":
    unittest.main()
