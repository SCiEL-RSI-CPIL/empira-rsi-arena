"""Global temperature emulation: the energy-balance reference, hidden test temperatures and the seed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_gmst as gm  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
GMST_SEED = seed_source("science-earth-global-temperature-emulation")

PROBE = '''
import numpy as np


def predict_gmst(training, inputs):
    assert training["year"].max() == 1979 and len(training["gmst"]) == 130
    assert "gmst" not in inputs and set(inputs) == {"forcing_year", "forcing", "enso_year", "enso_month", "relative_nino34", "predict_years"}
    return np.zeros(len(inputs["predict_years"]))
'''


class ModelTests(unittest.TestCase):
    def test_ebm_reaches_equilibrium(self):
        response = gm.ebm(np.full(3000, 3.7), lam=1.2)
        self.assertAlmostEqual(response[-1], 3.7 / 1.2, places=3)
        self.assertLess(response[50], 3.7 / 1.2)  # transient response is below equilibrium


@unittest.skipUnless((ec.data_dir() / gm.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_inputs_hide_test_temperatures(self):
        value, details = gm.score(module_from(PROBE))
        data = gm.load()
        truth = data["gmst"][data["year"] >= 1980]
        self.assertAlmostEqual(value, float(np.sqrt(np.mean(truth ** 2))), places=9)

    def test_seed_and_references(self):
        value, details = gm.score(module_from(GMST_SEED))
        self.assertAlmostEqual(value, details["regression_on_total_forcing_rmse"], places=4)
        self.assertLess(details["two_layer_ebm_rmse"], value)
        self.assertGreater(details["observed_trend_per_decade"], 0.15)

    def test_relative_index_is_trendless_and_enso_like(self):
        data = gm.load()
        r, y, m = data["relative_nino34"], data["enso_year"], data["enso_month"]
        self.assertGreater(r[(y == 1998) & (m == 1)][0], 1.0)
        self.assertLess(r[(y == 2011) & (m == 1)][0], -1.0)
        self.assertLess(abs(np.polyfit(np.arange(len(r)) / 1200.0, r, 1)[0]), 0.2)

    def test_out_of_range_predictions_are_invalid(self):
        bad = "import numpy as np\n\ndef predict_gmst(training, inputs):\n    return np.full(45, 9.0)\n"
        with self.assertRaises(CandidateError):
            gm.score(module_from(bad))


if __name__ == "__main__":
    unittest.main()
