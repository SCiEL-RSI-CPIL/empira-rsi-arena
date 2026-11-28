"""ENSO forecasts: the Nino 3.4 box mean, output validation, what each forecast origin sees and the seed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_enso as enso  # noqa: E402
from autoresearch_suite import earth_sandbox as es  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
ENSO_SEED = seed_source("science-earth-enso-nino34-forecast")

SPY = '''
import numpy as np


class Spy:
    def forecast(self, history):
        # encode how many months the candidate can see and the last one it saw
        n = len(history["year"])
        return np.full(12, 10.0 + n / 100.0) + np.r_[0.0, np.zeros(10), (history["year"][-1] - 1950) / 1000.0]


def fit_enso_model(training):
    return Spy()
'''


class IndexTests(unittest.TestCase):
    def test_box_mean_weights_by_cosine_latitude_and_skips_land(self):
        lat = np.array([-4.0, 0.0, 4.0, 8.0])
        lon = np.array([190.0, 200.0, 250.0])
        sst = np.zeros((1, 4, 3))
        sst[0, :, 0] = [1.0, 2.0, 3.0, 99.0]
        sst[0, :, 1] = [1.0, np.nan, 3.0, 99.0]
        sst[0, :, 2] = 99.0  # outside the box
        w = np.cos(np.radians([-4.0, 0.0, 4.0]))
        expected = (w[0] * 1 + w[1] * 2 + w[2] * 3 + w[0] * 1 + w[2] * 3) / (2 * w[0] + w[1] + 2 * w[2])
        value = enso.box_mean(sst, lat, lon, ((-5.0, 5.0), (190.0, 240.0)))
        self.assertAlmostEqual(float(value[0]), expected, places=12)

    def test_validation_rejects_bad_forecasts(self):
        np.testing.assert_array_equal(enso.validate(0, np.full(12, 27.0)), np.full(12, 27.0))
        for bad in (np.full(11, 27.0), np.full(12, np.nan), np.full(12, 45.0), "warm"):
            with self.assertRaises(CandidateError):
                enso.validate(0, bad)


@unittest.skipUnless((ec.data_dir() / enso.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = enso.load()

    def test_each_origin_sees_exactly_the_months_up_to_it(self):
        setup, steps, origins = enso.protocol(self.data)
        spy = module_from(SPY)
        outputs = es.run_online(spy, "fit_enso_model", setup, steps, enso.validate)
        seen = [round((o[0] - 10.0) * 100) for o in outputs]
        self.assertEqual(seen, [t + 1 for t in origins])
        self.assertEqual(origins[0], enso.month_index(2000, 1))
        self.assertEqual(len(origins), 300)
        last_year = [round(o[-1] * 1000 - o[0] * 1000) + 1950 for o in outputs]
        self.assertEqual(last_year[-1], 2024)

    def test_seed_equals_damped_persistence_reference(self):
        value, details = enso.score(module_from(ENSO_SEED))
        self.assertAlmostEqual(value, details["damped_persistence_mean_rmse"], places=3)
        self.assertLess(value, details["climatology_mean_rmse"])
        self.assertLess(details["rmse_by_lead"][0], details["rmse_by_lead"][-1])

    def test_nino34_matches_known_oni_peaks(self):
        n34, month, year = self.data["nino34"], self.data["month"], self.data["year"]
        base = (year >= 1981) & (year <= 2010)
        clim = np.array([n34[base & (month == m)].mean() for m in range(1, 13)])
        i = enso.month_index(1997, 12)
        self.assertAlmostEqual(float((n34 - clim[month - 1])[i - 1:i + 2].mean()), 2.4, delta=0.1)


if __name__ == "__main__":
    unittest.main()
