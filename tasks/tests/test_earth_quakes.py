"""Southern California earthquake forecasts: catalogue derivation, target counts, the Poisson score and the seed."""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_quakes as q  # noqa: E402
from autoresearch_suite import earth_sandbox as es  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
QUAKE_SEED = seed_source("science-earth-socal-earthquake-forecast")

CSV = """time,latitude,longitude,depth,mag,magType,nst,gap,dmin,rms,net,id,updated,place,type
2016-01-01T00:00:00.000Z,33.0,-117.0,10.0,3.5,ml,,,,,ci,ci1,2020-01-01T00:00:00.000Z,x,earthquake
2016-01-01T12:00:00.000Z,33.05,-116.95,40.0,4.0,ml,,,,,ci,ci2,2020-01-01T00:00:00.000Z,x,earthquake
2016-01-01T12:00:00.000Z,33.05,-116.95,40.0,4.0,ml,,,,,ci,ci2,2020-01-01T00:00:00.000Z,x,earthquake
1990-06-01T06:00:00.000Z,34.0,-118.0,5.0,,ml,,,,,ci,ci3,2020-01-01T00:00:00.000Z,x,earthquake
"""

SPY = '''
import numpy as np


class Spy:
    def forecast(self, history, day):
        rates = np.full((50, 70), 1e-3)
        rates[0, 0] = 1.0 + (history["t"].max() >= day)  # 2.0 would reveal a look-ahead
        rates[0, 1] = len(history["t"]) * 1e-3
        return rates


def fit_quake_model(training, grid):
    return Spy()
'''


class CatalogueTests(unittest.TestCase):
    def test_derive_parses_deduplicates_and_skips_missing_magnitudes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for k, name in enumerate(q.RAW_SOURCES):
                path = Path(tmp) / name
                path.write_text(CSV if k == 0 else CSV.splitlines()[0] + "\n")
                paths[name] = path
            payload = q.derive(paths)[q.DATA]
        self.assertEqual(len(payload["t"]), 2)
        self.assertAlmostEqual(payload["t"][0], q.FIRST_DAY, places=6)
        self.assertAlmostEqual(payload["t"][1], q.FIRST_DAY + 0.5, places=6)

    def test_target_counts_apply_magnitude_depth_and_grid(self):
        catalog = {"t": np.array([q.FIRST_DAY + 0.2, q.FIRST_DAY + 0.3, q.FIRST_DAY + 1.5, q.FIRST_DAY - 1.0, q.FIRST_DAY + 2.0]),
                   "lat": np.array([32.05, 32.05, 36.95, 32.05, 31.5]), "lon": np.array([-120.95, -120.95, -114.05, -120.95, -117.0]),
                   "depth": np.array([5.0, 31.0, 10.0, 5.0, 5.0]), "mag": np.array([3.5, 5.0, 3.45, 6.0, 4.0])}
        counts = q.target_counts(catalog)
        self.assertEqual(counts.sum(), 2)
        self.assertEqual(counts[0, 0, 0], 1)  # the 31-km-deep event is not a target
        self.assertEqual(counts[1, 49, 69], 1)

    def test_poisson_loglik_matches_closed_form(self):
        rng = np.random.default_rng(1)
        rates = rng.uniform(0.01, 2.0, size=(4, 5))
        counts = rng.poisson(rates).astype(float)
        expected = sum(n * math.log(r) - r - math.lgamma(n + 1) for r, n in zip(rates.ravel(), counts.ravel()))
        self.assertAlmostEqual(q.poisson_loglik(rates, counts), expected, places=9)

    def test_validation(self):
        good = np.full((50, 70), 1e-3)
        np.testing.assert_array_equal(q.validate(0, good), good)
        for bad in (np.zeros((50, 70)), np.full((50, 69), 1e-3), np.full((50, 70), np.nan), np.full((50, 70), 10.0)):
            with self.assertRaises(CandidateError):
                q.validate(0, bad)


@unittest.skipUnless((ec.data_dir() / q.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = q.load()

    def test_no_event_at_or_after_the_forecast_day_is_visible(self):
        setup, steps = q.protocol(self.catalog)
        outputs = es.run_online(module_from(SPY), "fit_quake_model", setup, steps, q.validate)
        self.assertEqual(len(outputs), q.LAST_DAY - q.FIRST_DAY + 1)
        self.assertTrue(all(o[0, 0] == 1.0 for o in outputs))
        seen = np.array([round(o[0, 1] * 1000) for o in outputs])
        expected = np.array([np.sum(self.catalog["t"] < day) for day in range(q.FIRST_DAY, q.LAST_DAY + 1)])
        np.testing.assert_array_equal(seen, expected)

    def test_target_count_and_seed(self):
        self.assertEqual(int(q.target_counts(self.catalog).sum()), 1047)  # ComCat count service, same query
        value, details = q.score(module_from(QUAKE_SEED))
        self.assertGreater(value, 0.0)
        self.assertLess(value, 0.5)
        self.assertEqual(details["target_events"], 1047)


if __name__ == "__main__":
    unittest.main()
