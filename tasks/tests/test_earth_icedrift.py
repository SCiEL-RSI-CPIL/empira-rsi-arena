"""Arctic sea-ice drift: geostrophic wind and position geometry, what each forecast day sees and the seed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_icedrift as ic  # noqa: E402
from autoresearch_suite import earth_sandbox as es  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
ICEDRIFT_SEED = seed_source("science-earth-arctic-sea-ice-drift")

SPY = '''
import numpy as np


class Spy:
    def forecast(self, history, today):
        late = float(history["day"].max() >= today["day"].min())
        return np.full((len(today["buoy"]), 2), late)


def fit_drift_model(training):
    return Spy()
'''


class GeometryTests(unittest.TestCase):
    def test_geostrophic_wind_of_a_uniform_gradient(self):
        lat0, lon0 = 80.0, 30.0
        glat, glon = np.meshgrid(np.arange(60.0, 90.1, 2.5), np.arange(-180.0, 180.0, 2.5), indexing="ij")
        glat, glon = glat.ravel(), glon.ravel()
        x, y = ic.local_xy(lat0, lon0, glat, glon)
        field = 101000.0 + 0.001 * y  # pressure rising northward by 1 hPa per 100 km
        slp, ug, vg = ic.geostrophic(field, glat, glon, lat0, lon0)
        f = 2 * ic.OMEGA * np.sin(np.radians(lat0))
        self.assertAlmostEqual(ug, -0.001 / (ic.RHO_AIR * f), places=2)  # westward wind
        self.assertAlmostEqual(vg, 0.0, places=2)
        self.assertAlmostEqual(slp, 1010.0, places=3)

    def test_positions_parse_00_utc_rows(self):
        text = "2014 12 0 00 5300 79.1 -179.3\n2015 1 1 00 5300 78.8 -177.3\n2015 1 1 12 5300 78.9 -177.0\n"
        pos = ic.positions(text)
        self.assertEqual(list(pos.values()), [(78.8, -177.3)])

    def test_local_xy_east_and_north(self):
        x, y = ic.local_xy(70.0, 0.0, np.array([70.0, 71.0]), np.array([1.0, 0.0]))
        self.assertAlmostEqual(x[0], 6.371e6 * np.cos(np.radians(70)) * np.sin(np.radians(1)), places=0)
        self.assertGreater(y[1], 1.1e5)


@unittest.skipUnless((ec.data_dir() / ic.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_history_ends_before_each_forecast_day(self):
        data = ic.load()
        setup, steps, targets = ic.protocol(data)
        outputs = es.run_online(module_from(SPY), "fit_drift_model", setup, steps, lambda i, v: np.asarray(v))
        self.assertTrue(all(np.all(o == 0.0) for o in outputs))
        self.assertEqual(sum(len(t) for t in targets), 356683)

    def test_seed_equals_reference(self):
        value, details = ic.score(module_from(ICEDRIFT_SEED))
        self.assertAlmostEqual(value, details["thorndike_colony_reference_rmse"], places=3)
        self.assertLess(value, details["zero_drift_rmse"])


if __name__ == "__main__":
    unittest.main()
