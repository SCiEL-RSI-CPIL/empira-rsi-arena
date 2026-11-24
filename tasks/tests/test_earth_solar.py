"""Diffuse solar irradiance: station-file parsing, closure of the targets, extraterrestrial irradiance and the seed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_solar as so  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
SOLAR_SEED = seed_source("science-earth-solar-irradiance-separation")

FILE = """ Bondville
   40.05  -88.37  213 m version 1
 2021   1  1  1 14 30 14.500  60.00   400.0 0     0.0 0   700.0 0   50.0 0   290.0 0   271.78 0   271.74 0   307.2 0   271.09 0   271.07 0     0.0 0     0.0 0     0.0 0   -17.1 0   -17.1 0    -2.2 0    84.7 0     4.2 0    91.4 0  1000.1 0
 2021   1  1  1 14 31 14.517  60.00   400.0 0     0.0 0   700.0 2   50.0 0   290.0 0   271.78 0   271.74 0   307.2 0   271.09 0   271.07 0     0.0 0     0.0 0     0.0 0   -17.1 0   -17.1 0    -2.2 0    84.7 0     4.2 0    91.4 0  1000.1 0
"""

PROBE = '''
import numpy as np


def estimate_diffuse(train, test):
    assert all("dhi" not in s and "dni" not in s for s in test.values())
    return {k: 0.5 * s["ghi"][s["target_index"].astype(int)] for k, s in test.items()}
'''


class UnitTests(unittest.TestCase):
    def test_parse_masks_flagged_values(self):
        lat, lon, elevation, rows = so.parse(FILE)
        self.assertEqual((lat, lon, elevation), (40.05, -88.37, 213.0))
        self.assertEqual(rows[0]["minute"], 14 * 60 + 30)
        self.assertEqual((rows[0]["ghi"], rows[0]["dni"], rows[0]["dhi"], rows[0]["temp"], rows[0]["pres"]), (400.0, 700.0, 50.0, -2.2, 1000.1))
        self.assertTrue(np.isnan(rows[1]["dni"]))

    def test_targets_require_closure(self):
        station = {"ghi": np.array([400.0, 400.0, 5.0]), "dni": np.array([700.0, 300.0, 1.0]), "dhi": np.array([50.0, 50.0, 4.0]),
                   "zen": np.array([60.0, 60.0, 60.0])}
        np.testing.assert_array_equal(so.targets(station), [0])  # closure fails for the second, GHI too low for the third

    def test_extraterrestrial(self):
        self.assertAlmostEqual(float(so.extraterrestrial_horizontal(0, 0.0)), 1361 * (1 + 0.033 * np.cos(2 * np.pi / 365)), places=6)
        self.assertEqual(float(so.extraterrestrial_horizontal(0, 95.0)), 0.0)


@unittest.skipUnless((ec.data_dir() / so.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_probe_and_seed(self):
        value, details = so.score(module_from(PROBE))
        self.assertEqual(details["test_minutes"], 87316)
        seed, _ = so.score(module_from(SOLAR_SEED))
        self.assertLess(seed, value)

    def test_invalid_output(self):
        with self.assertRaises(CandidateError):
            so.score(module_from("def estimate_diffuse(train, test):\n    return {'psu': [1.0]}\n"))


if __name__ == "__main__":
    unittest.main()
