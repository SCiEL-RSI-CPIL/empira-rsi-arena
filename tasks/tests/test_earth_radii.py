"""Hurricane wind-radii forecasts: best-track derivation, verifying cases, output validation and the seed."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_radii as rd  # noqa: E402
from autoresearch_suite import earth_sandbox as es  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
RADII_SEED = seed_source("science-earth-hurricane-wind-radii-forecast")

SAMPLE = """AL092021,                IDA,      3,
20210828, 0000,  , HU, 22.6N,  83.5W,  70,  989,  100,   60,   40,   70,   50,   30,    0,   30,   20,    0,    0,    0,   20
20210828, 0600,  , HU, 23.5N,  84.7W,  70,  987,  100,   60,   40,   70,   50,   30,    0,   30,   20,    0,    0,    0,   20
20210828, 0730, L, HU, 23.6N,  84.8W,  75,  985, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999
"""

SPY = '''
import numpy as np


class Spy:
    def forecast(self, history, future):
        last = history["time_hours"][-1]
        out = np.full((5, 4), 50.0)
        out[0, 0] = len(history["time_hours"])
        return out


def fit_radii_model(training_storms):
    assert max(s["year"] for s in training_storms) == 2014
    return Spy()
'''


def storm(status, r34, vmax=None):
    n = len(status)
    return {"status": list(status), "time_hours": np.arange(n) * 6.0, "vmax_kt": np.array(vmax or [50.0] * n),
            "r34": np.array(r34, float), "lat": np.full(n, 20.0)}


class RuleTests(unittest.TestCase):
    def test_derive_keeps_synoptic_records_with_radii(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / rd.RAW_SOURCES[0]
            path.write_text(SAMPLE)
            payload = rd.derive({rd.RAW_SOURCES[0]: path})[rd.DATA]
        records = payload["storms"][0]["records"]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][6:10], [100, 60, 40, 70])

    def test_cases_need_radii_at_origin_and_verifying_time(self):
        r = [[50, 50, 50, 50]] * 3 + [[np.nan] * 4] + [[60, 60, 60, 60]] * 10
        rows = rd.cases([storm(["TS"] * 14, r)])
        origins = [i for _, _, i, _ in rows]
        self.assertNotIn(3, origins)
        first = rows[0][3]
        self.assertEqual(first[0], 2)  # 12 h after index 0
        self.assertEqual(rd.cases([storm(["TD"] * 14, r)]), [])

    def test_validation(self):
        good = np.full((5, 4), 80.0)
        np.testing.assert_array_equal(rd.validate(0, good), good)
        for bad in (np.full((4, 4), 80.0), np.full((5, 4), -1.0), np.full((5, 4), np.nan)):
            with self.assertRaises(CandidateError):
                rd.validate(0, bad)


@unittest.skipUnless((ec.data_dir() / rd.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_history_ends_at_the_forecast_time(self):
        train, test = rd.split()
        rows = rd.cases(test)
        steps = [{"append": None, "args": [rd.view(test[k], i), rd.future(test[k], v)]} for _, k, i, v in rows]
        outputs = es.run_online(module_from(SPY), "fit_radii_model", {"args": [[rd.view(s) for s in train]], "history": None},
                                steps, rd.validate, with_history=False)
        self.assertEqual([int(o[0, 0]) for o in outputs], [i + 1 for _, _, i, _ in rows])
        self.assertTrue(all(np.diff([t for t, _, _, _ in rows]) >= 0))

    def test_seed_is_persistence(self):
        value, details = rd.score(module_from(RADII_SEED))
        self.assertAlmostEqual(value, details["persistence_mean_error"], places=2)
        self.assertLess(details["regression_reference_mean_error"], value)


if __name__ == "__main__":
    unittest.main()
