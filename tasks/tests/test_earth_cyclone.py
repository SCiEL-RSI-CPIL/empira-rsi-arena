"""Hurricane track and intensity forecasts: best-track derivation, verifying cases, output validation and held-out scoring."""
from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_cyclone as c  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from  # noqa: E402

# The evaluator audit moved the candidate's calls into the sandboxed child process, so a candidate is a module with a
# source file and shares no state with the test. The spy checks the order of its calls itself and fails its forecast
# when a history arrives out of order. The read-only check of the source repository's test no longer applies: the
# child holds its own copy of each history.
SPY = '''
import numpy as np


class Spy:
    latest = -1

    def forecast(self, history):
        now = int(history["time_hours"][-1])
        if now < Spy.latest or now >= (2025 - 1950) * 366 * 24:  # calls are chronological and end before 2025
            raise RuntimeError("history out of order")
        Spy.latest = now
        return np.full(7, float(history["vmax_kt"][-1]))


def fit_intensity_model(storms):
    return Spy()
'''
# The evaluator's persistence rule, copied into the candidate: the child cannot import the evaluator package.
CONSTANT = "import numpy as np\n\nLEADS = " + repr(c.LEADS) + "\n\n\n" + inspect.getsource(c.persistence) + '''

class Constant:
    def forecast(self, history):
        return persistence({"lat": np.asarray(history["lat"]), "lon": np.asarray(history["lon"]),
                            "vmax_kt": np.asarray(history["vmax_kt"])}, len(history["lat"]) - 1, "track")


def fit_track_model(storms):
    return Constant()
'''

SAMPLE = """AL011949,            UNNAMED,      1,
19490601, 0000,  , TS, 20.0N,  80.0W,  40, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999
AL022015,                ANA,      5,
20150508, 0000,  , SS, 31.5N,  77.3W,  40, 1003, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999
20150508, 0600,  , SS, 31.6N,  77.2W,  45, 1002, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999
20150508, 0915, L, TS, 31.7N,  77.1W,  45, 1001, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999
20150508, 1200,  , TS, 31.8N,  77.0W,  50,  999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999
20150508, 1800,  , EX, 32.0N,   1.0E,  45, 1000, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999
"""


def storm(status, lat, vmax):
    n = len(status)
    return {"id": "X", "year": 2016, "time_hours": np.arange(n) * 6, "month": np.full(n, 9), "day": np.full(n, 1),
            "hour": np.zeros(n, int), "day_of_year": np.full(n, 244), "status": tuple(status), "lat": np.array(lat, float),
            "lon": np.full(n, -60.0), "vmax_kt": np.array(vmax, float), "mslp_hpa": np.full(n, np.nan)}


class ParsingAndProtocolTests(unittest.TestCase):
    def test_derive_keeps_synoptic_records_from_1950(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / c.RAW_SOURCES[0]
            path.write_text(SAMPLE)
            payload = c.derive({c.RAW_SOURCES[0]: path})[c.DATA]
        self.assertEqual([s["id"] for s in payload["storms"]], ["AL022015"])
        records = payload["storms"][0]["records"]
        self.assertEqual(len(records), 4)  # the 09:15 landfall record is not synoptic
        self.assertEqual(records[0], [2015050800, "SS", 31.5, -77.3, 40, 1003])
        self.assertEqual(records[-1][3], 1.0)  # east longitudes are positive

    def test_verification_requires_tropical_status_at_both_times(self):
        status = ["LO", "TD", "TS"] + ["HU"] * 3 + ["EX"] * 20
        rows = c.cases([storm(status, np.linspace(10, 30, len(status)), [25] * len(status))])
        origins = [i for _, _, i, _ in rows]
        self.assertEqual(origins, [1, 2, 3])  # LO origin skipped; HU at indices 4-5 verify only on EX records
        first = rows[0][3]
        self.assertEqual(first[0], 3)  # 12 h after index 1 is index 3 (HU)
        self.assertTrue(all(j == -1 for j in first[2:]))  # later leads fall on EX records

    def test_validation_rejects_bad_forecasts(self):
        good = np.zeros((7, 2))
        np.testing.assert_array_equal(c.validate("track", good), good)
        for bad in (np.zeros((6, 2)), np.full((7, 2), np.nan), np.array([[95.0, 0.0]] * 7), "north"):
            with self.assertRaises(CandidateError):
                c.validate("track", bad)
        with self.assertRaises(CandidateError):
            c.validate("intensity", np.full(7, -5.0))

    def test_haversine_degree_of_latitude(self):
        self.assertAlmostEqual(float(ec.haversine_km(0.0, 0.0, 1.0, 0.0)), 111.195, places=2)
        self.assertAlmostEqual(float(ec.haversine_km(10.0, 179.5, 10.0, -179.5)), float(ec.haversine_km(10.0, 0.0, 10.0, 1.0)), places=9)


@unittest.skipUnless((ec.data_dir() / c.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutScoringTests(unittest.TestCase):
    def test_history_never_reaches_past_the_forecast_time(self):
        value, details = c.score(module_from(SPY), "intensity")  # an out-of-order history fails the candidate
        self.assertGreater(details["forecast_calls"], 0)

    def test_persistence_seeds_and_reference(self):
        value, details = c.score(module_from(CONSTANT), "track")
        self.assertAlmostEqual(value, details["persistence_mean_error"], places=3)
        self.assertLess(details["reference_regression_mean_error"], value)
        self.assertEqual(details["verifying_cases"][0], 3601)


if __name__ == "__main__":
    unittest.main()
