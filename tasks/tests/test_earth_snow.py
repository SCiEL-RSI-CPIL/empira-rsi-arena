"""Snow depth to water equivalent: season days, the station split, derivation and the seed."""
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
from autoresearch_suite import earth_snow as sn  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, needs, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
SNOW_SEED = seed_source("science-earth-snow-depth-to-swe")

PROBE = '''
import numpy as np


def estimate_swe(train, test):
    assert all("swe_mm" not in s for s in test.values()) and all("swe_mm" in s for s in train.values())
    return {k: 3.0 * s["depth_cm"][s["target_index"].astype(int)] for k, s in test.items()}
'''


class UnitTests(unittest.TestCase):
    def test_season_day_and_targets(self):
        days = np.array([(sn.dt.date(2020, 10, 1) - sn.EPOCH).days, (sn.dt.date(2021, 1, 1) - sn.EPOCH).days, (sn.dt.date(2021, 3, 1) - sn.EPOCH).days])
        np.testing.assert_array_equal(sn.doy(days), [-92, 0, 59])
        station = {"depth_cm": np.array([5.0, 20.0, 20.0, 30.0]), "swe_mm": np.array([20.0, 50.0, np.nan, 250.0])}
        np.testing.assert_array_equal(sn.targets(station), [1])  # too shallow, missing SWE, density 0.83

    def test_split_holds_out_about_30_percent_per_state(self):
        held = sn.split()
        self.assertEqual(len(held), 48)
        for state in {s.split(":")[1] for s in sn.STATIONS}:
            members = [s for s in sn.STATIONS if s.split(":")[1] == state]
            self.assertGreaterEqual(sum(s in held for s in members), 1)
            self.assertLess(sum(s in held for s in members), len(members))

    def test_derive_keeps_only_valid_swe(self):
        meta = [{"stationTriplet": "1:XX:SNTL", "name": "A", "stateCode": "XX", "latitude": 40.0, "longitude": -110.0, "elevation": 10000.0}]
        values = lambda pairs: [{"date": d, "value": v, "qcFlag": f} for d, v, f in pairs]  # noqa: E731
        data = [{"stationTriplet": "1:XX:SNTL", "data": [
            {"stationElement": {"elementCode": "SNWD"}, "values": values([("2010-01-01", 20, "V"), ("2010-01-02", 21, "E"), ("2010-01-03", 22, "S")])},
            {"stationElement": {"elementCode": "WTEQ"}, "values": values([("2010-01-01", 5.0, "V"), ("2010-01-02", 5.1, "V")])},
            {"stationElement": {"elementCode": "PREC"}, "values": values([("2010-09-30", 30.0, "V")])}]}]
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for k, name in enumerate(sn.RAW_SOURCES):
                path = Path(tmp) / name
                path.write_text(sn.json.dumps(meta if k == 0 else (data if k == 1 else [])))
                paths[name] = path
            station = sn.derive(paths)[sn.DATA]["stations"]["1:XX:SNTL"]
        self.assertEqual(station["depth_cm"], [50.8, 53.34])  # the suspect depth is dropped
        self.assertEqual(station["swe_mm"], [127.0, None])  # edited depth: SWE not used as truth
        self.assertEqual(station["elevation_m"], 3048.0)
        self.assertEqual(station["precip_normal_mm"], 762.0)


@unittest.skipUnless((ec.data_dir() / sn.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    @needs("scipy")
    def test_probe_and_seed(self):
        value, details = sn.score(module_from(PROBE))
        self.assertEqual(details["held_out_days"], 103150)
        seed, seed_details = sn.score(module_from(SNOW_SEED))
        self.assertLess(seed, value)
        self.assertLess(seed, seed_details["median_density_rmse_mm"])

    @needs("scipy")
    def test_invalid_output(self):
        with self.assertRaises(CandidateError):
            sn.score(module_from("def estimate_swe(train, test):\n    return {}\n"))


if __name__ == "__main__":
    unittest.main()
