"""Great Lakes wave hindcast: buoy record parsing, the station table and the seed."""
from __future__ import annotations

import gzip
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_waves as wv  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
WAVES_SEED = seed_source("science-earth-great-lakes-wave-hindcast")

TEXT = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC   mi    ft
2021 07 08 15 40 327  8.3  9.8  1.38  5.88  4.60 355 1012.7  16.1  18.2  15.5 99.0 99.00
2021 07 08 15 50 332  8.4 10.0 99.00 99.00 99.00 999 1012.7  16.1  18.2  15.3 99.0 99.00
2021 07 08 16 00 335  8.5 10.1 99.00 99.00 99.00 999 1012.8   MM  18.2  15.3 99.0 99.00
"""

PROBE = '''
import numpy as np


def hindcast_waves(train, test):
    assert all(k not in s for s in test.values() for k in ("wvht", "dpd", "apd", "mwd"))
    assert max(s["hour"].max() for s in train.values()) < min(s["hour"].min() for s in test.values())
    return {b: np.full(len(s["target_index"]), 0.5) for b, s in test.items()}
'''


BAD = """import numpy as np


def hindcast_waves(train, test):
    return {b: np.full(len(s["target_index"]), -1.0) for b, s in test.items()}
"""


class ParsingTests(unittest.TestCase):
    def test_ten_minute_records_are_merged_per_hour(self):
        records = wv.parse_year(gzip.compress(TEXT.encode()))
        hour = round((wv.dt.datetime(2021, 7, 8, 16) - wv.EPOCH).total_seconds() / 3600)
        self.assertEqual(list(records), [hour])
        row = records[hour]
        self.assertEqual(row["wspd"], 8.5)  # the 16:00 record is closest
        self.assertEqual(row["wvht"], 1.38)  # the only valid wave height, 20 minutes away
        self.assertEqual(row["atmp"], 16.1)  # MM at 16:00, so the 15:50 value is used
        self.assertIsNone(wv._value("99.00", 99))

    def test_station_table(self):
        line = "45007|NDBC|Weather Buoy|3D|SOUTH MICHIGAN|ARES 4.4|42.674 N 87.026 W (42 40 26 N 87 1 34 W)| |||"
        self.assertEqual(wv.parse_station_table(line)["45007"], (42.674, -87.026))


@unittest.skipUnless((ec.data_dir() / wv.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_probe_and_seed(self):
        value, details = wv.score(module_from(PROBE))
        self.assertEqual(details["held_out_hours"], 177719)
        seed, seed_details = wv.score(module_from(WAVES_SEED))
        self.assertLess(seed, value)
        self.assertLess(seed_details["sector_regression_reference_rmse_m"], seed)

    def test_invalid_output(self):
        with self.assertRaises(CandidateError):
            wv.score(module_from(BAD))


if __name__ == "__main__":
    unittest.main()
