"""River nitrate loads: record parsing, the sampling design, the weighted regression reference and the seed."""
from __future__ import annotations

import datetime
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_nitrate as ni  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
NITRATE_SEED = seed_source("science-earth-river-nitrate-loads")

RDB = """# comment
agency_cd\tsite_no\tdatetime\t1_00060_00003\t1_00060_00003_cd\t2_99133_00003\t2_99133_00003_cd
5s\t15s\t20d\t14n\t10s\t14n\t10s
USGS\t1\t2015-01-01\t100\tA\t5.5\tA
USGS\t1\t2015-01-02\t110\tA:e\t5.6\tA:e
USGS\t1\t2015-01-03\t\t\t5.7\tA
USGS\t1\t2015-01-04\t120\tP\t
"""

PROBE = '''
import numpy as np


def estimate_nitrate(sites):
    out = {}
    for name, site in sites.items():
        assert set(site) == {"day", "decimal_year", "discharge_m3s", "sample_index", "sample_nitrate_mg_l", "lat", "lon", "drainage_area_km2"}
        out[name] = np.full(len(site["day"]), float(np.mean(site["sample_nitrate_mg_l"])))
    return out
'''


class ParsingTests(unittest.TestCase):
    def test_parse_keeps_approved_values_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.rdb"
            path.write_text(RDB)
            rows = ni.parse(path)
        self.assertEqual([r[1] for r in rows], [100.0, 110.0, None, None])
        self.assertEqual([r[2] for r in rows], [5.5, None, 5.7, None])

    def test_sample_design_has_one_day_per_month_and_high_flow_days(self):
        days = np.arange(16071, 16071 + 365)  # 2014-01-01 onwards
        discharge = np.linspace(1, 10, 365)
        truth = np.ones(365)
        index = ni.sample_design(days, discharge, truth, seed=3)
        months = {(ni.EPOCH + datetime.timedelta(days=int(d))).month for d in days[index]}
        self.assertEqual(len(months), 12)
        self.assertGreaterEqual(int(np.sum(discharge[index] >= np.quantile(discharge, 0.8))), 4)


class WrtdsTests(unittest.TestCase):
    def test_recovers_an_exact_log_linear_model(self):
        rng = np.random.default_rng(0)
        t = 2010 + np.arange(2000) / 365.25
        q = np.exp(rng.normal(3, 1, size=2000))
        truth = np.exp(0.5 + 0.03 * (t - 2000) + 0.2 * np.log(q) + 0.3 * np.sin(2 * np.pi * t) - 0.1 * np.cos(2 * np.pi * t))
        index = np.sort(rng.choice(2000, size=150, replace=False))
        site = {"decimal_year": t, "discharge_m3s": q, "sample_index": index.astype(float), "sample_nitrate_mg_l": truth[index]}
        np.testing.assert_allclose(ni.wrtds(site), truth, rtol=1e-6)

    def test_load_errors(self):
        site = {"day": np.arange(16071, 16071 + 31, dtype=float), "discharge_m3s": np.full(31, 2.0)}
        truth = np.full(31, 4.0)
        self.assertAlmostEqual(float(ni.load_errors(site, truth, np.full(31, 5.0))[0]), 0.25)
        truth[:10] = np.nan  # only 21 sensor days: month not scored
        self.assertEqual(len(ni.load_errors(site, truth, np.full(31, 5.0))), 0)


@unittest.skipUnless((ec.data_dir() / ni.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_probe_and_seed(self):
        value, details = ni.score(module_from(PROBE))
        self.assertEqual(details["site_months"], 959)
        seed, seed_details = ni.score(module_from(NITRATE_SEED))
        self.assertLess(seed, value)
        self.assertLess(seed_details["wrtds_mean_abs_monthly_load_error_percent"], seed)

    def test_invalid_output(self):
        with self.assertRaises(CandidateError):
            ni.score(module_from("def estimate_nitrate(sites):\n    return {s: [-1.0] * len(v['day']) for s, v in sites.items()}\n"))


if __name__ == "__main__":
    unittest.main()
