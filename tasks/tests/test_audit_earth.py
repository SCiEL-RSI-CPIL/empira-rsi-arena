"""Regression tests of the earth-task audit: held-out data, pinned records and resumable preparation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "suite/_lib"), str(ROOT / "scripts")]
from autoresearch_suite import earth_co2, earth_common, earth_cyclone, earth_icedrift, earth_nitrate, earth_twostream, runtime
import prepare_earth

# Looks for the verifying best track in the frames above forecast(); stays at the last position when there is none.
FRAME_READER = '''
import sys

import numpy as np


class Reader:
    def forecast(self, history):
        frame = sys._getframe(1)
        while frame is not None:
            storm, verify = frame.f_locals.get("storm"), frame.f_locals.get("verify")
            if isinstance(storm, dict) and verify is not None and len(storm.get("lat", ())) > len(history["lat"]):
                return np.array([[storm["lat"][j], storm["lon"][j]] for j in verify])
            frame = frame.f_back
        return np.column_stack([np.full(7, history["lat"][-1]), np.full(7, history["lon"][-1])])


def fit_track_model(training_storms):
    return Reader()
'''
COUNTS_TRIPLES = '''
import numpy as np


def two_stream_coefficients(omega, g, mu0):
    assert len(omega) == %d and len(np.unique(np.column_stack([omega, g, mu0]), axis=0)) == len(omega)
    f = g ** 2
    omega_s, g_s = omega * (1 - f) / (1 - omega * f), (g - f) / (1 - f)
    return (7 - omega_s * (4 + 3 * g_s)) / 4, -(1 - omega_s * (4 - 3 * g_s)) / 4, (2 - 3 * g_s * mu0) / 4, f
'''


def prepared(module):
    return (earth_common.data_dir() / module.DATA).is_file()


class Candidates(unittest.TestCase):
    def candidate(self, source):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "candidate.py"
        path.write_text(source)
        return runtime.load_program(path)


def moving_storm(year, records=30, wobble=0.0):
    """A system moving one degree north every six hours (with an east-west wobble), tropical throughout."""
    n = np.arange(records)
    return {"id": "X", "year": year, "time_hours": n * 6, "month": np.full(records, 9), "day": np.full(records, 1), "hour": (n * 6) % 24,
            "day_of_year": np.full(records, 244), "status": ("TS",) * records, "lat": 10.0 + n, "lon": -60.0 + wobble * np.sin(n),
            "vmax_kt": np.full(records, 50.0), "mslp_hpa": np.full(records, np.nan)}


class HurricaneForecastTests(Candidates):
    def test_the_verifying_best_track_is_not_in_the_candidate_process(self):
        module = self.candidate(FRAME_READER)
        # Called beside the evaluator's loop, the same model reads the verifying positions: this was the leak.
        storm, verify = moving_storm(2016), (2, 4, 6, 8, 12, 16, 20)
        leaked = module.fit_track_model([]).forecast(earth_cyclone.view(storm, 0))
        np.testing.assert_array_equal(leaked[:, 0], storm["lat"][list(verify)])
        with mock.patch.object(earth_cyclone, "split", lambda: ([moving_storm(2000, 40), moving_storm(2001, 40)], [moving_storm(2016, wobble=0.5)])):
            value, details = earth_cyclone.score(module, "track")
        self.assertGreater(value, 500.0)  # no motion against one degree of latitude per six hours
        self.assertEqual(details["forecast_calls"], 28)

    @unittest.skipUnless(prepared(earth_cyclone), "run scripts/prepare_earth.py first")
    def test_persistence_scores_as_before_on_the_best_track(self):
        source = "import numpy as np\n\nclass P:\n    def forecast(self, history):\n        return np.full(7, float(history['vmax_kt'][-1]))\n\n"
        value, details = earth_cyclone.score(self.candidate(source + "def fit_intensity_model(storms):\n    return P()\n"), "intensity")
        self.assertAlmostEqual(value, 19.9109334324246, places=9)
        self.assertEqual(details["verifying_cases"], [3601, 3211, 2844, 2521, 1951, 1498, 1156])


class TwoStreamDesignTests(Candidates):
    def test_every_triple_has_one_depth_in_each_decade(self):
        tau, omega, g, mu0 = earth_twostream.sample_cases()
        triples, first, counts = np.unique(np.column_stack([omega, g, mu0]), axis=0, return_index=True, return_counts=True)
        self.assertEqual(len(tau), earth_twostream.CASES)
        self.assertEqual(len(triples), earth_twostream.CASES // earth_twostream.DEPTHS)
        self.assertTrue(np.all(counts == earth_twostream.DEPTHS))
        blocks = tau.reshape(-1, earth_twostream.DEPTHS)
        for k, low in enumerate((0.01, 0.1, 1.0, 10.0)):
            self.assertTrue(np.all((blocks[:, k] >= low) & (blocks[:, k] <= 10 * low)))
        self.assertAlmostEqual(float(np.mean(omega == 1.0)), 0.25, delta=0.02)

    @unittest.skipUnless(prepared(earth_twostream), "run scripts/prepare_earth.py first")
    def test_one_set_of_coefficients_serves_all_depths_of_a_triple(self):
        triples = earth_twostream.CASES // earth_twostream.DEPTHS
        value, details = earth_twostream.score(self.candidate(COUNTS_TRIPLES % triples))
        self.assertEqual(details["cases"], earth_twostream.CASES)
        self.assertAlmostEqual(value, details["delta_eddington_mean_error"], places=6)


def rdb(site, approved_through, provisional_value="9.9"):
    """A daily-values file of 2023-01-01 to 2025-09-30: approved up to a date, provisional afterwards."""
    import datetime as dt
    lines = ["# retrieved: test", "agency_cd\tsite_no\tdatetime\t1_00060_00003\t1_00060_00003_cd\t2_99133_00003\t2_99133_00003_cd", "5s\t15s\t20d\t14n\t10s\t14n\t10s"]
    day, last = dt.date(2023, 1, 1), dt.date(2025, 9, 30)
    while day <= last:
        approved = day.isoformat() <= approved_through
        flow = 1000 + 400 * np.sin(day.toordinal() / 58.0) + (day.toordinal() * 37 % 101)
        nitrate = f"{2 + np.cos(day.toordinal() / 40.0):.2f}" if approved else provisional_value
        lines.append(f"USGS\t{site}\t{day.isoformat()}\t{flow:.0f}\t{'A' if approved else 'P'}\t{nitrate}\t{'A' if approved else 'P'}")
        day += dt.timedelta(days=1)
    return "\n".join(lines) + "\n"


class NitrateRecordTests(unittest.TestCase):
    def derive(self, approved_through, provisional_value="9.9"):
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for site, name in zip(earth_nitrate.SITES, earth_nitrate.RAW_SOURCES):
                paths[name] = Path(folder) / name
                paths[name].write_text(rdb(site, approved_through if site == "07374000" else "2025-09-30", provisional_value))
            return earth_nitrate.derive(paths)[earth_nitrate.DATA]["sites"]

    def test_later_approvals_do_not_change_a_record(self):
        self.assertEqual(set(earth_nitrate.RECORD_END), set(earth_nitrate.SITES))
        end = earth_nitrate.RECORD_END["07374000"]
        locked = self.derive(end)  # the state when the lock was recorded
        later = self.derive("2025-01-19", "3.3")  # NWIS has since approved nine more months, with other values
        self.assertEqual(locked, later)
        import datetime as dt
        last = earth_nitrate.EPOCH + dt.timedelta(days=locked["07374000"]["day"][-1])
        self.assertEqual(last.isoformat(), end)
        self.assertEqual(len(locked["03374100"]["day"]), 1004)  # the other sites keep the whole request


class CarbonDioxideFileTests(unittest.TestCase):
    def derive(self, late_index):
        months = [(y, m) for y in range(1958, 2026) for m in range(1, 13)]
        index = [0.1 if (y, m) <= earth_co2.TEST_ORIGINS[1] else late_index for y, m in months]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "co2.txt"
            path.write_text("# header\n" + "".join(f"{y} {m} {y + m / 12:.4f} {300 + 0.1 * i:.2f} 0 {25 if y > 1974 else -1} 0 0\n"
                                                   for i, (y, m) in enumerate(months) if (y, m) >= (1958, 3)))
            with mock.patch.object(earth_co2.earth_gmst, "relative_nino34",
                                   lambda source: (np.array([y for y, _ in months]), np.array([m for _, m in months]), np.array(index))):
                return earth_co2.derive({earth_co2.RAW_SOURCES[0]: path, earth_co2.RAW_SOURCES[1]: path})[earth_co2.DATA]

    def test_index_months_after_the_last_origin_are_not_stored(self):
        first, revised = self.derive(0.2), self.derive(0.7)  # ERSSTv5 revises its recent months with every update
        self.assertEqual(first, revised)
        last_origin = first["year"].index(2024) + 11
        self.assertEqual(first["relative_nino34"][last_origin], 0.1)
        self.assertEqual(first["relative_nino34"][last_origin + 1:], [None] * 12)
        self.assertEqual(len(first["co2_ppm"]), len(first["relative_nino34"]))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        for name in ("RAW", "WORK"):
            patch = mock.patch.object(prepare_earth, name, Path(self.folder.name) / name.lower())
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_archived_copy_of_a_live_source_is_used(self):
        locked, live = b"the locked bytes\n", b"the publisher has moved on\n"
        import hashlib
        source = {"name": "file.txt", "url": "https://publisher.example/file.txt", "archive_url": "https://archive.example/file.txt",
                  "pinning": "live", "sha256": hashlib.sha256(locked).hexdigest()}
        served = {source["url"]: live, source["archive_url"]: locked}
        asked = []

        def download(url, destination):
            asked.append(url)
            if served[url] is None:
                raise RuntimeError(f"download failed for {url}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(served[url])

        with mock.patch.object(prepare_earth, "download", download):
            (prepare_earth.RAW).mkdir(parents=True)
            (prepare_earth.RAW / "file.txt").write_bytes(live)  # an earlier download of the live file
            self.assertEqual(prepare_earth.fetch(source).read_bytes(), locked)
            self.assertEqual(asked, [source["archive_url"]])
            self.assertEqual(prepare_earth.fetch(source).read_bytes(), locked)  # cached and verified: no request
            self.assertEqual(len(asked), 1)
            (prepare_earth.RAW / "file.txt").unlink()
            served[source["archive_url"]] = None  # archive unreachable: the live file and the derived-file hashes decide
            self.assertEqual(prepare_earth.fetch(source).read_bytes(), live)

    def test_a_stepwise_derivation_resumes_and_gives_the_same_file(self):
        calls = []

        def derive_part(paths, part):
            calls.append(part)
            return [(part, np.float64(part) / 3, -0.0)]

        module = SimpleNamespace(__file__=__file__, PARTS=(1, 2, 3), derive_part=derive_part,
                                 assemble=lambda parts: {"x": sorted(tuple(r) for p in parts for r in p)})
        whole = earth_common.canonical_bytes(module.assemble([derive_part({}, p) for p in module.PARTS]))
        calls.clear()
        with self.assertRaises(prepare_earth.Unfinished):
            prepare_earth.derive(module, "fake", {}, deadline=time.monotonic() - 1)  # budget used: nothing is started
        self.assertEqual(calls, [])
        with mock.patch.object(module, "derive_part", lambda paths, part: derive_part(paths, part) if part < 3 else 1 / 0):
            with self.assertRaises(ZeroDivisionError):  # interrupted in the third part
                prepare_earth.derive(module, "fake", {})
        payload, work = prepare_earth.derive(module, "fake", {})
        self.assertEqual(calls, [1, 2, 3])  # the two finished parts were not computed again
        self.assertEqual(earth_common.canonical_bytes({k: [list(r) for r in v] for k, v in payload.items()}),
                         earth_common.canonical_bytes(json.loads(json.dumps({"x": sorted([p, p / 3, -0.0] for p in (1, 2, 3))}))))
        self.assertEqual(earth_common.canonical_bytes(payload), whole)
        self.assertTrue(work.is_dir())

    def test_ice_drift_parts_assemble_to_the_sorted_file(self):
        self.assertEqual(earth_icedrift.PARTS, tuple(earth_icedrift.YEARS))
        rows = [[(8, 2, 80.0, 10.0, 1010.25, 1.5, -2.0, 3.0, -0.0)], [(3, 9, 75.5, -20.0, 990.0, 0.0, 0.25, -1.125, 4.0), (3, 1, 71.0, 5.0, 1000.0, 2.0, 2.0, 2.0, 2.0)]]
        direct = earth_icedrift.assemble(rows)
        stored = earth_icedrift.assemble(json.loads(json.dumps(rows)))  # as scripts/prepare_earth.py reads finished parts
        self.assertEqual(earth_common.canonical_bytes(direct), earth_common.canonical_bytes(stored))
        data = direct[earth_icedrift.DATA]
        self.assertEqual((data["day"], data["buoy"]), ([3, 3, 8], [1, 9, 2]))


if __name__ == "__main__":
    unittest.main()
