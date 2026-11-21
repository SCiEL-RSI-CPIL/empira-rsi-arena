"""The packages added after the first import: completeness, receipts and a sample of seed scores."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "suite/_lib")]
from autoresearch_tasks.registry import get_task
from autoresearch_tasks.runner import evaluate_suite

PACKAGE_FILES = ("task.json", "problem.json", "evaluator.py", "initial_program.py", "config.yaml", "README.txt")


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


class SuiteAdditionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load("validation/suite-additions-report.json")
        cls.rows = {row["id"]: row for row in cls.report["tasks"]}

    def test_every_added_package_is_complete_and_catalogued(self):
        catalog = {record["id"]: record for record in load("catalog/tasks.json")}
        index = {row["id"]: row for row in load("suite/index.json")["tasks"]}
        provenance = load("validation/suite-additions-provenance.json")
        self.assertEqual(len(self.rows), provenance["added_task_count"])
        for task_id in self.rows:
            for name in PACKAGE_FILES:
                self.assertTrue((ROOT / "suite" / task_id / name).is_file(), f"{task_id}/{name}")
            problem = load(f"suite/{task_id}/problem.json")
            suite = catalog[task_id]["execution"]["suite"]
            self.assertEqual((problem["id"], problem["protocol"], problem["family"]),
                             (task_id, suite["protocol"], suite["family"]))
            self.assertEqual((index[task_id]["protocol"], index[task_id]["family"], index[task_id]["direction"]),
                             (suite["protocol"], suite["family"], problem["direction"]))
            self.assertEqual(load(f"suite/{task_id}/task.json")["labels"], catalog[task_id]["labels"])

    def test_seed_receipts_name_the_files_that_are_in_the_library(self):
        executed = [row for row in self.rows.values() if row["status"] == "ok"]
        self.assertGreater(len(executed), 300)
        for row in executed:
            self.assertTrue(row["valid"] is True and row["source_execution"] is True and row["fixture_only"] is False)
            self.assertEqual(row["score"], row["result"]["score"])
            for name, recorded in row["file_sha256"].items():
                self.assertEqual(digest(name), recorded, f"{row['id']}: {name}")
        for row in self.rows.values():
            if row["status"] != "ok":
                self.assertTrue(row["score"] is None and row["reason"], row["id"])

    def test_provenance_matches_the_files(self):
        for row in load("validation/suite-additions-provenance.json")["files"]:
            self.assertEqual(digest(row["path"]), row["sha256"], row["path"])
            if row["origin"].startswith("identical copy"):
                self.assertEqual(row["sha256"], row["source_sha256"])
            elif row["origin"] == "adapted":
                self.assertTrue(row["note"] and row["sha256"] != row["source_sha256"], row["path"])

    def test_sample_of_seeds_reproduces_the_recorded_scores(self):
        # One package per collection among those that need nothing beyond the standard library.
        sample = {}
        for row in self.rows.values():
            suite = get_task(ROOT, row["id"])["execution"]["suite"]
            collection = get_task(ROOT, row["id"])["paper_ids"][0]
            if row["status"] == "ok" and not suite["requires"] and collection not in sample:
                sample[collection] = row
        self.assertGreaterEqual(len(sample), 8)
        for row in sample.values():
            with self.subTest(task=row["id"]):
                result = evaluate_suite(ROOT, get_task(ROOT, row["id"]), timeout=120)
                self.assertEqual((result["status"], result["valid"]), ("completed", True))
                self.assertEqual(result["score"], row["score"])
                self.assertEqual(result["evaluator_sha256"], row["evaluator_sha256"])

    def test_prepared_data_bytes_do_not_depend_on_the_platform(self):
        from autoresearch_suite import bio_physics_common, earth_common
        payload = {"values": list(range(64)), "name": "fixture"}
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        for module in (earth_common, bio_physics_common):
            data = module.canonical_bytes(payload)
            # zlib would write its build's operating-system code here; the recorded digests carry 19.
            self.assertEqual(data[9], 19)
            self.assertEqual(data[4:8], b"\x00\x00\x00\x00")
            self.assertEqual(gzip.decompress(data), text)
        self.assertEqual(earth_common.canonical_bytes(payload), bio_physics_common.canonical_bytes(payload))

    def test_split_number_theory_module_keeps_its_interface(self):
        from autoresearch_suite import science_prize_h, science_prize_h_arithmetic, science_prize_h_sets
        for name in ("factor", "int_set", "primes_up_to", "run_certificate", "is_prime"):
            self.assertIs(getattr(science_prize_h, name), getattr(science_prize_h_arithmetic, name))
        for name in ("sidon_squares", "odd_covering", "two_cube_sums", "van_der_waerden"):
            self.assertIs(getattr(science_prize_h, name), getattr(science_prize_h_sets, name))
        self.assertTrue(science_prize_h.is_prime(2**61 - 1))
        # 3215031751 = 151 * 751 * 28351 passes the strong test for the bases 2, 3, 5 and 7.
        self.assertFalse(science_prize_h.is_prime(3215031751))


if __name__ == "__main__":
    unittest.main()
