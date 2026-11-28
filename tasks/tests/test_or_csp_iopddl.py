"""IOPDDL auto-sharding evaluator (package or_csp): port of iopddl::Evaluate and contest points."""
from __future__ import annotations

import copy
import csv
import gzip
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "suite" / "_lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import or_csp_iopddl as I  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

# example.json of github.com/google/iopddl (Apache-2.0); expected values are those of its iopddl_test.cc
EXAMPLE = {"name": "example",
           "nodes": {"intervals": [[30, 70], [40, 70], [50, 120], [110, 140], [110, 150]],
                     "costs": [[15], [55, 65], [25, 45, 35], [85, 75], [95]],
                     "usages": [[10], [25, 25], [15, 20, 15], [10, 10], [15]]},
           "edges": {"nodes": [[0, 1], [0, 2], [1, 3], [2, 4], [3, 4]],
                     "costs": [[30, 40], [50, 10, 40], [90, 10, 20, 80], [60, 20, 30], [70, 60]]},
           "usage_limit": 50}


class EvaluateTests(unittest.TestCase):
    def test_official_unit_test_values(self):
        self.assertEqual(I.total_cost(EXAMPLE, [0, 1, 2, 0, 0]), 495)
        unlimited = copy.deepcopy(EXAMPLE)
        del unlimited["usage_limit"]
        self.assertEqual(I.total_cost(unlimited, [0, 0, 1, 1, 0]), 415)
        self.assertEqual(I.total_cost(EXAMPLE, [0, 0, 2, 1, 0]), 445)  # the optimum quoted by the contest
        for bad in ([0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 4, 0, 0]):
            with self.subTest(solution=bad), self.assertRaises(ValueError):
                I.total_cost(EXAMPLE, bad)

    def test_half_open_intervals_and_exact_integers(self):
        problem = {"nodes": {"intervals": [[0, 2], [2, 4]], "costs": [[10 ** 18, 1], [10 ** 18]], "usages": [[5, 5], [5]]},
                   "edges": {"nodes": [[0, 1]], "costs": [[10 ** 18, 10 ** 18]]}, "usage_limit": 5}
        self.assertEqual(I.total_cost(problem, [0, 0]), 3 * 10 ** 18)  # beyond int64 when summed
        self.assertEqual(I.total_cost(problem, [1, 0]), 2 * 10 ** 18 + 1)  # touching intervals never overlap

    def test_best_costs_match_the_contest_results(self):
        path = I.DATA / "iopddl_final_results.csv"
        if not path.is_file():
            self.skipTest("prepared contest results absent: run scripts/prepare_or_csp.py")
        rows = [r for r in csv.reader(path.open()) if r]
        header, teams = rows[0], rows[1:]
        for j, benchmark in enumerate(header[1:], 1):
            costs = [int(t[j]) for t in teams if j < len(t) and t[j].strip().isdigit()]
            self.assertEqual(I.BEST_TEAM_COST[benchmark], min(costs), benchmark)


class ContestPointsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with gzip.open(Path(self.tmp.name) / "asplos-2025-iopddl-G.json.gz", "wt") as stream:
            json.dump({"problem": EXAMPLE}, stream)
        self.best = I.BEST_TEAM_COST["G"]
        I.BEST_TEAM_COST["G"] = 445  # the example's optimum stands in for the contest's best cost

    def tearDown(self):
        I.BEST_TEAM_COST["G"] = self.best
        self.tmp.cleanup()

    def points(self, solve, time_limit=5.0):
        module = types.SimpleNamespace(solve=solve)
        return I.contest_points(module, {"benchmarks": ["G"], "time_limit_s": time_limit, "grace_s": 0.2}, data_dir=self.tmp.name)

    def test_points_and_statuses(self):
        score, details = self.points(lambda p, t: [0, 0, 2, 1, 0])
        self.assertAlmostEqual(score, 1.0)
        score, _ = self.points(lambda p, t: [0, 1, 2, 0, 0])
        self.assertAlmostEqual(score, 445 / 495)
        for solve, status in ((lambda p, t: [0, 0, 1, 1, 0], "usage_limit_exceeded"), (lambda p, t: [], "no_solution")):
            score, details = self.points(solve)
            self.assertEqual((score, details["benchmarks"][0]["status"]), (0.0, status))

    def test_late_answers_earn_nothing(self):
        def slow(problem, time_limit_s):
            time.sleep(time_limit_s + 0.5)
            return [0, 0, 2, 1, 0]
        score, details = self.points(slow, time_limit=0.2)
        self.assertEqual((score, details["benchmarks"][0]["status"]), (0.0, "late"))

    def test_malformed_answers_are_invalid_and_mutation_is_harmless(self):
        for bad in (lambda p, t: [0, 0, 2], lambda p, t: [0, 0, 2, 1, 7], lambda p, t: [0.0] * 5, lambda p, t: None, lambda p, t: [True] * 5):
            with self.assertRaises(CandidateError):
                self.points(bad)

        def cheat(problem, time_limit_s):
            for row in problem["edges"]["costs"]:
                row[:] = [0] * len(row)
            problem["usage_limit"] = 10 ** 9
            return [0, 1, 2, 0, 0]
        score, _ = self.points(cheat)
        self.assertAlmostEqual(score, 445 / 495)


class LockTests(unittest.TestCase):
    def test_lock_entries_are_complete(self):
        lock = json.loads((ROOT / "catalog" / "or_csp-data-lock.json").read_text())
        self.assertTrue(lock["directory"].startswith("data/or_csp of the task cache"))  # the Arena keeps data outside the library
        for entry in lock["files"]:
            with self.subTest(path=entry["path"]):
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(entry["url"].startswith("https://"))
                self.assertTrue(entry["license"])
                self.assertGreater(entry["size_bytes"], 0)
        self.assertLess(lock["total_stored_bytes"], 50 * 2 ** 20)
        self.assertEqual(lock["total_stored_bytes"], sum(e["stored_bytes"] for e in lock["files"]))


if __name__ == "__main__":
    unittest.main()
