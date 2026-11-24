"""QuixBugs repair: patch judging, held-out test hygiene, generated inputs and the seed procedure."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite/_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from arena_task_paths import cache_path  # noqa: E402
from autoresearch_suite import software_data, software_quixbugs as qb, software_quixbugs_inputs as inputs  # noqa: E402
from autoresearch_suite.runtime import CandidateError, load_program  # noqa: E402
from suite_test_support import seed_module  # noqa: E402

# The seed is the package's: the seed module of the source repository is not in this library.
seed = seed_module("science-software-quixbugs-repair")

PREPARED = (software_data.data_dir() / "quixbugs/repair.json").is_file()
RAW = cache_path("software-raw", "quixbugs")

BUGGY = "def double(x):\n    \"\"\"Twice x.\"\"\"\n    return x + 1\n"
FIXED = "def double(x):\n    return x * 2\n"
TESTS = [{"input": [2], "expected": 4}, {"input": [0], "expected": 0}]


def program(**extra):
    record = {"name": "double", "entry": "double", "source": BUGGY, "visible": TESTS[:1],
              "held_out": TESTS[1:], "generated": [{"input": [n], "expected": 2 * n} for n in range(3, 9)]}
    record.update(extra)
    return record


class Judging(unittest.TestCase):
    def test_statement_edits_ignore_formatting_and_docstrings(self):
        self.assertEqual(qb.statement_edits(BUGGY, "def double(x):\n    return (x + 1)\n"), 0)
        self.assertEqual(qb.statement_edits(BUGGY, FIXED), 1)
        self.assertEqual(qb.statement_edits(BUGGY, "def double(x):\n    y = x\n    z = y\n    return x * 2\n"), 3)

    def test_statuses(self):
        self.assertEqual(qb.judge_patch(program(), FIXED)["status"], "repaired")
        # passes the visible test (2 -> 4) but not the held-out ones: overfitting
        self.assertEqual(qb.judge_patch(program(), "def double(x):\n    return x + 2\n")["status"], "plausible")
        generated_only = qb.judge_patch(program(held_out=[]), "def double(x):\n    return 4 if x == 2 else 0\n")
        self.assertEqual(generated_only["status"], "plausible")
        self.assertTrue(generated_only["passes_all_quixbugs_tests"])
        self.assertEqual(qb.judge_patch(program(), "def double(x):\n    return x\n")["status"], "unrepaired")
        self.assertEqual(qb.judge_patch(program(), "def double(x:\n")["status"], "rejected")
        rewrite = "def double(x):\n    a = 1\n    b = 2\n    c = 3\n    return x * 2\n"
        self.assertEqual(qb.judge_patch(program(), rewrite)["status"], "rejected")

    def test_looping_patch_fails_quickly(self):
        result = qb.judge_patch(program(), "def double(x):\n    while True:\n        pass\n")
        self.assertEqual(result["status"], "unrepaired")

    def test_public_task_hides_held_out_and_generated_tests(self):
        task = qb.public_task(program())
        self.assertEqual(task["tests"], TESTS[:1])
        self.assertNotIn("held_out", json.dumps(task))
        self.assertNotIn("generated", json.dumps(task))


class GeneratedInputs(unittest.TestCase):
    def test_deterministic_and_unique(self):
        for name in inputs.GENERATORS:
            with self.subTest(name):
                first = inputs.generate(name)
                self.assertEqual(first, inputs.generate(name))
                self.assertEqual(len({repr(a) for a in first}), len(first))
                self.assertGreaterEqual(len(first), 30)

    def test_preconditions(self):
        for args in inputs.generate("kheapsort"):
            arr, k = args
            self.assertTrue(all(abs(sorted(arr).index(v) - i) <= k for i, v in enumerate(arr)))
        for a, b in inputs.generate("longest_common_subsequence"):
            self.assertEqual(len(inputs.all_lcs(a, b)), 1)
        for (perm,) in inputs.generate("next_permutation"):
            self.assertEqual(len(set(perm)), len(perm))
            self.assertNotEqual(perm, sorted(perm, reverse=True))
        for x, eps in inputs.generate("sqrt"):
            self.assertTrue(x >= 1 and eps > 0)
        self.assertEqual(inputs.all_lcs("ab", "ba"), {"a", "b"})


@unittest.skipUnless(PREPARED, "run scripts/prepare_software.py --only quixbugs")
class PreparedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.programs = software_data.load_json("quixbugs/repair.json")

    def test_shape(self):
        self.assertEqual(len(self.programs), 31)
        for p in self.programs:
            self.assertGreaterEqual(len(p["generated"]), 10)
            self.assertTrue(p["visible"] and p["held_out"])
            self.assertGreaterEqual(p["failing_tests"], 1)

    @unittest.skipUnless(RAW.is_file(), "raw QuixBugs archive not cached")
    def test_reference_fixes_pass_every_test_and_buggy_programs_fail_a_visible_one(self):
        with zipfile.ZipFile(io.BytesIO(RAW.read_bytes())) as archive:
            for p in self.programs[:8]:
                fixed = archive.read(f"{qb.PREFIX}correct_python_programs/{p['name']}.py").decode()
                with self.subTest(p["name"]):
                    self.assertTrue(all(qb.run_tests(fixed, p["entry"], p["visible"] + p["held_out"] + p["generated"], cpu_seconds=60)))
                    self.assertFalse(all(qb.run_tests(p["source"], p["entry"], p["visible"], cpu_seconds=60)))

    def test_seed_repairs_some_programs(self):
        chosen = [p for p in self.programs if p["name"] in ("knapsack", "find_in_sorted", "flatten")]
        statuses = {p["name"]: qb.judge_patch(p, seed.repair(qb.public_task(p)))["status"] for p in chosen}
        self.assertEqual(statuses["knapsack"], "repaired")
        self.assertEqual(statuses["find_in_sorted"], "plausible")
        self.assertIn(statuses["flatten"], {"unrepaired", "rejected"})

    def test_invalid_candidates(self):
        directory = Path(tempfile.mkdtemp())
        cases = {"missing.py": "def other(task):\n    return ''\n",
                 "not_text.py": "def repair(task):\n    return 42\n"}
        for name, source in cases.items():
            (directory / name).write_text(source)
            with self.subTest(name), self.assertRaises(CandidateError):
                qb.repair_score(load_program(directory / name), cpu_seconds=2)


if __name__ == "__main__":
    unittest.main()
