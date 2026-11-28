"""Loop invariants on Code2Inv: term whitelist, verification conditions, counterexamples, oracle budgets and the seed."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite/_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import software_data, software_invariants as inv  # noqa: E402
from autoresearch_suite.runtime import CandidateError, load_program  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

HAVE_Z3 = importlib.util.find_spec("z3") is not None
PREPARED = (software_data.data_dir() / "code2inv/problems.json").is_file()


class Parsing(unittest.TestCase):
    def test_canonical_form(self):
        self.assertEqual(inv.parse(" (and (>= x 1)\n(<= (+ x (* 2 y)) (- 5)))", ["x", "y"]), "(and (>= x 1) (<= (+ x (* 2 y)) (- 5)))")
        self.assertEqual(inv.parse("(= (mod x 2) 0)", ["x"]), "(= (mod x 2) 0)")
        self.assertEqual(inv.parse("(ite (> x 0) (= y 1) true)", ["x", "y"]), "(ite (> x 0) (= y 1) true)")

    def test_rejections(self):
        bad = ["(>= x 1", "(>= x 1))", "(>= z 1)", "(>= x -1)", "(* x y)", "(>= (* x y) 0)", "(= (mod x y) 0)",
               "(+ x 1)", "(and (>= x 1) 3)", "(assert false)", "true) (assert false) (define-fun g () Bool (and true",
               "(>= x 10000000000)", "(= x true)", 42, "(" * 70 + "x" + ")" * 70]
        for text in bad:
            with self.subTest(text=str(text)[:40]), self.assertRaises(CandidateError):
                inv.parse(text, ["x", "y"])


@unittest.skipUnless(HAVE_Z3 and PREPARED, "needs z3-solver and scripts/prepare_software.py --only invariants")
class Verification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems = {p["name"]: p for p in software_data.load_json("code2inv/problems.json")}

    def test_invalid_programs_match_the_published_count(self):
        invalid = software_data.load_json("code2inv/invalid.json")
        self.assertEqual(len(self.problems), 124)
        self.assertEqual(sorted(int(p["name"]) for p in invalid), [26, 27, 31, 32, 61, 62, 72, 75, 106])

    def test_hand_written_invariant_and_counterexamples(self):
        p = self.problems["1"]  # x = 1, y = 0; while (y < 100000) {x += y; y += 1}; assert x >= y
        self.assertTrue(inv.check(p, "(and (>= x 1) (>= y 0) (>= x y))")["valid"])
        weak = inv.check(p, "(>= x y)")  # not inductive without x >= 1
        self.assertEqual(weak["failed"], "consecution")
        s, n = weak["state"], weak["next"]
        self.assertTrue(s["x"] >= s["y"] and n["x"] < n["y"])
        self.assertEqual((n["x"], n["y"]), (s["x"] + s["y"], s["y"] + 1))
        start = inv.check(p, "(> x 1)")
        self.assertEqual((start["failed"], start["state"]), ("initiation", {"x": 1, "y": 0}))
        self.assertEqual(inv.check(p, "true")["failed"], "safety")

    def test_oracle_budget(self):
        oracle = inv._Oracle([self.problems["1"]], 2)
        self.assertEqual(oracle({"problem": 0, "invariant": "(x >= 1)"})["failed"], "syntax")
        self.assertEqual(oracle({"problem": 0, "invariant": "true"})["queries_left"], 0)
        self.assertEqual(oracle({"problem": 0, "invariant": "true"})["failed"], "query budget exhausted")
        with self.assertRaises(CandidateError):
            oracle({"problem": 5, "invariant": "true"})

    def test_seed_and_invalid_candidates(self):
        directory = Path(tempfile.mkdtemp())
        (directory / "seed.py").write_text(seed_source("science-software-loop-invariants-code2inv"))
        score, details = inv.invariant_score(load_program(directory / "seed.py"), cpu_seconds=120)
        self.assertEqual(details["verified"], 63)
        (directory / "true.py").write_text("def synthesize(problem, oracle):\n    return 'true'\n")
        trivial = inv.invariant_score(load_program(directory / "true.py"))[1]
        # five programs have post-conditions that hold at every loop exit (for example 'c < 0 and c > 40 implies ...')
        self.assertEqual(trivial["verified"], 5)
        (directory / "short.py").write_text("def synthesize(problem, oracle):\n    raise ValueError('no')\n")
        with self.assertRaises(CandidateError):
            inv.invariant_score(load_program(directory / "short.py"))


if __name__ == "__main__":
    unittest.main()
