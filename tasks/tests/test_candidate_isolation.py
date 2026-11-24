"""Sandboxed evaluators never execute the candidate in their own process."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "suite/_lib")]
from autoresearch_suite import runtime, sandboxed

# Library modules that run candidate code in a child process, and the wrappers that use them.
SANDBOX = re.compile(r"software_sandbox|chem_sandbox|earth_sandbox|bio_physics_common|finance_sandbox|run_sandboxed")
LIBRARY = ROOT / "suite/_lib/autoresearch_suite"
SANDBOXED = {path.stem for path in LIBRARY.glob("*.py") if SANDBOX.search(path.read_text())}
NASH = "science-economics-maximum-nash-welfare"
# What a hostile candidate would do while it is imported: replace the scoring function of the evaluator.
TAMPER = '''
try:
    from autoresearch_suite import finance_fairdiv, runtime
    runtime.TAMPERED = True
    finance_fairdiv.fair_division_score = lambda candidate: (1.0, {})
except Exception:
    pass
'''
POOR = '''
def allocate(values):
    return [0] * values.shape[1]
'''


def wrapper(task):
    """The package's evaluator.py as a module."""
    spec = importlib.util.spec_from_file_location(f"evaluator_{task.replace('-', '_')}", ROOT / "suite" / task / "evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def imported(text):
    return {name.strip() for group in re.findall(r"from autoresearch_suite import ([\w, ]+)", text) for name in group.split(",")}


class CandidateIsolationTests(unittest.TestCase):
    def candidate(self, source):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "candidate.py"
        path.write_text(source)
        return path

    def test_candidate_code_is_not_executed_by_the_evaluator(self):
        evaluate = wrapper(NASH).evaluate
        self.addCleanup(lambda: runtime.__dict__.pop("TAMPERED", None))
        honest = evaluate(str(self.candidate(POOR)))
        hostile = evaluate(str(self.candidate(TAMPER + POOR)))
        self.assertFalse(hasattr(runtime, "TAMPERED"))
        self.assertEqual((honest["valid"], hostile["valid"]), (True, True))
        self.assertEqual(hostile["score"], honest["score"])
        self.assertLess(hostile["score"], 1.0)

    def test_the_same_candidate_would_win_if_it_were_imported(self):
        # The leak this closes, shown on a copy of the scoring module's namespace: importing runs the tampering.
        from autoresearch_suite import finance_fairdiv
        original = finance_fairdiv.fair_division_score
        self.addCleanup(lambda: (setattr(finance_fairdiv, "fair_division_score", original), runtime.__dict__.pop("TAMPERED", None)))
        runtime.load_program(self.candidate(TAMPER + POOR))
        self.assertTrue(getattr(runtime, "TAMPERED", False))
        self.assertIsNot(finance_fairdiv.fair_division_score, original)

    def test_an_unexecuted_candidate_carries_only_its_path(self):
        path = self.candidate("SECRET = 1\n\ndef allocate(values):\n    return []\n")
        module = sandboxed.source_program(path)
        self.assertEqual(module.__file__, str(path.resolve()))
        self.assertFalse(hasattr(module, "SECRET") or hasattr(module, "allocate"))
        self.assertTrue(sandboxed.defines(module, "allocate"))
        self.assertFalse(sandboxed.defines(module, "forecast"))
        with self.assertRaisesRegex(runtime.CandidateError, "not found"):
            sandboxed.source_program(path.with_name("missing.py"))

    def test_entry_points_are_found_in_the_source(self):
        cases = {"def solve(x):\n    return x\n": True, "import math\nsolve = math.sqrt\n": True,
                 "from math import sqrt as solve\n": True, "class solve:\n    pass\n": True,
                 "if True:\n    def solve(x):\n        return x\n": True, "def other(x):\n    return x\n": False}
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertIs(sandboxed.defines(sandboxed.source_program(self.candidate(source)), "solve"), expected)
        with self.assertRaisesRegex(runtime.CandidateError, "does not parse"):
            sandboxed.defines(sandboxed.source_program(self.candidate("def solve(:\n")), "solve")
        loaded = runtime.load_program(self.candidate("def solve(x):\n    return x\nvalue = 3\n"))
        self.assertTrue(sandboxed.defines(loaded, "solve"))
        self.assertFalse(sandboxed.defines(loaded, "value"))

    def test_a_missing_entry_point_is_the_candidates_failure(self):
        result = wrapper(NASH).evaluate(str(self.candidate("def other():\n    return 1\n")))
        self.assertIsNone(result["score"])
        self.assertIn("must define allocate", result["error"])

    def test_every_sandboxed_wrapper_loads_the_candidate_unexecuted(self):
        wrappers = [path for path in sorted((ROOT / "suite").glob("*/evaluator.py")) if imported(path.read_text()) & SANDBOXED]
        self.assertGreaterEqual(len(wrappers), 85)
        for path in wrappers:
            text = path.read_text()
            self.assertIn("sandboxed.run(program_path, task", text, path.parent.name)
            self.assertNotIn("runtime.run(", text, path.parent.name)


if __name__ == "__main__":
    unittest.main()
