"""Finance and economics candidates run in the sandbox: nothing of the scoring loop is readable."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "suite/_lib")]
from autoresearch_suite import finance_fairdiv, finance_sandbox, finance_school, runtime
from autoresearch_tasks.registry import get_task
from autoresearch_tasks.runner import evaluate_suite

SECRETS = [37, 5, 91, 12, 64, 28]
# Looks for the realised values in the frames above it and in every suspended scoring loop of its
# process; answers -1 when there is nothing to read.
LOOP_READER = '''
import gc
import sys
import types

def guess(step, previous):
    frames = []
    frame = sys._getframe(1)
    while frame is not None:
        frames.append(frame)
        frame = frame.f_back
    frames += [item.gi_frame for item in gc.get_objects() if isinstance(item, types.GeneratorType) and item.gi_frame]
    for frame in frames:
        if "secrets" in frame.f_locals:
            return frame.f_locals["secrets"][step]
    return -1
'''
# Uses the harness's channel to hand in a throwaway answer and fetch the next call early.
EARLY_FETCH = '''
import sys

def guess(step, previous):
    oracle = sys._getframe(1).f_locals["oracle"]
    following = oracle({"op": "result", "value": -1})
    return -1 if following is None else following["args"][1]
'''
IMPORTS_EVALUATOR = '''
def allocate(values):
    from autoresearch_suite import finance_fairdiv
    return [0] * values.shape[1]
'''
READS_A_FILE = '''
def allocate(values):
    open(%r).read()
    return [0] * values.shape[1]
'''


def rounds(secrets):
    """A scoring loop of the kind the forecasting tasks use: a step reveals the previous realised value."""
    answers = []
    for step in range(len(secrets)):
        answers.append((yield finance_sandbox.step("guess", step, secrets[step - 1] if step else None)))
    return answers


class FinanceSandboxTests(unittest.TestCase):
    def candidate(self, source):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "candidate.py"
        path.write_text(source)
        return runtime.load_program(path)

    def test_the_scoring_loop_is_not_in_the_candidate_process(self):
        module = self.candidate(LOOP_READER)
        # In this process the same code reads every realised value: this is the leak the sandbox closes.
        self.assertEqual(finance_sandbox.run(module.guess, rounds(SECRETS), "guess"), SECRETS)
        self.assertEqual(finance_sandbox.run(module, rounds(SECRETS), "guess"), [-1] * len(SECRETS))

    def test_a_value_is_bound_before_the_next_call_is_released(self):
        module = self.candidate(EARLY_FETCH)
        try:
            answers = finance_sandbox.run(module, rounds(SECRETS), "guess")
        except runtime.CandidateError:
            return  # the loop ended while the candidate still had calls to answer
        self.assertFalse(any(answer == secret for answer, secret in zip(answers, SECRETS)))

    def test_evaluator_package_and_files_are_out_of_reach(self):
        with self.assertRaisesRegex(runtime.CandidateError, "autoresearch_suite"):
            finance_fairdiv.fair_division_score(self.candidate(IMPORTS_EVALUATOR))
        with self.assertRaisesRegex(runtime.CandidateError, "file access denied"):
            finance_fairdiv.fair_division_score(self.candidate(READS_A_FILE % str(ROOT / "catalog/tasks.json")))

    def test_missing_entry_point_and_bad_values_are_the_candidates_failure(self):
        with self.assertRaisesRegex(runtime.CandidateError, "must define allocate"):
            finance_fairdiv.fair_division_score(self.candidate("def other():\n    return 1\n"))
        with self.assertRaisesRegex(runtime.CandidateError, "agent ind"):
            finance_fairdiv.fair_division_score(self.candidate("def allocate(values):\n    return [0.5]\n"))

    def test_sandboxed_and_trusted_routes_give_the_same_score(self):
        for task_id, score, entry in (("science-economics-maximum-nash-welfare", finance_fairdiv.fair_division_score, "allocate"),
                                      ("science-economics-school-choice-tie-breaking", finance_school.school_score, "assign_students")):
            with self.subTest(task=task_id):
                module = runtime.load_program(ROOT / "suite" / task_id / "initial_program.py")
                sandboxed, trusted = score(module), score(getattr(module, entry))
                self.assertEqual(sandboxed[0], trusted[0])
                self.assertEqual(json.dumps(sandboxed[1], sort_keys=True), json.dumps(trusted[1], sort_keys=True))

    def test_seed_scores_are_the_recorded_ones(self):
        rows = {row["id"]: row for row in json.loads((ROOT / "validation/suite-additions-report.json").read_text())["tasks"]}
        for task_id in ("science-economics-maximum-nash-welfare", "science-economics-school-choice-tie-breaking"):
            with self.subTest(task=task_id):
                result = evaluate_suite(ROOT, get_task(ROOT, task_id), timeout=120)
                self.assertEqual((result["status"], result["valid"]), ("completed", True))
                self.assertEqual(result["score"], rows[task_id]["score"])

    def test_every_finance_evaluator_hands_over_the_module(self):
        wrappers = sorted(path for path in (ROOT / "suite").glob("science-*/evaluator.py")
                          if "import finance_" in path.read_text())
        self.assertEqual(len(wrappers), 18)
        for path in wrappers:
            text = path.read_text()
            self.assertRegex(text, r"finance_\w+\.\w+\(module[,)]", path.parent.name)
            self.assertNotRegex(text, r"finance_\w+\.\w+\(module\.", path.parent.name)


if __name__ == "__main__":
    unittest.main()
