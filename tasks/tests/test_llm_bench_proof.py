"""Offline tests of the IMO-ProofBench grading path (scripted judge)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import llm_bench_core as core  # noqa: E402
from autoresearch_suite import llm_bench_proof as proof  # noqa: E402
from test_llm_bench import DataDir, FakeLLM  # noqa: E402

ROWS = [{"id": f"PB-Basic-00{i}", "problem": f"Prove claim {i}.", "solution": "Reference proof.", "guidelines": "(Partial) lemma.",
         "category": "Algebra", "level": "IMO-easy", "source": "test"} for i in range(1, 4)]


class ProofTests(unittest.TestCase):
    def test_points_parsing(self):
        self.assertEqual(proof.parse_points("analysis <points>6 out of 7</points> and later <points>7 out of 7</points>"), (6, "tag"))
        self.assertEqual(proof.parse_points("<points> 0 out of 7 </points>"), (0, "tag"))
        self.assertEqual(proof.parse_points("It earns 1 out of 7 at first glance.\nThe score is 0 out of 7"), (0, "phrase"))
        self.assertEqual(proof.parse_points("<points>9 out of 7</points>"), (None, None))
        self.assertEqual(proof.parse_points("seven points"), (None, None))

    def test_grading_and_aggregates(self):
        judge = FakeLLM("The proof is complete. <points>7 out of 7</points>")
        with DataDir({"proofbench.json": ROWS}), mock.patch.object(core, "task_llm", return_value=FakeLLM("A proof.")), \
                mock.patch.object(core, "judge_llm", return_value=judge):
            out = proof.evaluate(lambda q, llm: llm(q), {"samples": 3, "seed": 0})
            empty = proof.evaluate(lambda q, llm: "   ", {"samples": 3, "seed": 0})
        self.assertEqual((out["score"], out["details"]["pass_at_7"]), (1.0, 1.0))
        self.assertIn("Reference proof.", judge.prompts[0])
        self.assertIn("(Partial) lemma.", judge.prompts[0])
        self.assertEqual(empty["score"], 0.0)

    def test_unparsable_grade_counts_zero(self):
        result = proof.grade(FakeLLM("It is fine."), ROWS[0], "A proof.")
        self.assertEqual((result["score"], result["parsed"]), (0.0, False))


if __name__ == "__main__":
    unittest.main()
