"""Offline tests of the PUPA judge pipeline (scripted judge and model clients)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import llm_bench_core as core  # noqa: E402
from autoresearch_suite import llm_bench_pupa as pupa  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from test_llm_bench import DataDir, FakeLLM  # noqa: E402

ROWS = [{"id": 222 + i, "user_query": f"Email Ms. Yoyo about order {i}", "target_response": f"Dear Ms. Yoyo, order {i}.", "pii_units": ["ms. yoyo", "yoyo"]}
        for i in range(3)]


class ScriptedJudge(FakeLLM):
    """Says 'yes' to quality questions and counts the PII units whose text occurs in the prompt section."""

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if prompt.startswith("You compare"):
            return "Reasoning.\nJudgment: yes"
        units = eval(prompt.split("Information pieces: ")[1].split("\n")[0])  # noqa: S307 - JSON list written by the evaluator
        sent = prompt.split("Prompt sent to the external model:\n")[1].split("\n\nCount the pieces")[0].lower()
        return f"Count: {sum(unit in sent for unit in units)}"


class PupaTests(unittest.TestCase):
    def evaluate(self, solve):
        judge = ScriptedJudge()
        with DataDir({"pupa_test.json": ROWS}), mock.patch.object(core, "task_llm", return_value=FakeLLM("trusted reply")), \
                mock.patch.object(core, "judge_llm", return_value=judge), mock.patch.object(pupa.llm_client, "LLM", return_value=FakeLLM("external reply")):
            return pupa.evaluate(solve, {"samples": 3, "seed": 0}), judge

    def test_redacting_candidate_scores_full_marks(self):
        out, _ = self.evaluate(lambda query, llm, untrusted: untrusted("Write a polite order email.") and "Dear Ms. Yoyo, ...")
        self.assertEqual((out["score"], out["details"]["mean_leakage"]), (1.0, 0.0))

    def test_leaking_candidate_is_penalised(self):
        out, judge = self.evaluate(lambda query, llm, untrusted: untrusted(query))
        self.assertEqual(out["details"]["mean_leakage"], 1.0)
        self.assertEqual(out["score"], 0.5)
        self.assertTrue(any("Prompt sent to the external model" in p for p in judge.prompts))

    def test_no_untrusted_call_means_no_leakage_judgement(self):
        out, judge = self.evaluate(lambda query, llm, untrusted: llm(query))
        self.assertEqual(out["details"]["mean_leakage"], 0.0)
        self.assertFalse(any("Prompt sent" in p for p in judge.prompts))

    def test_recorder_captures_messages_and_system_prompts(self):
        recorder = pupa.Recorder(FakeLLM())
        recorder([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], system="c")
        self.assertEqual(recorder.prompts, ["a", "b", "c"])

    def test_parsers_and_contract(self):
        self.assertTrue(pupa._yes("... Judgment: **Yes**"))
        self.assertFalse(pupa._yes("I would say yes but\nJudgment: no"))
        self.assertEqual(pupa._count("Count: 2"), 2)
        self.assertIsNone(pupa._count("two"))
        with self.assertRaises(CandidateError):
            self.evaluate(lambda query, llm, untrusted: None)
        with self.assertRaises(CandidateError):
            self.evaluate(lambda query, llm: "two-argument solve")


if __name__ == "__main__":
    unittest.main()
