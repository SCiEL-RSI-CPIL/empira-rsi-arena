"""Regression tests for the defects found while porting the llm_bench evaluators (no network, no prepared data, fake clients)."""
from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import llm_bench_code as code  # noqa: E402
from autoresearch_suite import llm_bench_core as core  # noqa: E402
from autoresearch_suite import llm_bench_lcb as lcb  # noqa: E402
from autoresearch_suite import llm_bench_lmx as lmx  # noqa: E402
from autoresearch_suite import llm_bench_pupa as pupa  # noqa: E402
from autoresearch_suite import llm_bench_qa as qa  # noqa: E402
from autoresearch_suite import runtime  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from test_llm_bench import DataDir, FakeLLM  # noqa: E402


class Scripted:
    """A judge or model client that always gives the same reply."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.reply

    def usage(self):
        return {"model": "scripted", "requests": len(self.prompts), "cost_usd": 0.0}


@unittest.skipUnless(hasattr(signal, "setitimer"), "needs POSIX interval timers")
class DeadlineTests(unittest.TestCase):
    """runtime.Deadline raises TimeoutError inside the candidate; run_items used to count it as one crashed item and go on."""

    ROWS = [{"id": i, "question": f"q{i}", "answer": 0.0} for i in range(12)]

    def test_deadline_inside_a_candidate_call_ends_the_evaluation(self):
        calls = []

        def solve(question, llm):
            calls.append(question)
            if len(calls) == 2:
                time.sleep(3)  # the evaluation deadline (1 s) fires here
            return "Answer: 0"

        with tempfile.TemporaryDirectory() as tmp:
            program = Path(tmp) / "candidate.py"
            program.write_text("def solve(question, llm):\n    return ''\n")

            def task(module):
                out = qa.evaluate(solve, {"benchmark": "addsub", "samples": 6, "seed": 0})
                return runtime.success(out["score"], "maximize", valid=None, **out["details"])
            with DataDir({"addsub.json": self.ROWS}), mock.patch.object(core, "task_llm", return_value=FakeLLM()):
                started = time.time()
                result = runtime.run(str(program), task, timeout=1)
        self.assertIsNone(result["score"])
        self.assertFalse(result["valid"])
        self.assertIn("deadline", result["error"])
        self.assertEqual(len(calls), 2)  # no item after the deadline was run
        self.assertLess(time.time() - started, 3.5)

    def test_without_a_deadline_crashes_still_score_zero(self):
        def solve(question, llm):
            raise TimeoutError("an ordinary timeout of the candidate's own")
        with DataDir({"addsub.json": self.ROWS}), mock.patch.object(core, "task_llm", return_value=FakeLLM()):
            out = qa.evaluate(solve, {"benchmark": "addsub", "samples": 6, "seed": 0})
        self.assertEqual((out["score"], out["details"]["items"], out["details"]["crashes"]), (0.0, 6, 6))


class PupaVerdictTests(unittest.TestCase):
    """A judge reply without a 'Judgment:' line used to count as False, and two such replies as an agreement (quality 1)."""

    ROW = {"id": 222, "user_query": "Write to Ms. Yoyo about order 7", "target_response": "Dear Ms. Yoyo, order 7.", "pii_units": ["ms. yoyo"]}

    def quality(self, reply):
        return pupa.judge_item(Scripted(reply), self.ROW, "Hello.", [])["quality"]

    def test_unreadable_verdicts_earn_no_quality(self):
        for reply in ("I cannot decide.", "Response A is polite but", ""):
            self.assertEqual(self.quality(reply), 0.0, reply)

    def test_readable_verdicts_follow_gepa(self):
        self.assertEqual(self.quality("Judgment: yes"), 1.0)
        self.assertEqual(self.quality("Judgment: no"), 1.0)  # both orders say no: GEPA counts the inconsistency as acceptance

        class Alternating(Scripted):
            def __call__(self, prompt, **kwargs):
                self.prompts.append(prompt)
                return "Judgment: no" if len(self.prompts) == 1 else "no verdict"
        self.assertEqual(pupa.judge_item(Alternating(""), self.ROW, "Hello.", [])["quality"], 0.0)

    def test_evaluate_reports_unread_verdicts(self):
        with DataDir({"pupa_test.json": [dict(self.ROW, id=222 + i) for i in range(3)]}), \
                mock.patch.object(core, "task_llm", return_value=Scripted("x")), mock.patch.object(core, "judge_llm", return_value=Scripted("no verdict")), \
                mock.patch.object(pupa.llm_client, "LLM", return_value=Scripted("x")):
            out = pupa.evaluate(lambda query, llm, untrusted: "A reply.", {"samples": 3, "seed": 0})
        self.assertEqual(out["details"]["mean_quality"], 0.0)
        self.assertEqual(out["details"]["unread_verdicts"], 6)
        self.assertEqual(out["score"], 0.5)  # no untrusted call: no leakage, quality 0


class HugeNumberTests(unittest.TestCase):
    """Integers too large for a float used to raise OverflowError in the scorer, which fails the whole evaluation."""

    def test_livecodebench_comparison(self):
        self.assertFalse(lcb.matches(2 ** 2000, 5))
        self.assertFalse(lcb.matches([2 ** 2000, 1], [5, 1]))
        self.assertTrue(lcb.matches(2 ** 2000, 2 ** 2000))
        self.assertTrue(lcb.matches([1, 2], [1.0000001, 2]))
        self.assertTrue(lcb.matches(0.1 + 0.2, 0.3))

    def test_livecodebench_scorer_with_a_huge_result(self):
        problem = {"id": "x", "title": "t", "difficulty": "easy", "content": "c", "public": 1, "tests": [{"input": "3", "output": "6"}, {"input": "4", "output": "8"}]}
        with tempfile.TemporaryDirectory() as tmp:
            import gzip
            (Path(tmp) / "livecodebench_sica.json.gz").write_bytes(gzip.compress(json.dumps([dict(problem, sica_loads_private=True)]).encode()))
            with mock.patch.object(core, "DATA", Path(tmp)):
                items, score = lcb.items({"per_test": 5.0, "time_limit": 10.0})
                result = score(items[0], "```python\ndef solution(n):\n    return 2 ** 2000 if n == 3 else 2 * n\n```")
        self.assertEqual((result["tests_passed"], result["tests"]), (1, 2))

    def test_resocratic_quantities(self):
        item = {"id": "r", "answer": {"x": 1.0}}
        self.assertEqual(qa.score_quantities(item, 'Answer: {"x": 1' + "0" * 400 + "}"), 0.0)
        self.assertEqual(qa.score_quantities(item, 'Answer: {"x": 1.0}'), 1.0)
        self.assertFalse(core.numbers_close(10 ** 400, 1.0))


class CodeHarnessTests(unittest.TestCase):
    CALLS = ["add(1, 2)", "add(-5, 5)"]
    GOOD = "def add(a, b):\n    return a + b\n"

    def outputs(self, source, calls=None, kind="call", prelude=""):
        return code.run_outputs(source, calls or self.CALLS, kind, time_limit=10.0, per_test=5.0, prelude=prelude)

    def test_demonstration_block_is_not_run(self):
        """CoCoEvo exec()s the code in a fresh dict and SICA imports solution.py: a __main__ block never runs there."""
        for block in ("    a, b = map(int, input().split())\n    print(add(a, b))\n", "    import sys\n    print(add(int(sys.argv[1]), int(sys.argv[2])))\n",
                      "    raise SystemExit(1)\n"):
            self.assertEqual(self.outputs(self.GOOD + "\nif __name__ == '__main__':\n" + block), [3, 0], block)

    def test_livecodebench_import_block(self):
        """SICA's scorer puts LiveCodeBench's import block in front of a solution that lacks it."""
        source = "def solution(nums: List[int]) -> int:\n    return max(Counter(nums).values()) + floor(inf if False else 0)\n"
        self.assertEqual(self.outputs(source, ["[1, 1, 2]", "[3]"], kind="lcb"), [None, None])
        self.assertEqual(self.outputs(source, ["[1, 1, 2]", "[3]"], kind="lcb", prelude=lcb.IMPORTS), [2, 1])
        self.assertEqual(self.outputs("from __future__ import annotations\n" + source, ["[3]"], kind="lcb", prelude=lcb.IMPORTS), [1])
        for name in ("string", "re", "datetime", "collections", "heapq", "bisect", "copy", "math", "random", "statistics", "itertools", "functools",
                     "operator", "io", "sys", "json", "builtins", "typing"):
            self.assertIn(f"from {name} import *\n", lcb.IMPORTS)

    @unittest.skipUnless(Path("/proc/self").is_dir(), "needs /proc")
    def test_processes_started_by_the_code_are_stopped(self):
        source = ("import subprocess, sys\n\ndef add(a, b):\n    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
                  "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n    return child.pid\n")
        pid = self.outputs(source, ["add(1, 2)"])[0]
        self.assertIsInstance(pid, int)
        deadline = time.time() + 5
        while time.time() < deadline and Path(f"/proc/{pid}").is_dir() and "zombie" not in self._state(pid):
            time.sleep(0.05)
        alive = Path(f"/proc/{pid}").is_dir() and "zombie" not in self._state(pid)
        if alive:
            os.kill(pid, 9)
        self.assertFalse(alive)

    @staticmethod
    def _state(pid):
        try:
            return Path(f"/proc/{pid}/status").read_text().lower()
        except OSError:
            return "zombie"


class ParserTimeTests(unittest.TestCase):
    """The answer marker and \\boxed{} readers used to take quadratic time on long runs of blanks or unclosed braces."""

    OLD_ANSWER = re.compile(r"(?:final\s+answer|answer)\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?(.*)", re.I)

    def test_long_adversarial_replies_are_read_quickly(self):
        for text in ("answer" + " " * 100_000 + "x", "\\boxed{" * 25_000, "Answer" + " \t" * 50_000 + ": 7"):
            started = time.time()
            core.final_answer(text)
            self.assertLess(time.time() - started, 2.0)
        self.assertEqual(core.final_answer("Answer" + " " * 50_000 + ": 7"), "7")

    def test_same_matches_as_before(self):
        rng = random.Random(0)
        alphabet = ["answer", "Answer", "final answer", " ", "\t", "\n", "**", ":", "：", "x", "7", "\\boxed{", "{", "}"]
        for _ in range(3000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
            self.assertEqual(core._ANSWER.findall(text), self.OLD_ANSWER.findall(text), repr(text))
            self.assertEqual(core.last_boxed(text), self._old_last_boxed(text), repr(text))

    @staticmethod
    def _old_last_boxed(text):
        start = text.rfind("\\boxed{")
        while start != -1:
            depth, i = 0, start + len("\\boxed")
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i + 1:j]
            start = text.rfind("\\boxed{", 0, start)
        return None


class LmxInputTests(unittest.TestCase):
    def test_unencodable_sentence_is_a_contract_violation(self):
        with self.assertRaises(CandidateError):
            lmx.check_sentences(["a sentence with \ud800 inside"], 10)
        self.assertEqual(lmx.check_sentences(["  An ordinary sentence. "], 10), ["An ordinary sentence."])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_sentence_longer_than_the_classifier_reads(self):
        import torch

        class Tokenizer:
            def __call__(self, sentence, return_tensors=None):
                return {"input_ids": torch.zeros((1, 4 * len(sentence)), dtype=torch.long)}

        class Classifier:
            config = type("Config", (), {"max_position_embeddings": 514})()

            def __call__(self, **encoded):
                raise AssertionError("the classifier must not see an over-long input")
        with mock.patch.object(lmx, "models", return_value=(Tokenizer(), Classifier(), 2, None)):
            with self.assertRaises(CandidateError):
                lmx.positive_probability("\U0001F600" * 200)


if __name__ == "__main__":
    unittest.main()
