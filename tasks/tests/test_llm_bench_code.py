"""Offline tests of the code-execution harness used by the llm_bench code benchmarks."""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import llm_bench_code as code  # noqa: E402
from autoresearch_suite import llm_bench_core as core  # noqa: E402
from test_llm_bench import DataDir, FakeLLM  # noqa: E402

CALLS = ["add(1, 2)", "add(-5, 5)", "add(10, 20)"]
EXPECTED = [code.encode(3), code.encode(0), code.encode(30)]


def outputs(source, calls=CALLS, **kwargs):
    return code.run_outputs(source, calls, "call", time_limit=kwargs.get("time_limit", 5.0), per_test=kwargs.get("per_test", 5.0))


class HarnessTests(unittest.TestCase):
    def test_correct_code_passes(self):
        self.assertEqual(outputs("def add(a, b):\n    return a + b\n"), EXPECTED)

    def test_wrong_and_crashing_code(self):
        self.assertEqual(outputs("def add(a, b):\n    return a - b\n")[0], code.encode(-1))
        self.assertEqual(outputs("def add(a, b):\n    raise ValueError()\n"), [None, None, None])
        self.assertEqual(outputs("def add(a, b) return 1"), [None, None, None])

    def test_forged_equality_cannot_pass(self):
        forged = "class Any:\n    def __eq__(self, other):\n        return True\n\ndef add(a, b):\n    return Any()\n"
        self.assertEqual(outputs(forged), [None, None, None])

    def test_stdout_spoofing_is_ignored(self):
        spoof = "import sys, os\nprint('R 0 3')\nos.write(1, b'R 1 0\\n')\ndef add(a, b):\n    print('R 2 30')\n    return -1\n"
        self.assertEqual(outputs(spoof), [-1, -1, -1])

    def test_timeout_keeps_finished_tests(self):
        loop = "def add(a, b):\n    if a == -5:\n        while True:\n            pass\n    return a + b\n"
        start = time.time()
        result = outputs(loop, time_limit=4.0, per_test=1.0)
        self.assertLess(time.time() - start, 8.0)
        self.assertEqual(result, [3, None, 30])

    def test_no_secrets_in_environment(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "secret-key"}):
            leak = "import os\ndef add(a, b):\n    return os.environ.get('OPENROUTER_API_KEY', 'absent')\n"
            self.assertEqual(outputs(leak, calls=["add(1, 2)"]), ["absent"])

    def test_encoding_preserves_equality_semantics(self):
        self.assertNotEqual(code.encode((1, 2)), code.encode([1, 2]))
        self.assertEqual(code.encode({"b": 1, "a": 2}), code.encode({"a": 2, "b": 1}))
        self.assertEqual(code.encode(True), True)
        self.assertEqual(code.encode(2.0), 2)
        with self.assertRaises(TypeError):
            code.encode(object())

    def test_extract_code(self):
        reply = "Plan\n```python\nx = 1\n```\nBetter:\n```python\ndef f():\n    return 2\n```\n"
        self.assertEqual(code.extract_code(reply), "def f():\n    return 2\n")
        self.assertEqual(code.extract_code("def g(): pass"), "def g(): pass")


class LiveCodeBenchTests(unittest.TestCase):
    PROBLEM = {"id": "9", "rank": 0, "release_position": 1, "title": "Add", "difficulty": "easy", "contest_date": "2024-01-01T00:00:00",
               "content": "Add two numbers.", "public": 1, "sica_loads_private": False,
               "tests": [{"input": "1\n2", "output": "3"}, {"input": "5\n5", "output": "10"}, {"input": "[1]\n[2]", "output": "[1, 2]"}]}

    def run_items(self, solve, **params):
        import gzip
        import json as js
        with DataDir({}) as data:
            (data / "livecodebench_sica.json.gz").write_bytes(gzip.compress(js.dumps([self.PROBLEM]).encode()))
            with mock.patch.object(core, "task_llm", return_value=FakeLLM()):
                return code.evaluate(solve, {"benchmark": "livecodebench", "samples": 1, "per_test": 5.0, "time_limit": 20.0, **params})

    def test_fraction_of_tests(self):
        from autoresearch_suite import llm_bench_lcb as lcb
        good = self.run_items(lambda q, llm: "```python\ndef solution(a, b):\n    return a + b\n```")
        self.assertEqual(good["score"], 1.0)
        partial = self.run_items(lambda q, llm: "def solution(a, b):\n    return 3")
        self.assertAlmostEqual(partial["score"], 1 / 3)
        sica = self.run_items(lambda q, llm: "def solution(a, b):\n    return 3", private_tests="sica_loader")
        self.assertEqual(sica["score"], 1.0)
        statement = lcb.statement(self.PROBLEM)
        self.assertIn("solution(1, 2)", statement)
        self.assertNotIn("10", statement)

    def test_sica_comparison(self):
        from autoresearch_suite import llm_bench_lcb as lcb
        self.assertTrue(lcb.matches(code.encode((1, 2)), [1, 2]))
        self.assertTrue(lcb.matches(code.encode([0.1 + 0.2]), [0.3]))
        self.assertTrue(lcb.matches(code.encode(0.30000001), 0.3))
        self.assertTrue(lcb.matches(code.encode(True), 1.0000001))  # as in SICA's template
        self.assertFalse(lcb.matches(code.encode({1: 2}), {"1": 2}))
        self.assertTrue(lcb.matches(code.encode({"a": [1]}), {"a": [1]}))


class ElmRepairTests(unittest.TestCase):
    def test_instances_follow_the_paper(self):
        programs = {p["id"]: p["code"] for p in code.elm_buggy_programs()}
        self.assertEqual(sorted(programs), ["parity-1", "parity-2", "parity-3", "parity-4", "parity-5", "quadratic-1", "quadratic-2"])
        self.assertIn("sum([c1, c2, b3, b4])", programs["parity-2"])
        self.assertIn("bit_sum % 3", programs["parity-5"])
        self.assertIn("def parity(b1, b2, b3, b4):", programs["parity-5"])
        self.assertIn("return a*pow(x,2)-b*x-c", programs["quadratic-2"])

    def test_repairs_are_checked(self):
        fixed = {"parity": "```python\ndef parity(b1, b2, b3, b4):\n    return sum([b1, b2, b3, b4]) % 2\n```",
                 "quadratic": "```python\ndef quadratic(a, b, c, x):\n    return a*x**2 + b*x + c\n```"}
        with mock.patch.object(core, "task_llm", return_value=FakeLLM()):
            good = code.evaluate(lambda q, llm: fixed["parity" if "parity" in q else "quadratic"], {"benchmark": "elm_repair", "trials": 1, "samples": 7})
            echo = code.evaluate(lambda q, llm: q, {"benchmark": "elm_repair", "trials": 1, "samples": 7})
        self.assertEqual((good["score"], good["details"]["items"]), (1.0, 7))
        self.assertEqual(echo["score"], 0.0)


class LeetCodeContestTests(unittest.TestCase):
    ROWS = [{"id": i, "title": f"T{i}", "difficulty": "Easy", "contest": "c", "entry_point": "add", "prompt": "def add(a, b):\n    \"\"\"Add.\"\"\"",
             "tests": [{"call": c, "expected": e} for c, e in zip(CALLS, EXPECTED)]} for i in range(3)]

    def test_pass_at_one(self):
        with DataDir({"leetcode_contest.json": self.ROWS}), mock.patch.object(core, "task_llm", return_value=FakeLLM()):
            good = code.evaluate(lambda q, llm: "```python\ndef add(a, b):\n    return a + b\n```", {"benchmark": "leetcode_contest", "samples": 3})
            partial = code.evaluate(lambda q, llm: "def add(a, b):\n    return 3", {"benchmark": "leetcode_contest", "samples": 3})
        self.assertEqual(good["score"], 1.0)
        self.assertEqual(partial["score"], 0.0)
        self.assertEqual(partial["details"]["per_item"][0]["tests_passed"], 1)


if __name__ == "__main__":
    unittest.main()
