"""Offline tests of the llm_tools HoVer retrieval evaluator (tiny corpus, scripted model)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import llm_tools_common as common  # noqa: E402
from autoresearch_suite import llm_tools_hover as hover  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

HOVER_SEED = seed_source("mipro-hover")


class FakeLLM:
    def __init__(self, reply="query"):
        self.reply, self.prompts = reply, []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.reply

    def usage(self):
        return {"model": "fake", "requests": len(self.prompts), "cost_usd": 0.0}


def module_from(source):
    path = Path(tempfile.mkdtemp()) / "candidate.py"
    path.write_text(source)
    module = type(sys)("candidate")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


ABSTRACTS = [{"title": "Eiffel Tower", "text": ["The Eiffel Tower is a tower in Paris."]},
             {"title": "Gustave Eiffel", "text": ["Gustave Eiffel was a French engineer."]},
             {"title": "Paris", "text": ["Paris is the capital of France."]},
             {"title": "Berlin", "text": ["Berlin is the capital of Germany."]}] + \
            [{"title": f"Filler {i}", "text": [f"Filler article number {i} about rivers."]} for i in range(12)]
CLAIMS = {"test": [{"uid": f"c{i}", "claim": "The tower in Paris was designed by a French engineer.", "label": "SUPPORTED", "num_hops": 3,
                    "gold_titles": ["Eiffel Tower", "Gustave Eiffel", "Paris"]} for i in range(3)]}


@unittest.skipUnless(importlib.util.find_spec("bm25s"), "bm25s is not installed (the HoVer retriever index)")
class HoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "hover").mkdir()
        source = root / "abstracts.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in ABSTRACTS))
        subprocess.run([sys.executable, "-m", "autoresearch_suite.llm_tools_hover", str(source), str(root / "hover" / "index")], check=True,
                       env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}, cwd=str(ROOT / "suite" / "_lib"),
                       stdout=subprocess.DEVNULL)
        (root / "hover" / "claims.json").write_text(json.dumps(CLAIMS))
        cls.patch = mock.patch.object(common, "data_dir", return_value=root)
        cls.patch.start()
        hover._RETRIEVER.clear()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        hover._RETRIEVER.clear()
        cls.tmp.cleanup()

    def test_normalisation_matches_dspy(self):
        self.assertEqual(hover.normalize_text("The,  Eiffel  Tower!"), "eiffel tower")
        self.assertEqual(hover.solved(["The Eiffel Tower"], ["eiffel tower | text"]), (True, 1, 1))

    def test_search_returns_title_documents(self):
        found = hover.bm25_search("French engineer Eiffel", 2)
        self.assertEqual(found[0], "Gustave Eiffel | Gustave Eiffel was a French engineer.")
        self.assertEqual(len(found), 2)

    def test_budget_is_enforced(self):
        budget = hover.SearchBudget(lambda query, k: [f"{query}{i} | x" for i in range(k)])
        budget("a", 10)
        budget("b", 10)
        with self.assertRaises(common.ProxyError):
            budget("c", 5)  # 25 documents > 24
        budget("c", 4)
        with self.assertRaises(common.ProxyError):
            budget("d", 1)  # fourth call
        with self.assertRaises(common.ProxyError):
            hover.SearchBudget()("x", 11)

    def test_seed_and_cheater(self):
        P = {"samples": 3, "seed": 0}
        score, details = hover.evaluate(module_from(HOVER_SEED), P, client=FakeLLM("Gustave Eiffel engineer"))
        self.assertEqual(details["items"], 3)
        self.assertEqual(score, 1.0)  # the first hop (k=7) already ranks the three gold articles first
        cheat = "def solve(claim, search, llm):\n    return ['Eiffel Tower | x', 'Gustave Eiffel | y', 'Paris | z']\n"
        score, details = hover.evaluate(module_from(cheat), P, client=FakeLLM())
        self.assertEqual(score, 0.0)  # only documents returned by search count
        greedy = "def solve(claim, search, llm):\n    for _ in range(5):\n        search(claim, 10)\n"
        score, details = hover.evaluate(module_from(greedy), P, client=FakeLLM())
        self.assertEqual(details["crashes"], 3)  # the third search call exceeds the 24-document budget and the error is not caught

    def test_candidate_cannot_read_gold_titles(self):
        reader = ("def solve(claim, search, llm):\n    import pathlib\n"
                  f"    return pathlib.Path({str(common.data_dir() / 'hover' / 'claims.json')!r}).read_text()\n")
        score, details = hover.evaluate(module_from(reader), {"samples": 1}, client=FakeLLM())
        self.assertIn("denied", details["per_item"][0]["crash"])


if __name__ == "__main__":
    unittest.main()
