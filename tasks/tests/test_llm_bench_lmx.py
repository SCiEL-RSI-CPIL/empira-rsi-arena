"""Tests of the LMX sentence-sentiment scorer (model calls replaced by fixed functions)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import llm_bench_core as core  # noqa: E402
from autoresearch_suite import llm_bench_lmx as lmx  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from test_llm_bench import FakeLLM  # noqa: E402

FITNESS = {"base quote here": 0.1, "happy variant": 0.9, "far happy text": 0.95, "far sad text": 0.2}
POSITION = {"base quote here": 0.0, "happy variant": 0.5, "far happy text": 2.0, "far sad text": 1.3}


def fake_embedding(sentence):
    return np.array([POSITION[sentence], 0.0])


class LmxTests(unittest.TestCase):
    def test_niche_mapping(self):
        self.assertEqual(lmx.niche(0.0, 30, 1.25), 0)
        self.assertEqual(lmx.niche(0.5, 30, 1.25), 12)
        self.assertEqual(lmx.niche(1.25, 30, 1.25), 29)
        self.assertEqual(lmx.niche(9.0, 30, 1.25), 29)

    def test_qd_score_keeps_best_per_niche(self):
        with mock.patch.object(lmx, "positive_probability", FITNESS.get), mock.patch.object(lmx, "embedding", fake_embedding):
            score, details = lmx.qd_score("base quote here", list(FITNESS))
        self.assertAlmostEqual(score, 0.1 + 0.9 + 0.95)  # the two far sentences share the last niche
        self.assertEqual(details["niches_filled"], 3)
        self.assertEqual(details["archive"]["29"]["sentence"], "far happy text")

    def test_validation(self):
        for bad in ([], "a sentence", [1], ["ok sentence", "no"], ["line one\nline two"], ["x" * 401]):
            with self.assertRaises(CandidateError):
                lmx.check_sentences(bad, 300)
        with self.assertRaises(CandidateError):
            lmx.check_sentences(["valid sentence"] * 301, 300)
        self.assertEqual(lmx.check_sentences(["  padded sentence  "], 300), ["padded sentence"])

    def test_evaluate_passes_quote_and_optional_client(self):
        seen = []
        with mock.patch.object(lmx, "positive_probability", FITNESS.get), mock.patch.object(lmx, "embedding", fake_embedding), \
                mock.patch.object(core, "task_llm", return_value=FakeLLM()):
            out = lmx.evaluate(lambda q, llm: seen.append(q) or ["happy variant"], {"quote": "base quote here"})
        self.assertEqual(seen, ["base quote here"])
        self.assertAlmostEqual(out["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
