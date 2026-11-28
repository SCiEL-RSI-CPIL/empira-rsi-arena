"""IFBench checker integration (needs `python scripts/prepare_llm_bench.py`; skipped when the files are absent)."""
from __future__ import annotations

import json
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import llm_bench_core as core  # noqa: E402
from autoresearch_suite import llm_bench_ifbench as ifb  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

PREPARED = (core.DATA / "ifbench_checkers" / "instructions.py").is_file() and (core.DATA / "ifbench_test.json").is_file()


def no_network(*args, **kwargs):
    raise OSError("network access during IFBench scoring")


class VariantTests(unittest.TestCase):
    def test_loose_variants(self):
        variants = ifb.response_variants("Sure!\n**Body** text\nHope this helps")
        self.assertEqual(len(variants), 8)
        self.assertIn("**Body** text", variants)
        self.assertIn("Body text", variants)


@unittest.skipUnless(PREPARED, "IFBench files not prepared")
class CheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {row["id"]: row for row in json.loads((core.DATA / "ifbench_test.json").read_text())}

    def check(self, key, response):
        with mock.patch.object(socket.socket, "connect", no_network):
            return ifb.followed(self.rows[key], response)

    def test_keyword_counts(self):
        good = ("kaleidoscope nebula nebula whisper whisper whisper " + "labyrinth " * 5 + "paradox " * 7).strip()
        self.assertEqual(self.check("0", good)[:2], (1, 1))
        self.assertEqual(self.check("0", "kaleidoscope nebula")[:2], (0, 1))

    def test_word_count_range_and_loose_rule(self):
        body = " ".join(["word"] * 72)
        self.assertEqual(self.check("41", body)[:2], (1, 1))
        self.assertEqual(self.check("41", "Here is my answer:\n" + body)[:2], (1, 1))  # first line dropped by the loose rule
        self.assertEqual(self.check("41", " ".join(["word"] * 80))[:2], (0, 1))

    def test_person_names_uses_local_spacy_model(self):
        names = "Emma, Liam, Sophia, Jackson, Olivia, Noah, Ava, Lucas, Isabella and Mason"
        self.assertEqual(self.check("25", f"At the party I met {names}. Everyone had fun.")[:2], (1, 1))
        self.assertEqual(self.check("25", "At the party I met Emma.")[:2], (0, 1))

    def test_pos_tagger_resource(self):
        self.assertEqual(self.check("257", "Go to the store.")[:2], (1, 1))
        self.assertEqual(self.check("257", "An empire of endless suffering minds.")[:2], (0, 1))

    def test_evaluate_with_fake_client(self):
        class Fake:
            def __call__(self, prompt, **kwargs):
                return "ok"

            def usage(self):
                return {"cost_usd": 0.0}
        with mock.patch.object(core, "task_llm", return_value=Fake()), mock.patch.object(socket.socket, "connect", no_network):
            out = ifb.evaluate(lambda question, llm: llm(question), {"samples": 3, "seed": 0})
            self.assertEqual(out["details"]["items"], 3)
            self.assertTrue(0.0 <= out["score"] <= 1.0)
            with self.assertRaises(CandidateError):
                ifb.evaluate(lambda question, llm: None, {"samples": 3, "seed": 0})


if __name__ == "__main__":
    unittest.main()
