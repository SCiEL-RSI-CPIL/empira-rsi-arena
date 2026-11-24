"""Register coalescing: challenge-format parsing, colouring certification, published costs and the seed."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite/_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import software_coalesce as rc, software_data  # noqa: E402
from autoresearch_suite.runtime import CandidateError, load_program  # noqa: E402
from suite_test_support import seed_module  # noqa: E402

# The seed is the package's: the seed module of the source repository is not in this library.
seed = seed_module("science-software-coalescing-small")

PREPARED = all((software_data.data_dir() / f"coalesce/{g}.json").is_file() for g in rc.GROUPS)
# 3 registers (nodes 0-2 precoloured), nodes 3-5; a comment, a duplicate pair in both directions and a self-affinity.
TINY = """6 3 % comment
3 0 -1  3 4 -1  4 5 -1
3 4 7  4 3 1  3 5 2  4 1 5  5 5 9
"""


class Format(unittest.TestCase):
    def setUp(self):
        self.problem = rc.parse_problem(TINY, "tiny")

    def test_parsing_merges_pairs_and_drops_self_affinities(self):
        self.assertEqual(self.problem["interference"], [[0, 3], [3, 4], [4, 5]])
        self.assertEqual(self.problem["affinities"], [[1, 4, 5], [3, 4, 8], [3, 5, 2]])

    def test_cost_matches_the_definition(self):
        self.assertEqual(rc.cost(self.problem, [0, 1, 2, 1, 2, 1]), 8 + 5)  # (3,4) split, (1,4) split, (3,5) joined
        self.assertEqual(rc.cost(self.problem, [0, 1, 2, 2, 1, 2]), 8)

    def test_illegal_colourings_are_rejected(self):
        for colours, message in (([0, 1, 2, 0, 1, 2], "interfering"), ([1, 0, 2, 1, 2, 1], "precoloured"),
                                 ([0, 1, 2, 1, 2], "list of 6"), ([0, 1, 2, 1, 2, 3], "0..2"),
                                 ([0, 1, 2, 1, 2, True], "ints"), ([0, 1, 2, 1.0, 2, 1], "ints")):
            with self.subTest(colours=colours), self.assertRaisesRegex(CandidateError, message):
                rc.cost(self.problem, colours)

    def test_degree_condition_is_enforced(self):
        with self.assertRaises(ValueError):
            rc.parse_problem("4 2\n2 0 -1 2 1 -1 2 3 -1\n", "bad")

    def test_seed_colouring_is_legal(self):
        colours = seed.coalesce(rc.public_problem(self.problem))
        self.assertLessEqual(rc.cost(self.problem, colours), 15)


@unittest.skipUnless(PREPARED, "run scripts/prepare_software.py --only coalesce")
class PreparedInstances(unittest.TestCase):
    def test_groups_and_reference_values(self):
        small, large = (software_data.load_json(f"coalesce/{g}.json") for g in rc.GROUPS)
        self.assertEqual((len(small), len(large)), (452, 22))
        self.assertTrue(all(p["best_status"] == "optimal" for p in small))
        self.assertEqual(sorted(p["name"] for p in large if p["best_status"] != "optimal"), ["139", "304", "387"])
        for p in small + large:
            self.assertLessEqual(p["best_cost"], min(p["published"].values()))
            self.assertGreater(p["weight"], 0)

    def test_seed_score_and_invalid_candidates(self):
        directory = Path(tempfile.mkdtemp())
        (directory / "seed.py").write_text(Path(seed.__file__).read_text())
        score, details = rc.coalesce_score(load_program(directory / "seed.py"), "small", cpu_seconds=60)
        self.assertGreater(score, details["best_known_score"])
        self.assertLess(score, 0.5)
        cases = {"clash.py": "def coalesce(problem):\n    return [i if i < problem['registers'] else 0 for i in range(problem['nodes'])]\n",
                 "short.py": "def coalesce(problem):\n    return [0]\n",
                 "missing.py": "def colour(problem):\n    return []\n"}
        for name, source in cases.items():
            (directory / name).write_text(source)
            with self.subTest(name), self.assertRaises(CandidateError):
                rc.coalesce_score(load_program(directory / name), "large", cpu_seconds=30)


if __name__ == "__main__":
    unittest.main()
