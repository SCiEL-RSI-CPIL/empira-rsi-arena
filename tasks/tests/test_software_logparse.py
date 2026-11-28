"""Log parsing: grouping accuracy and pairwise F1 semantics, prepared Loghub-2k data and candidate checks."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite/_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import software_data, software_logparse as lp  # noqa: E402
from autoresearch_suite.runtime import CandidateError, load_program  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

PREPARED = (software_data.data_dir() / "logparse/datasets.json").is_file()


class Metrics(unittest.TestCase):
    def test_grouping_accuracy(self):
        truth = ["A", "A", "B", "B", "C"]
        self.assertEqual(lp.grouping_accuracy(truth, [1, 1, 2, 2, 3]), 1.0)
        self.assertEqual(lp.grouping_accuracy(truth, ["x", "x", "y", "z", "w"]), 3 / 5)  # B split: only A and C count
        self.assertEqual(lp.grouping_accuracy(truth, [0, 0, 0, 0, 1]), 1 / 5)  # A and B merged
        self.assertEqual(lp.grouping_accuracy(truth, [0, 1, 2, 3, 4]), 1 / 5)  # only the singleton event C

    def test_pairwise_f1(self):
        truth = ["A", "A", "B", "B", "C"]
        self.assertEqual(lp.pairwise_f1(truth, [1, 1, 2, 2, 3]), 1.0)
        merged = lp.pairwise_f1(truth, [0, 0, 0, 0, 1])  # 6 predicted pairs, 2 true pairs, 2 correct
        self.assertAlmostEqual(merged, 2 * (2 / 6) * 1.0 / (2 / 6 + 1.0))
        self.assertEqual(lp.pairwise_f1(truth, [0, 1, 2, 3, 4]), 0.0)


@unittest.skipUnless(PREPARED, "run scripts/prepare_software.py --only logparse")
class PreparedData(unittest.TestCase):
    def candidate(self, name, source):
        path = Path(tempfile.mkdtemp()) / name
        path.write_text(source)
        return load_program(path)

    def test_data_and_seed(self):
        datasets = software_data.load_json("logparse/datasets.json")
        self.assertEqual([d["name"] for d in datasets], list(lp.DATASETS))
        self.assertTrue(all(len(d["messages"]) == len(d["events"]) == 2000 for d in datasets))
        seed = seed_source("science-software-log-parsing-loghub2k")
        score, details = lp.logparse_score(self.candidate("seed.py", seed))
        self.assertAlmostEqual(score, 0.79365625)
        self.assertEqual(details["grouping_accuracy"]["Apache"], 1.0)
        perfect_truth_free = "def parse(dataset):\n    return list(range(len(dataset['messages'])))\n"
        self.assertLess(lp.logparse_score(self.candidate("singletons.py", perfect_truth_free))[0], 0.2)

    def test_invalid_candidates(self):
        cases = {"short.py": "def parse(dataset):\n    return [0]\n",
                 "floats.py": "def parse(dataset):\n    return [0.5] * len(dataset['messages'])\n",
                 "missing.py": "def cluster(dataset):\n    return []\n",
                 "reads.py": "def parse(dataset):\n    return open('/etc/passwd').read()\n"}
        for name, source in cases.items():
            with self.subTest(name), self.assertRaises(CandidateError):
                lp.logparse_score(self.candidate(name, source))


if __name__ == "__main__":
    unittest.main()
