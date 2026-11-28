"""Fault localisation: block ownership of lines, EXAM with ties, shuffled spectra and the prepared faults."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite/_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import software_data, software_sbfl as sb  # noqa: E402
from autoresearch_suite.runtime import CandidateError, load_program  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

PREPARED = (software_data.data_dir() / "sbfl/faults.json").is_file()
SOURCE = '''def f(x):
    """Doc."""
    y = x + 1
    if y > 2:
        y = y * 2
        z = 0
    else:
        z = 1
    if x: return y
    return z
'''


class Blocks(unittest.TestCase):
    def test_lines_map_to_their_innermost_block(self):
        owner = sb.fault_blocks(SOURCE)
        self.assertEqual(owner[3], owner[4])  # y = x + 1 and the if header share the function block
        self.assertEqual(owner[5], owner[6])  # the then-branch
        self.assertNotEqual(owner[5], owner[3])
        self.assertNotIn(owner[8], (owner[3], owner[5]))  # the else-branch
        self.assertNotIn(9, owner)  # header and body on one line: ambiguous, left out
        self.assertEqual(owner[10], owner[3])

    def test_exam_with_ties_and_tie_bound(self):
        self.assertEqual(sb.exam([0.9, 0.5, 0.5, 0.1], 0), 1 / 4)
        self.assertEqual(sb.exam([0.9, 0.5, 0.5, 0.1], 1), 2.5 / 4)
        self.assertEqual(sb.exam([0.0] * 4, 3), 2.5 / 4)
        fault = {"ef": [3, 3, 1, 0], "ep": [5, 5, 0, 9], "block": 1}
        self.assertEqual(sb.tie_bound(fault), 1.5 / 4)

    def test_shuffling_is_deterministic_and_tracks_the_faulty_block(self):
        fault = {"subject": "textwrap", "failed": 3, "passed": 9, "ef": list(range(20)), "ep": [0] * 20, "block": 7}
        view, position = sb.shuffled(fault, 4)
        self.assertEqual((view, position), sb.shuffled(fault, 4))
        self.assertEqual(view["ef"][position], 7)
        self.assertNotEqual(view["ef"], fault["ef"])
        self.assertNotIn("block", view)


@unittest.skipUnless(PREPARED, "run scripts/prepare_software.py --only sbfl")
class PreparedFaults(unittest.TestCase):
    def candidate(self, name, source):
        path = Path(tempfile.mkdtemp()) / name
        path.write_text(source)
        return load_program(path)

    def test_data_and_scores(self):
        data = software_data.load_json("sbfl/faults.json")
        self.assertEqual(len(data["faults"]), 532)
        self.assertEqual(data["blocks"], {"fractions": 239, "textwrap": 96, "urllib.parse": 361})
        self.assertTrue(all(f["failed"] > 0 and f["ef"][f["block"]] > 0 for f in data["faults"]))
        seed = seed_source("science-software-fault-localization-seeded")
        score, details = sb.sbfl_score(self.candidate("seed.py", seed))
        self.assertAlmostEqual(score, details["baseline_mean_exam"]["tarantula"], places=12)
        dstar = ("def localize(s):\n    return [ef * ef / (ep + s['failed'] - ef) if ep + s['failed'] - ef else 1e9 + ef * ef\n"
                 "            for ef, ep in zip(s['ef'], s['ep'])]\n")
        self.assertAlmostEqual(sb.sbfl_score(self.candidate("dstar.py", dstar))[0], details["baseline_mean_exam"]["dstar2"], places=12)
        self.assertLess(details["spectrum_tie_bound"], details["baseline_mean_exam"]["dstar2"])

    def test_invalid_candidates(self):
        cases = {"short.py": "def localize(s):\n    return [0.0]\n",
                 "nan.py": "def localize(s):\n    return [float('nan')] * s['blocks']\n",
                 "text.py": "def localize(s):\n    return ['1'] * s['blocks']\n",
                 "missing.py": "def rank(s):\n    return []\n"}
        for name, source in cases.items():
            with self.subTest(name), self.assertRaises(CandidateError):
                sb.sbfl_score(self.candidate(name, source))


if __name__ == "__main__":
    unittest.main()
