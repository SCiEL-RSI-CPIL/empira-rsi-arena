"""CI test prioritisation: data derivation, APFD, the online protocol and the seed."""
from __future__ import annotations

import collections
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite/_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import software_data, software_tcp as tcp  # noqa: E402
from autoresearch_suite.runtime import CandidateError, load_program  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

PREPARED = all((software_data.data_dir() / f"tcp/{d}.json").is_file() for d in tcp.DATASETS)
CSV = """Id;Name;Duration;LastRun;LastResults;Verdict;Cycle
1;20;5;2016-01-01 10:00:00;[];0;1
2;10;7;2016-01-01 10:01:00;[];1;1
3;20;5;2016-01-01 10:02:00;[0];1;1
4;10;7;2016-01-02 10:00:00;[1];0;2
5;30;9;2016-01-02 10:01:00;[];0;2
"""


class Derivation(unittest.TestCase):
    def test_repeated_executions_are_merged_and_tests_sorted(self):
        files = tcp.derive({"atcs-paintcontrol": CSV.encode(), "atcs-iofrol": CSV.encode()})
        import json
        cycles = json.loads(files["tcp/paintcontrol.json"])
        self.assertEqual(cycles, [{"cycle": 1, "tests": [[10, 7, True], [20, 5, True]]},
                                  {"cycle": 2, "tests": [[10, 7, False], [30, 9, False]]}])

    def test_apfd(self):
        self.assertAlmostEqual(tcp.apfd([1, 2, 3, 4], {1}), 1 - 1 / 4 + 1 / 8)
        self.assertAlmostEqual(tcp.apfd([2, 3, 4, 1], {1}), 1 - 4 / 4 + 1 / 8)
        self.assertAlmostEqual(tcp.apfd([1, 2], {1, 2}), 0.5)


class Protocol(unittest.TestCase):
    def setUp(self):
        self.cycles = [{"cycle": 1, "tests": [[1, 5, True], [2, 5, False]]}, {"cycle": 2, "tests": [[1, 5, False], [3, 1, True]]}]
        self.protocol = tcp._Protocol(self.cycles)

    def test_verdicts_follow_the_committed_order(self):
        first = self.protocol({"request": "next"})
        self.assertEqual(first, {"cycle": 1, "tests": [{"id": 1, "duration": 5}, {"id": 2, "duration": 5}]})
        self.assertNotIn("failed", str(first))
        with self.assertRaises(CandidateError):
            self.protocol({"request": "next"})  # the open cycle must be ordered first
        self.assertEqual(self.protocol({"request": "order", "order": [2, 1]}), [True, False])
        with self.assertRaises(CandidateError):
            self.protocol({"request": "order", "order": [2, 1]})  # no cycle is waiting
        self.protocol({"request": "next"})
        for bad in ([1], [1, 1], [1, 4], [1, 3.0], "13"):
            with self.subTest(bad=bad), self.assertRaises(CandidateError):
                self.protocol({"request": "order", "order": bad})
        self.assertEqual(self.protocol({"request": "order", "order": [3, 1]}), [False, True])
        self.assertIsNone(self.protocol({"request": "next"}))
        self.assertEqual(self.protocol.orders, [[2, 1], [3, 1]])


@unittest.skipUnless(PREPARED, "run scripts/prepare_software.py --only tcp")
class PreparedData(unittest.TestCase):
    def offline_last_verdict(self, dataset):
        last, values = {}, []
        for cycle in software_data.load_json(f"tcp/{dataset}.json"):
            ids = [t for t, _, _ in cycle["tests"]]
            failing = {t for t, _, f in cycle["tests"] if f}
            if 0 < len(failing) < len(ids):
                values.append(tcp.apfd(sorted(ids, key=lambda t: (-last.get(t, 0.5), t)), failing))
            last.update({t: 1.0 if f else 0.0 for t, _, f in cycle["tests"]})
        return statistics.fmean(values)

    def test_seed_matches_an_offline_computation_and_bad_candidates_fail(self):
        directory = Path(tempfile.mkdtemp())
        for dataset in tcp.DATASETS:
            (directory / f"seed_{dataset}.py").write_text(seed_source(f"science-software-ci-prioritization-{dataset}"))
            score, details = tcp.tcp_score(load_program(directory / f"seed_{dataset}.py"), dataset)
            self.assertAlmostEqual(score, self.offline_last_verdict(dataset), places=12)
            self.assertLess(score, details["optimal_score"])
        cases = {"dup.py": "def prioritize(cycle, history):\n    return [t['id'] for t in cycle['tests']] * 2\n",
                 "peek.py": "def prioritize(cycle, history):\n    return sorted(t['id'] for t in cycle['tests'] if t['failed'])\n",
                 "missing.py": "def order(cycle, history):\n    return []\n"}
        for name, source in cases.items():
            (directory / name).write_text(source)
            with self.subTest(name), self.assertRaises(CandidateError):
                tcp.tcp_score(load_program(directory / name), "paintcontrol")

    def test_cycles_are_ordered_and_tests_unique(self):
        for dataset in tcp.DATASETS:
            cycles = software_data.load_json(f"tcp/{dataset}.json")
            self.assertEqual([c["cycle"] for c in cycles], sorted(c["cycle"] for c in cycles))
            for c in cycles:
                ids = [t for t, _, _ in c["tests"]]
                self.assertEqual(ids, sorted(collections.Counter(ids)))


if __name__ == "__main__":
    unittest.main()
