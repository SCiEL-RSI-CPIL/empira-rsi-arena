"""Offline tests of the llm_tools telemetry-repair evaluator and the shared sandbox plumbing."""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import llm_tools_common as common  # noqa: E402
from autoresearch_suite import llm_tools_telemetry as tel  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402


def cell(value):
    return repr({"ground_truth": value, "perturbed": value, "corrected": None, "confidence": None})


ROW = {"": "0", "timestamp": "t0", "low_A_egress_to_B": cell(10.0), "low_A_ingress_from_B": cell(4.0),
       "low_B_egress_to_A": cell(4.0), "low_B_ingress_from_A": cell(10.0), "high_A": "ignored"}

ECHO = '''
def run_repair(telemetry, topology):
    return {k: {"rx_rate": (d["rx_rate"], d["rx_rate"], 0.9), "tx_rate": (d["tx_rate"], d["tx_rate"], 0.9),
                "interface_status": (d["interface_status"], d["interface_status"], 1.0), "keys": sorted(d)}
            for k, d in telemetry.items()}
'''


class TelemetryTests(unittest.TestCase):
    def test_row_conversion_pairs_rx_and_tx(self):
        interfaces = tel.row_interfaces(ROW)
        self.assertEqual(sorted(interfaces), ["A_to_B", "B_to_A"])
        self.assertEqual((interfaces["A_to_B"]["tx_rate"], interfaces["A_to_B"]["rx_rate"]), (10.0, 4.0))
        self.assertEqual(interfaces["A_to_B"]["connected_to"], "B_to_A")
        self.assertEqual(tel.router_topology(interfaces), {"A": ["A_to_B"], "B": ["B_to_A"]})

    def test_scenarios_are_deterministic_and_hide_ground_truth(self):
        rows = [dict(ROW, timestamp=f"t{i}") for i in range(5)]
        first, second = tel.scenarios(rows, seed=42, count=3), tel.scenarios(rows, seed=42, count=3)
        self.assertEqual(first, second)
        for case in first:
            for data in case["telemetry"].values():
                self.assertFalse(any(key.startswith("_ground_truth") for key in data))
            self.assertEqual(case["truth"]["A_to_B"]["tx_rate"], 10.0)

    def test_correlated_bug_copies_would_leak_without_stripping(self):
        interfaces = tel.row_interfaces(ROW)
        topology = tel.router_topology(interfaces)
        for seed in range(200):
            perturbed, _, bug = tel.perturb(interfaces, topology, random.Random(seed))
            if bug.startswith("correlated") and any("_ground_truth_rx" in d for d in perturbed.values()):
                return  # the original's perturbation keeps the fields; scenarios() removes them (tested above)
        self.fail("expected a correlated bug that copies the ground-truth fields")

    def test_scoring_matches_hand_computation(self):
        truth = {"A_to_B": {"rx_rate": 4.0, "tx_rate": 10.0, "interface_status": "up"}}
        perfect = {"A_to_B": {"rx_rate": (0, 4.0, 1.0), "tx_rate": (0, 10.0, 1.0), "interface_status": ("up", "up", 1.0)}}
        checked = tel.validate(perfect, truth, 0)
        self.assertEqual((tel.counter_repair(checked, truth), tel.status_repair(checked, truth), tel.calibration(checked, truth)), (1.0, 1.0, 1.0))
        zeros = tel.validate({"A_to_B": {"rx_rate": (0, 0.0, 1.0), "tx_rate": (0, 0.0, 1.0), "interface_status": ("up", "up", 1.0)}}, truth, 0)
        self.assertEqual(tel.counter_repair(zeros, truth), 0.0)
        self.assertEqual(tel.calibration(zeros, truth), 0.0)
        humble = tel.validate({"A_to_B": {"rx_rate": (0, 0.0, 0.1), "tx_rate": (0, 0.0, 0.1), "interface_status": ("up", "up", 1.0)}}, truth, 0)
        self.assertAlmostEqual(tel.calibration(humble, truth), 0.8)

    def test_strict_contract(self):
        truth = {"A_to_B": {"rx_rate": 4.0, "tx_rate": 10.0, "interface_status": "up"}, "B_to_A": {"rx_rate": 10.0, "tx_rate": 4.0, "interface_status": "up"}}
        good = {"rx_rate": (0, 4.0, 1.0), "tx_rate": (0, 10.0, 1.0), "interface_status": ("up", "up", 1.0)}
        with self.assertRaises(CandidateError):
            tel.validate({"A_to_B": good}, truth, 0)  # missing interface
        for bad in ({**good, "rx_rate": (0, float("nan"), 1.0)}, {**good, "tx_rate": (0, 1.0, 1.5)}, {**good, "rx_rate": 4.0},
                    {**good, "interface_status": ("up", 1, 1.0)}):
            with self.assertRaises(CandidateError):
                tel.validate({"A_to_B": bad, "B_to_A": good}, truth, 0)

    def test_sandboxed_candidate_sees_only_the_view(self):
        rows = [dict(ROW, timestamp=f"t{i}") for i in range(4)]
        cases = tel.scenarios(rows, seed=1, count=2)
        ok, outputs, failure = common.call_candidate(ECHO, "run_repair", [[c["telemetry"], c["topology"]] for c in cases], loop=True, cpu_seconds=20)
        self.assertTrue(ok, failure)
        self.assertEqual(outputs[0]["A_to_B"]["keys"], ["connected_to", "interface_status", "local_router", "remote_router", "rx_rate", "tx_rate"])
        score, details = tel.score_outputs(cases, outputs)
        self.assertEqual(details["num_tests"], 2)
        self.assertTrue(0.0 <= score <= 1.0)

    def test_sandbox_blocks_file_access(self):
        source = "def run_repair(t, g):\n    return open(" + repr(str(Path(__file__))) + ").read()\n"
        ok, _, failure = common.call_candidate(source, "run_repair", [{}, {}], cpu_seconds=10)
        self.assertFalse(ok)
        self.assertIn("denied", failure["message"])

    def test_proxy_round_trip_and_fatal_errors(self):
        source = ("def solve(x, tool):\n    try:\n        tool('boom')\n    except RuntimeError:\n        pass\n"
                  "    return tool(x) + 1\n")

        def tool(value):
            if value == "boom":
                raise common.ProxyError("no such thing")
            return value * 2
        self.assertEqual(common.call_candidate(source, "solve", [3], proxies=["tool"], handlers={"tool": tool})[1], 7)

        def broke(value):
            raise CandidateError("budget exhausted")
        with self.assertRaises(CandidateError):
            common.call_candidate("def solve(x, tool):\n    try:\n        tool(x)\n    except RuntimeError:\n        return 0\n",
                                  "solve", [3], proxies=["tool"], handlers={"tool": broke})

    @unittest.skipUnless((common.data_dir() / tel.CSV_NAME).is_file(), "prepared telemetry data missing")
    def test_prepared_data_reproduce_original_sample(self):
        cases = tel.scenarios(tel.load_rows(), seed=42, count=30)
        self.assertEqual(len(cases), 30)
        self.assertEqual(len(cases[0]["telemetry"]), 30)  # 30 directed Abilene interfaces


if __name__ == "__main__":
    unittest.main()
