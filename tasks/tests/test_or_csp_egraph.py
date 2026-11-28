"""E-graph extraction proxy (package or_csp): extraction-gym check / DAG cost port and scoring."""
from __future__ import annotations

import gzip
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "suite" / "_lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import or_csp_egraph as E  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
import suite_test_support  # noqa: E402

TASK = "magellan-xla-enzyme-jax-e-graph-extraction"
UPGRADE = json.loads((ROOT / "catalog" / "upgrades" / "or_csp.json").read_text())["tasks"][TASK]


def node(eclass, cost, *children):
    return {"op": eclass, "eclass": eclass, "cost": cost, "children": list(children)}


# a = f(b, c) or a = g(c); b = h(c) or a leaf; c is a leaf, or c = k(a) (a cycle back to a)
TOY = {"nodes": {"a0": node("a", 1.0, "b0", "c0"), "a1": node("a", 5.0, "c0"), "b0": node("b", 1.0, "c0"),
                 "b1": node("b", 3.0), "c0": node("c", 2.0), "c1": node("c", 0.5, "a1")},
       "root_eclasses": ["a"]}


def seed_module():
    """The package's seed program (a port of extraction-gym's bottom-up extractor)."""
    return suite_test_support.seed_module(TASK)


class CheckTests(unittest.TestCase):
    def test_dag_cost_counts_shared_classes_once(self):
        choices = {"a": "a0", "b": "b0", "c": "c0"}
        self.assertEqual(E.check_extraction(TOY, choices), 3)
        self.assertEqual(E.dag_cost(TOY, choices), 4.0)  # tree cost would count c twice (6.0)
        self.assertEqual(E.dag_cost(TOY, {"a": "a1", "c": "c0"}), 7.0)

    def test_invalid_extractions(self):
        for label, choices in {"missing root": {"b": "b1", "c": "c0"},
                               "missing reachable": {"a": "a0", "c": "c0"},
                               "wrong class": {"a": "a0", "b": "c0", "c": "c0"},
                               "unknown node": {"a": "a9", "b": "b0", "c": "c0"},
                               "cycle": {"a": "a1", "c": "c1"},
                               "not a dict": [("a", "a0")],
                               "non-string ids": {"a": 0}}.items():
            with self.subTest(case=label), self.assertRaises(CandidateError):
                E.check_extraction(TOY, choices)

    def test_extra_unreachable_choices_are_ignored(self):
        self.assertEqual(E.dag_cost(TOY, {"a": "a1", "b": "b1", "c": "c0"}), 7.0)
        E.check_extraction(TOY, {"a": "a1", "b": "b1", "c": "c0"})

    def test_sequential_summation_like_rust(self):
        values = [0.1] * 10 + [1e16, 1.0, -1e16]
        graph = {"nodes": {f"n{i}": node(f"k{i}", v, *([f"n{i + 1}"] if i + 1 < len(values) else [])) for i, v in enumerate(values)},
                 "root_eclasses": ["k0"]}
        choices = {f"k{i}": f"n{i}" for i in range(len(values))}
        expected = 0.0
        for v in values:  # the chain is visited from the root, so the insertion order is k0..kN
            expected += v
        self.assertEqual(E.dag_cost(graph, choices), expected)
        self.assertNotEqual(expected, math.fsum(values))


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with gzip.open(Path(self.tmp.name) / "tensat-toy.json.gz", "wt") as stream:
            json.dump(TOY, stream)
        self.params = {"graphs": ["toy"], "reference_dag_cost": {"toy": 4.0}, "time_limit_s": 5.0}

    def tearDown(self):
        self.tmp.cleanup()

    def test_scores_and_tampering(self):
        def good(graph):
            return {"a": "a0", "b": "b0", "c": "c0"}

        def tamper(graph):
            for item in graph["nodes"].values():
                item["cost"] = 0.0
            return {"a": "a1", "c": "c0"}
        score, details = E.extraction_score(types.SimpleNamespace(extract=good), self.params, data_dir=self.tmp.name)
        self.assertEqual(score, 1.0)
        score, _ = E.extraction_score(types.SimpleNamespace(extract=tamper), self.params, data_dir=self.tmp.name)
        self.assertEqual(score, 7.0 / 4.0)

    def test_seed_reproduces_bottom_up_on_toy(self):
        module = seed_module()
        choices = module.extract(json.loads(json.dumps(TOY)))
        E.check_extraction(TOY, choices)
        self.assertEqual(choices["a"], "a0")  # tree costs: a0 = 1 + 1 + 2 + 2 = 6 < a1 = 5 + 2 = 7

    def test_reference_costs_match_bottom_up_seed_when_data_present(self):
        if not (E.DATA / "tensat-vgg.json.gz").is_file():
            self.skipTest("prepared e-graphs absent: run scripts/prepare_or_csp.py")
        module = seed_module()
        for name in ("vgg", "resnet50", "bert"):
            with self.subTest(graph=name):
                graph = E.load_egraph(name)
                choices = module.extract(graph)
                E.check_extraction(graph, choices)
                self.assertEqual(E.dag_cost(graph, choices), UPGRADE["params"]["bottom_up_dag_cost"][name])


if __name__ == "__main__":
    unittest.main()
