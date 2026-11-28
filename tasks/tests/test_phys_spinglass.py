"""Spin-glass tasks: instance integrity, exact cut recomputation and contract enforcement."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import phys_spinglass as sg  # noqa: E402
from autoresearch_suite.bio_physics_common import data_dir  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

PREPARED = (sg.load_derived.__module__ is not None) and (data_dir() / "spinglass/torus_2d.json.gz").is_file()


class CutWeightTests(unittest.TestCase):
    def test_cut_weight_on_a_hand_checked_instance(self):
        instance = {"nodes": 4, "side": 2, "edges": [[0, 1, 5], [1, 2, -3], [2, 3, 2], [3, 0, 7]]}
        self.assertEqual(sg.cut_weight(instance, [1, 1, 1, 1]), 0)
        self.assertEqual(sg.cut_weight(instance, [1, -1, 1, -1]), 5 - 3 + 2 + 7)
        self.assertEqual(sg.cut_weight(instance, [1, 1, -1, -1]), -3 + 7)

    def test_invalid_spin_vectors_are_rejected(self):
        instance = {"nodes": 4, "side": 2, "edges": [[0, 1, 5], [1, 2, -3], [2, 3, 2], [3, 0, 7]]}
        for bad in ([1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 2], [1, 1, 1, float("nan")], [1, 1, 1, 0.5]):
            with self.assertRaises(CandidateError):
                sg.cut_weight(instance, bad)

    def test_published_optima_cover_both_lattices(self):
        self.assertEqual(len(sg.OPTIMA), 18)
        self.assertEqual(sum(1 for side, dim, _ in sg.OPTIMA.values() if dim == 2), 9)
        for name, (side, dimension, optimum) in sg.OPTIMA.items():
            self.assertGreater(optimum, 0, name)
            self.assertEqual(name.startswith("t2g"), dimension == 2, name)
            self.assertEqual(int(name.split("_")[0][3:]), side, name)


@unittest.skipUnless(PREPARED, "prepared data missing; run scripts/prepare_bio_physics.py --only phys_spinglass")
class PreparedInstanceTests(unittest.TestCase):
    def test_instances_are_tori_with_the_published_optima(self):
        for dimension in (2, 3):
            instances = sg.load_instances(dimension)
            self.assertEqual(len(instances), 9)
            for name, instance in instances:
                side, dim, optimum = sg.OPTIMA[name]
                self.assertEqual(dim, dimension)
                self.assertEqual(instance["nodes"], side ** dimension, name)
                self.assertEqual(len(instance["edges"]), dimension * side ** dimension, name)
                self.assertEqual(instance["optimum"], optimum, name)
                degree = [0] * instance["nodes"]
                for i, j, _ in instance["edges"]:
                    degree[i] += 1
                    degree[j] += 1
                self.assertEqual(set(degree), {2 * dimension}, name)  # every site has 2*dimension neighbours

    def test_public_instance_hides_the_optimum(self):
        name, instance = sg.load_instances(2)[0]
        public = sg.public_instance(name, instance)
        self.assertNotIn("optimum", public)
        self.assertEqual(set(public), {"name", "nodes", "side", "dimension", "edges"})
        self.assertEqual(public["dimension"], 2)


if __name__ == "__main__":
    unittest.main()
