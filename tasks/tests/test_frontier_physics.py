"""Frontier physics collection: exact scorers, strict validation, seeds and catalog records."""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import sys
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import frontier_physics_packing as fp, frontier_physics_sphere as fs  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import evidence, needs, package, seed_source  # noqa: E402

CATALOG = json.loads((ROOT / "catalog/science/frontier_physics.json").read_text())
# The evidence file of the source repository is not part of this library; its entries are the science_relevance field
# of the catalog records.
EVIDENCE = evidence("frontier_physics")
# The two sphere families share one seed in the generator script; it is read from one of their packages.
SPHERE_SEED = seed_source("science-frontier-physics-sphere-covering-50")
PHI = (1 + 5 ** 0.5) / 2


def seed_points(source, n):
    module = types.ModuleType("seed")
    module.N = n
    exec(source, module.__dict__)  # noqa: S102 - the task's own seed program
    return module.construct()


def icosahedron():
    points = [(0, a, b * PHI) for a in (-1, 1) for b in (-1, 1)] + [(a, b * PHI, 0) for a in (-1, 1) for b in (-1, 1)] + \
             [(b * PHI, 0, a) for a in (-1, 1) for b in (-1, 1)]
    return np.array(points, float)


def random_rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


class PackingTests(unittest.TestCase):
    def test_known_optimal_packings(self):
        hexagon = [(0.0, 0.0)] + [(2 / 3 * math.cos(k * math.pi / 3), 2 / 3 * math.sin(k * math.pi / 3)) for k in range(6)]
        cases = [([(-0.5, 0.0), (0.5, 0.0)], "disk", 0.5), (hexagon, "disk", 1 / 3),
                 ([(x, y) for x in (-0.25, 0.25) for y in (-0.25, 0.25)], "square", 0.25),
                 (list(itertools.product((-0.25, 0.25), repeat=3)), "cube", 0.25)]
        for points, container, radius in cases:
            with self.subTest(container=container, n=len(points)):
                value, details = fp.packing_radius(points, len(points), container)
                self.assertAlmostEqual(value, radius, places=15)
                self.assertGreater(details["pair_contacts"], 0)
                self.assertGreater(details["wall_contacts"], 0)

    def test_radius_is_the_binding_constraint(self):
        value, details = fp.packing_radius([(-0.4, 0.0), (0.0, 0.0)], 2, "disk")
        self.assertAlmostEqual(value, 0.2, places=15)
        self.assertEqual(details["limited_by"], "pair")
        value, details = fp.packing_radius([(-0.9, 0.0), (0.5, 0.0)], 2, "disk")
        self.assertAlmostEqual(value, 0.1, places=15)
        self.assertEqual(details["limited_by"], "wall")

    def test_invalid_candidates_are_rejected(self):
        bad = {"shape": np.zeros((3, 3)), "rows": np.zeros((4, 2)), "nan": [(0.0, 0.0), (math.nan, 0.1), (0.2, 0.2)],
               "outside": [(0.0, 0.0), (0.2, 0.0), (1.0, 0.0)], "coincident": [(0.1, 0.1), (0.1, 0.1), (-0.3, 0.0)],
               "ragged": [(0.0, 0.0), (0.1,), (0.2, 0.2)], "text": "centres", "object": [{"x": 0}, {"x": 1}, {"x": 2}]}
        for name, candidate in bad.items():
            with self.subTest(name):
                with self.assertRaises(CandidateError):
                    fp.packing_radius(candidate, 3, "disk")
        with self.assertRaises(CandidateError):
            fp.packing_radius([(0.5, 0.0, 0.0), (0.0, 0.0, 0.0)], 2, "cube")  # on the face

    def test_record_comparison_margin(self):
        tie = fp.record_comparison(0.1 + 5e-12, 0.1, "maximize", fp.IMPROVEMENT_MARGIN)
        self.assertEqual(tie["record_status"], "ties_record")
        above = fp.record_comparison(0.1 + 2e-11, 0.1, "maximize", fp.IMPROVEMENT_MARGIN)
        self.assertEqual(above["record_status"], "above_record")
        below = fp.record_comparison(11.0, 10.0, "minimize", 1e-7)
        self.assertEqual(below["record_status"], "below_record")
        self.assertEqual(fp.record_comparison(1.0, None, "maximize", 1e-11), {})


class SphereTests(unittest.TestCase):
    @needs("scipy")
    def test_known_coverings(self):
        octahedron = [s * e for e in np.eye(3) for s in (1, -1)]
        tetrahedron = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        for points, degrees in ((tetrahedron, math.degrees(math.acos(1 / 3))), (octahedron, math.degrees(math.acos(1 / math.sqrt(3)))),
                                (icosahedron(), 37.37736814064969)):
            with self.subTest(n=len(points)):
                value, details = fs.covering_radius(points, len(points))
                self.assertAlmostEqual(value, degrees, places=9)
                self.assertLessEqual(details["empty_cap_violation"], fs.EMPTY_CAP_TOLERANCE)
        # Hardin-Sloane-Smith table values (rounded up to 1e-7 degrees) for N = 4, 6, 12.
        self.assertLessEqual(math.degrees(math.acos(1 / 3)), 70.5287794)
        self.assertLessEqual(fs.covering_radius(icosahedron(), 12)[0], 37.3773682)

    @needs("scipy")
    def test_covering_invariances_and_rejections(self):
        points = seed_points(SPHERE_SEED, 40)
        value = fs.covering_radius(points, 40)[0]
        self.assertAlmostEqual(fs.covering_radius(3.5 * points, 40)[0], value, places=10)
        self.assertAlmostEqual(fs.covering_radius(points @ random_rotation(7).T, 40)[0], value, places=9)
        north = points[points[:, 2] > 0.05]
        with self.assertRaises(CandidateError):
            fs.covering_radius(north, len(north))  # all points in an open hemisphere
        twin = points.copy()
        twin[1] = twin[0]
        with self.assertRaises(CandidateError):
            fs.covering_radius(twin, 40)
        with self.assertRaises(CandidateError):
            fs.covering_radius(np.zeros((40, 3)), 40)
        ring = [(math.cos(t), math.sin(t), 0.0) for t in np.linspace(0, 2 * math.pi, 9)[:-1]]
        with self.assertRaises(CandidateError):
            fs.covering_radius(ring, 8)  # a great circle: closed hemisphere, degenerate hull

    @needs("scipy")
    def test_covering_is_a_maximum_distance(self):
        points = seed_points(SPHERE_SEED, 30)
        value, details = fs.covering_radius(points, 30)
        probe = np.random.default_rng(3).normal(size=(20000, 3))
        probe /= np.linalg.norm(probe, axis=1, keepdims=True)
        unit = points / np.linalg.norm(points, axis=1, keepdims=True)
        sampled = np.degrees(np.arccos(np.clip(np.max(probe @ unit.T, axis=1), -1, 1))).max()
        self.assertLessEqual(sampled, value + 1e-9)  # every sampled point is within the covering radius
        hole = np.array(details["deepest_hole_direction"])
        self.assertAlmostEqual(np.degrees(np.arccos(np.max(unit @ hole))), value, places=8)

    def test_log_energy_exact_values(self):
        tetrahedron = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        triangle = [(math.cos(t), math.sin(t), 0.0) for t in (0, 2 * math.pi / 3, 4 * math.pi / 3)]
        for points, energy in (([(0, 0, 1), (0, 0, -1)], -math.log(2)), (triangle, -3 * math.log(math.sqrt(3))),
                               (tetrahedron, -3 * math.log(8 / 3)), ([s * e for e in np.eye(3) for s in (1, -1)], -12 * math.log(math.sqrt(2)) - 3 * math.log(2))):
            with self.subTest(n=len(points)):
                self.assertAlmostEqual(fs.log_energy(points, len(points))[0], energy, places=13)
        # Nerattini, Brauchart and Kiessling (2014) tabulate v_0(4) = -0.49041462650586 = E / 6.
        self.assertAlmostEqual(fs.log_energy(tetrahedron, 4)[0] / 6, -0.49041462650586, places=13)

    def test_log_energy_invariances_and_rejections(self):
        points = seed_points(SPHERE_SEED, 60)
        value = fs.log_energy(points, 60)[0]
        self.assertAlmostEqual(fs.log_energy(points @ random_rotation(11).T, 60)[0], value, places=9)
        self.assertAlmostEqual(fs.log_energy(0.25 * points, 60)[0], value, places=10)
        twin = points.copy()
        twin[5] = twin[4] * 2.0
        for bad in (twin, points[:59], np.full((60, 3), math.inf), "points"):
            with self.assertRaises(CandidateError):
                fs.log_energy(bad, 60)


class CollectionTests(unittest.TestCase):
    def test_seeds_are_valid_and_below_the_records(self):
        for record in CATALOG:
            family, n, published = record["family"], record["parameters"]["n"], record["parameters"]["published"]
            with self.subTest(task=record["id"]):
                if family.endswith("sphere_covering"):
                    if importlib.util.find_spec("scipy") is None:  # the covering radius is computed from scipy's convex hull
                        self.skipTest("scipy is not installed")
                    value = fs.covering_radius(seed_points(seed_source(record["id"]), n), n, published)[0]
                    self.assertGreater(value, published)
                elif family.endswith("log_energy"):
                    value = fs.log_energy(seed_points(seed_source(record["id"]), n), n, published)[0]
                    self.assertGreater(value, published)
                else:
                    # the container is the one the package's evaluator passes to the scorer
                    container = re.search(r"P\['n'\], '(\w+)'", package(record["id"])["body"]).group(1)
                    value = fp.packing_radius(seed_points(seed_source(record["id"]), n), n, container, published)[0]
                    self.assertLess(value, published)
                    self.assertGreater(value, 0.8 * published)

    def test_records_are_open_human_best_known_values(self):
        # the generated packages, under the family name each of them states
        registered = {spec["family"]: spec for spec in (package(record["id"]) for record in CATALOG)}
        self.assertEqual(len(CATALOG), 10)
        for record in CATALOG:
            with self.subTest(task=record["id"]):
                self.assertTrue(record["id"].startswith("science-frontier-physics-"))
                self.assertIn(record["family"], registered)
                self.assertEqual(registered[record["family"]]["protocol"], "recreated_verified")
                sota = record["state_of_the_art"]
                self.assertEqual((sota["kind"], sota["attribution"]), ("best_known", "human"))
                self.assertEqual(sota["value"], record["parameters"]["published"])
                self.assertTrue(sota["as_of"] and sota["statement"] and sota["sources"])
                self.assertTrue(all(s["url"].startswith("https://") and s["title"] for s in sota["sources"]))
                self.assertTrue(all(s["url"].startswith("https://") and s["type"] in ("paper", "benchmark", "archive") for s in record["sources"]))
                labels = record["labels"]
                self.assertEqual((labels["scientific_contribution"], labels["data"], labels["verification"], labels["reference"]),
                                 ("open_problem", "none", "exact", "published_value"))
                self.assertEqual(EVIDENCE[record["id"]]["status"], "supported")
        self.assertEqual(set(EVIDENCE), {r["id"] for r in CATALOG})


if __name__ == "__main__":
    unittest.main()
