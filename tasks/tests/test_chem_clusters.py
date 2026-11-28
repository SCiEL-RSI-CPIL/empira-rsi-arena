"""Cluster global-optimisation tasks: published minima are reproduced from constructed geometries."""
from __future__ import annotations

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import chem_clusters as cc  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import needs, seed_source, slow  # noqa: E402

PHI = (1 + 5 ** 0.5) / 2


def icosahedron():
    points = [(a, b * PHI, 0) for a in (-1, 1) for b in (-1, 1)] + [(0, a, b * PHI) for a in (-1, 1) for b in (-1, 1)] + \
             [(b * PHI, 0, a) for a in (-1, 1) for b in (-1, 1)]
    vertices = np.array(points, float)
    return vertices / np.linalg.norm(vertices[0])


def icosahedral(n, scale):
    v = icosahedron()
    if n == 13:
        return np.vstack([[0, 0, 0], scale * v])
    dmin = min(np.linalg.norm(v[i] - v[j]) for i, j in itertools.combinations(range(12), 2))
    shell = [2 * p for p in v] + [v[i] + v[j] for i, j in itertools.combinations(range(12), 2)
                                   if abs(np.linalg.norm(v[i] - v[j]) - dmin) < 1e-9]
    return np.vstack([[0, 0, 0], scale * v, scale * np.array(shell)])


def fcc_cluster(n, centre, scale):
    lattice = np.array([p for p in itertools.product(range(-5, 6), repeat=3) if sum(p) % 2 == 0], float)
    order = np.argsort(np.linalg.norm(lattice - centre, axis=1), kind="stable")[:n]
    return (lattice[order] - centre) * scale / np.sqrt(2)


def pentagonal_bipyramid(edge):
    radius = edge / (2 * np.sin(np.pi / 5))
    height = np.sqrt(edge ** 2 - radius ** 2)
    ring = [[radius * np.cos(2 * np.pi * k / 5), radius * np.sin(2 * np.pi * k / 5), 0.0] for k in range(5)]
    return np.array(ring + [[0, 0, height], [0, 0, -height]])


HOLE = np.array([1.0, 0.0, 0.0])  # octahedral hole of the fcc lattice: the 38-atom truncated octahedron
PUBLISHED = [  # (potential, parameter, start geometry, published energy, published decimals)
    ("morse", 3.0, icosahedral(13, 1.0), -51.737046, 6), ("morse", 6.0, icosahedral(13, 1.0), -42.439863, 6),
    ("morse", 6.0, icosahedral(55, 1.0), -250.286609, 6), ("morse", 10.0, icosahedral(13, 1.0), -39.662975, 6),
    ("morse", 10.0, fcc_cluster(38, HOLE, 1.0), -145.849817, 6),
    ("sutton_chen", "12-6", icosahedral(13, 0.7), -10968.5082, 4), ("sutton_chen", "12-6", fcc_cluster(38, HOLE, 0.7), -35419.9804, 4),
    ("sutton_chen", "12-6", icosahedral(55, 0.7), -52884.6806, 4), ("sutton_chen", "9-6", icosahedral(13, 0.7), -2808.5765, 4),
    ("sutton_chen", "9-6", fcc_cluster(38, HOLE, 0.7), -8917.7056, 4), ("sutton_chen", "9-6", icosahedral(55, 0.7), -13217.8963, 4),
    ("sutton_chen", "10-8", icosahedral(13, 0.7), -3280.3843, 4), ("sutton_chen", "10-8", fcc_cluster(38, HOLE, 0.7), -10117.2454, 4),
    ("sutton_chen", "10-8", fcc_cluster(79, np.zeros(3), 0.7), -21547.5979, 4),
    ("dzugutov", None, pentagonal_bipyramid(1.13), -9.110478, 6), ("dzugutov", None, icosahedral(13, 1.1), -23.206834, 6),
]


class PublishedMinimaTests(unittest.TestCase):
    @needs("scipy")
    def test_constructed_geometries_reproduce_published_minima(self):
        for potential, parameter, start, published, decimals in PUBLISHED:
            with self.subTest(potential=potential, parameter=parameter, n=len(start)):
                energy, _ = cc.relax(start, potential, parameter)
                self.assertLessEqual(abs(energy - published), 0.5 * 10 ** -decimals + 1e-9)

    def test_gradients_match_finite_differences(self):
        rng = np.random.default_rng(0)
        for function, parameter, start in ((cc.morse_energy_gradient, 6.0, icosahedral(13, 1.0)),
                                           (cc.sutton_chen_energy_gradient, "10-8", icosahedral(13, 0.7)),
                                           (cc.dzugutov_energy_gradient, None, icosahedral(13, 1.1))):
            x = start + rng.normal(0, 0.03, start.shape)
            _, gradient = function(x, parameter)
            step = 1e-6
            for i, k in ((0, 0), (3, 1), (12, 2)):
                plus, minus = x.copy(), x.copy()
                plus[i, k] += step
                minus[i, k] -= step
                numeric = (function(plus, parameter)[0] - function(minus, parameter)[0]) / (2 * step)
                self.assertAlmostEqual(numeric, gradient[i, k], delta=1e-4 * max(1.0, abs(numeric)))

    def test_scored_references_are_the_tabulated_values(self):
        self.assertEqual([(r, n) for r, n, _, _ in cc.MORSE_INSTANCES], [(3.0, 80), (6.0, 38), (6.0, 75), (10.0, 38), (10.0, 50), (10.0, 80)])
        self.assertEqual(cc.MORSE_INSTANCES[3][2], -145.849817)
        self.assertEqual(cc.SUTTON_CHEN_INSTANCES[5][2], -21547.5979)
        self.assertEqual(cc.DZUGUTOV_INSTANCES[4][2], -120.619381)
        excluded = {("12-6", 68), ("10-8", 58), ("10-8", 60)}  # improved or not reproduced by Lopez-Castro et al. (2024)
        self.assertFalse(excluded & {(k, n) for k, n, _, _ in cc.SUTTON_CHEN_INSTANCES})


class CandidateValidationTests(unittest.TestCase):
    def test_invalid_coordinates_are_rejected(self):
        for value in (np.zeros((80, 3)), np.ones((79, 3)), [[0.0, 0.0, float("nan")]] * 80, "not coordinates"):
            with self.assertRaises(CandidateError):
                cc.cluster_set_score(lambda instance, v=value: v, "morse")

    def test_overlapping_atoms_are_rejected_even_for_other_instances(self):
        def construct(instance):
            x = fcc_cluster(instance["n"], np.zeros(3), 1.0)
            x[1] = x[0] + 0.1
            return x
        with self.assertRaises(CandidateError):
            cc.cluster_set_score(construct, "dzugutov")

    # Measured at 23 s on one thread: runs only with ARENA_SLOW_TESTS=1.
    @slow
    @needs("scipy")
    def test_seed_is_valid_and_weaker_than_the_published_minima(self):
        namespace = {}
        exec(seed_source("science-chemistry-morse-cluster-minima"), namespace)  # the package's seed
        for potential in ("morse", "sutton_chen", "dzugutov"):
            score, details = cc.cluster_set_score(namespace["construct"], potential)
            self.assertGreater(score, 0.0)
            self.assertEqual(details["instance_count"], 6)
            self.assertTrue(all(row["gap"] > -1e-6 for row in details["instances"]))


if __name__ == "__main__":
    unittest.main()
