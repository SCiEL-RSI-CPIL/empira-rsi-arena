"""Audit checks of the record-construction and physics-model evaluators against independent implementations.

Every test compares a library function with a definition written out here (brute force, exact rational
arithmetic, a different formula) or with a literature value. No network; tests that need SciPy or prepared
data skip when it is absent.
"""
from __future__ import annotations

import cmath
import collections
from fractions import Fraction
import importlib.util
import itertools
import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "suite/_lib")]
from autoresearch_suite import (chem_clusters, frontier_combinatorics_codes as codes, frontier_combinatorics_designs as designs,
                                frontier_combinatorics_graphs as graphs, frontier_combinatorics_qap as qap,
                                frontier_physics_packing as packing, frontier_physics_sphere as sphere, phys_coating,
                                phys_nuclear, phys_spinglass)
from autoresearch_suite.runtime import CandidateError

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
FANO = [[0, 1, 2], [0, 3, 4], [0, 5, 6], [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]]
PHI = (1 + 5 ** 0.5) / 2
ICOSAHEDRON = np.array([[0, a, b * PHI] for a in (-1, 1) for b in (-1, 1)] + [[a, b * PHI, 0] for a in (-1, 1) for b in (-1, 1)]
                       + [[b * PHI, 0, a] for a in (-1, 1) for b in (-1, 1)], float)


def prepared(loader):
    """Value of a data loader, or a skip when the prepared file is absent."""
    try:
        return loader()
    except (RuntimeError, FileNotFoundError) as error:
        raise unittest.SkipTest(str(error)[:120]) from None


class CoveringDesignsAndTriangleSets(unittest.TestCase):
    def test_covering_design_agrees_with_subset_definition(self):
        rng = random.Random(1)
        for _ in range(300):
            v = rng.randint(3, 8)
            k = rng.randint(1, v)
            t = rng.randint(1, k)
            blocks = {frozenset(rng.sample(range(v), k)) for _ in range(rng.randint(1, 10))}
            if rng.random() < 0.6:  # complete greedily so that valid designs occur
                for subset in itertools.combinations(range(v), t):
                    if not any(set(subset) <= b for b in blocks):
                        rest = [x for x in range(v) if x not in subset]
                        blocks.add(frozenset(subset) | frozenset(rest[:k - t]))
            covered = all(any(set(s) <= b for b in blocks) for s in itertools.combinations(range(v), t))
            candidate = [sorted(b, reverse=True) for b in blocks]
            if covered and len(blocks) <= math.comb(v, t):
                self.assertEqual(designs.covering_design(candidate, v, k, t)[0], len(blocks))
            else:
                with self.assertRaises(CandidateError):
                    designs.covering_design(candidate, v, k, t)

    def test_known_covering_numbers_and_hostile_blocks(self):
        self.assertEqual(designs.covering_design(FANO, 7, 3, 2)[0], 7)
        self.assertEqual(designs.covering_design(np.array(FANO, dtype=np.int32), 7, 3, 2)[0], 7)
        triples = [list(b) for b in itertools.combinations(range(5), 3)]
        valid = [m for m in range(1, 5) for c in itertools.combinations(triples, m) if self._ok(list(c), 5, 3, 2)]
        self.assertEqual(min(valid), 4)  # C(5,3,2) = 4
        for bad in (FANO[:-1], FANO + [[2, 1, 0]], [[0, 0, 1]] + FANO, [[0, 1, 7]] + FANO, [[True, 1, 2]] + FANO[1:],
                    [[0.0, 1, 2]] + FANO[1:], [set(b) for b in FANO], [], None):
            with self.assertRaises(CandidateError):
                designs.covering_design(bad, 7, 3, 2)
        self.assertEqual([designs.schoenheim(19, 7, 4), designs.schoenheim(19, 8, 5), designs.schoenheim(24, 7, 3)], [114, 228, 69])

    @staticmethod
    def _ok(blocks, v, k, t):
        try:
            designs.covering_design(blocks, v, k, t)
            return True
        except CandidateError:
            return False

    def test_difference_triangle_sets(self):
        self.assertEqual(designs.difference_triangle_set([[0, 1, 4, 6]], 1, 3)[0], 6)  # optimal Golomb rulers
        self.assertEqual(designs.difference_triangle_set([[0, 1, 4, 9, 11]], 1, 4)[0], 11)
        self.assertEqual(designs.difference_triangle_set([[0, 2, 7], [0, 1, 4]], 2, 2)[0], 7)  # m(2,2) = 7
        rng = random.Random(2)
        for _ in range(500):
            n, k = rng.randint(1, 3), rng.randint(1, 4)
            rows = [[0] + sorted(rng.sample(range(1, 40), k)) for _ in range(n)]
            diffs = [b - a for row in rows for a, b in itertools.combinations(row, 2)]
            if len(set(diffs)) == len(diffs):
                self.assertEqual(designs.difference_triangle_set(rows, n, k)[0], max(row[-1] for row in rows))
            else:
                with self.assertRaises(CandidateError):
                    designs.difference_triangle_set(rows, n, k)
        for bad in ([[1, 2, 5, 7]], [[0, 4, 1, 6]], [[False, 1, 4, 6]], [[0, 1.0, 4, 6]], [[0, 1, 4, 6], [0, 1, 4, 6]]):
            with self.assertRaises(CandidateError):
                designs.difference_triangle_set(bad, len(bad), 3)


class CoveringCodesAndSuperpermutations(unittest.TestCase):
    def test_covering_code_agrees_with_hamming_definition(self):
        rng = random.Random(3)
        for _ in range(200):
            q, n = rng.randint(2, 4), rng.randint(1, 4)
            radius = rng.randint(0, min(n, 2))
            space = list(itertools.product(range(q), repeat=n))
            code = rng.sample(space, rng.randint(1, max(1, len(space) // 2)))
            covers = all(any(sum(a != b for a, b in zip(w, c)) <= radius for c in code) for w in space)
            if covers:
                self.assertEqual(codes.covering_code([list(c) for c in code], q, n, radius)[0], len(code))
            else:
                with self.assertRaises(CandidateError):
                    codes.covering_code([list(c) for c in code], q, n, radius)

    def test_perfect_codes(self):
        checks = [[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]]
        hamming = [list(w) for w in itertools.product(range(2), repeat=7) if all(sum(h[i] * w[i] for i in range(7)) % 2 == 0 for h in checks)]
        self.assertEqual(codes.covering_code(hamming, 2, 7, 1)[0], 16)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        self.assertEqual(codes.covering_code(tetracode, 3, 4, 1)[0], 9)
        for bad in (hamming[:-1], hamming + [hamming[0]], [[2] + hamming[0][1:]] + hamming, [[True] + hamming[0][1:]] + hamming[1:]):
            with self.assertRaises(CandidateError):
                codes.covering_code(bad, 2, 7, 1)

    def test_superpermutations(self):
        self.assertEqual(codes.superpermutation("123121321", 3)[0], 9)  # the minimal word for n = 3
        self.assertEqual(codes.superpermutation([1, 2, 1], 2)[0], 3)
        rng = random.Random(4)
        for _ in range(300):
            word = "".join(rng.choice("123") for _ in range(rng.randint(3, 18)))
            complete = all("".join(p) in word for p in itertools.permutations("123"))
            if complete:
                self.assertEqual(codes.superpermutation(word, 3)[0], len(word))
            else:
                with self.assertRaises(CandidateError):
                    codes.superpermutation(word, 3)
        for bad in ("1230121321", "123 121321", "12312132١", [1.0, 2, 1], [True, 2, 1], b"121"):
            with self.assertRaises(CandidateError):
                codes.superpermutation(bad, 3)


def bfs_girth(n, edges):
    adjacency = collections.defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    best = math.inf
    for source in range(n):
        dist, parent, queue = {source: 0}, {source: -1}, collections.deque([source])
        while queue:
            u = queue.popleft()
            for w in adjacency[u]:
                if w not in dist:
                    dist[w], parent[w] = dist[u] + 1, u
                    queue.append(w)
                elif parent[u] != w:
                    best = min(best, dist[u] + dist[w] + 1)
    return best


def lcf(n, pattern):
    edges = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    edges |= {(min(i, (i + pattern[i % len(pattern)]) % n), max(i, (i + pattern[i % len(pattern)]) % n)) for i in range(n)}
    return n, sorted(edges)


class Cages(unittest.TestCase):
    def test_known_cages(self):
        petersen = (10, [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)] + [(5 + i, 5 + (i + 2) % 5) for i in range(5)])
        for (n, edges), g in ((petersen, 5), (lcf(14, [5, -5]), 6), (lcf(24, [12, 7, -7]), 7), (lcf(30, [-13, -9, 7, -7, 9, 13]), 8)):
            self.assertEqual(bfs_girth(n, edges), g)
            self.assertEqual(graphs.regular_girth_graph(n, edges, 3, g, 50000)[0], n)
        self.assertEqual([graphs.moore_bound(3, 16), graphs.moore_bound(4, 10), graphs.moore_bound(5, 9)], [510, 242, 426])
        n, edges = petersen
        for bad in (edges[:-1] + [(0, 0)], edges[:-1] + [edges[0]], edges[:-1] + [(edges[0][1], edges[0][0])], edges[:-1],
                    edges[:-1] + [(True, 9)], edges[:-1] + [(7.0, 9)]):
            with self.assertRaises(CandidateError):
                graphs.regular_girth_graph(n, bad, 3, 5, 50000)

    def test_girth_certificate_agrees_with_breadth_first_search(self):
        rng = random.Random(5)
        seen = 0
        while seen < 60:
            n, k = rng.choice([10, 14, 20, 30, 40]), rng.choice([3, 4])
            points = [v for v in range(n) for _ in range(k)]
            rng.shuffle(points)
            edges = {(min(a, b), max(a, b)) for a, b in zip(points[::2], points[1::2]) if a != b}
            if len(edges) != n * k // 2:
                continue
            seen += 1
            true_girth = bfs_girth(n, edges)
            for g in range(3, 8):
                if graphs.moore_bound(k, g) > n:
                    continue
                if true_girth >= g:
                    self.assertEqual(graphs.regular_girth_graph(n, sorted(edges), k, g, 50000)[0], n)
                else:
                    with self.assertRaisesRegex(CandidateError, f"cycle of length {true_girth},"):
                        graphs.regular_girth_graph(n, sorted(edges), k, g, 50000)


class QuadraticAssignment(unittest.TestCase):
    def test_objective_is_the_qaplib_sum(self):
        rng = random.Random(6)
        for _ in range(20):
            n = rng.randint(2, 5)
            a = [[rng.randint(-9, 9) for _ in range(n)] for _ in range(n)]
            b = [[rng.randint(-9, 9) for _ in range(n)] for _ in range(n)]
            for p in itertools.permutations(range(n)):
                expected = sum(a[i][j] * b[p[i]][p[j]] for i in range(n) for j in range(n))
                self.assertEqual(qap.objective(list(p), np.array(a), np.array(b))[0], expected)
        a, b = qap.parse("2\n1 2\n3 4\n\n5 6\n7 8\n")
        self.assertEqual((a.tolist(), b.tolist()), ([[1, 2], [3, 4]], [[5, 6], [7, 8]]))
        for bad in ([0, 0], [1, 2], [0.0, 1.0], [False, True], [0], "01"):
            with self.assertRaises(CandidateError):
                qap.objective(bad, a, b)

    def test_pinned_instances(self):
        pins = {"tai256c": "e7d72083d49f4768e5a15a02ee33fde038a794e7407e5e485c3c110f1509fafd",
                "tho150": "40136bc14aa404f92e7928c386e93218cf64437bd9c15f95830222701728690f",
                "wil100": "9477abe6459def2fc664ee49f612a592bba7b59c2a3027c4a0382d068c6aa4ca"}
        for name, digest in pins.items():
            a, b = prepared(lambda: qap.instance(name, digest))
            n = len(a)
            self.assertEqual(n, int(name[3:6]))
            p = list(range(1, n)) + [0]
            self.assertEqual(qap.objective(p, a, b)[0], sum(int(a[i, j]) * int(b[p[i], p[j]]) for i in range(n) for j in range(n) if a[i, j]))


def exact_radius(points, container):
    """Packing radius in exact rational arithmetic, rounded once at the end."""
    scale = 10 ** 30
    root = lambda q: Fraction(math.isqrt(q.numerator * scale * scale // q.denominator), scale)
    pts = [[Fraction(repr(c)) for c in p] for p in points]
    half = root(min(sum((a - b) ** 2 for a, b in zip(p, q)) for p, q in itertools.combinations(pts, 2))) / 2
    if container in ("disk", "ball"):
        wall = min(1 - root(sum(a * a for a in p)) for p in pts)
    else:
        wall = min(Fraction(1, 2) - abs(a) for p in pts for a in p)
    return float(min(half, wall))


class PackingsAndSphericalCodes(unittest.TestCase):
    def test_packing_radius_matches_exact_arithmetic(self):
        rng = random.Random(7)
        for _ in range(100):
            container = rng.choice(sorted(packing.CONTAINERS))
            dim = packing.CONTAINERS[container]
            points = [[round(rng.uniform(-0.3, 0.3), 6) for _ in range(dim)] for _ in range(rng.randint(2, 9))]
            self.assertAlmostEqual(packing.packing_radius(points, len(points), container)[0], exact_radius(points, container), delta=1e-15)

    def test_packing_reference_configurations_and_boundaries(self):
        hexagon = [[0.0, 0.0]] + [[2 / 3 * math.cos(k * math.pi / 3), 2 / 3 * math.sin(k * math.pi / 3)] for k in range(6)]
        self.assertAlmostEqual(packing.packing_radius(hexagon, 7, "disk")[0], 1 / 3, delta=1e-15)  # optimal for N = 7
        square = [[x, y] for x in (-0.25, 0.25) for y in (-0.25, 0.25)]
        self.assertEqual(packing.packing_radius(square, 4, "square")[0], 0.25)  # optimal for N = 4
        cube = [[x, y, z] for x in (-0.25, 0.25) for y in (-0.25, 0.25) for z in (-0.25, 0.25)]
        self.assertEqual(packing.packing_radius(cube, 8, "cube")[0], 0.25)  # optimal for N = 8
        for bad, container in (([[1.0, 0.0], [0.0, 0.0]], "disk"), ([[0.5, 0.0], [0.0, 0.0]], "square"), ([[0.1, 0.1], [0.1, 0.1]], "square"),
                               ([[math.nan, 0.0], [0.0, 0.0]], "disk"), ([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], "square")):
            with self.assertRaises(CandidateError):
                packing.packing_radius(bad, 2, container)
        status = lambda gain: packing.record_comparison(0.1 + gain, 0.1, "maximize", packing.IMPROVEMENT_MARGIN)["record_status"]
        self.assertEqual([status(0.9e-11), status(1.1e-11), status(-0.9e-11), status(-1.1e-11)],
                         ["ties_record", "above_record", "ties_record", "below_record"])

    def test_log_energy(self):
        tetrahedron = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
        octahedron = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        self.assertAlmostEqual(sphere.log_energy([[0, 0, 2], [0, 0, -5]], 2)[0], -math.log(2), delta=1e-15)
        self.assertAlmostEqual(sphere.log_energy(tetrahedron, 4)[0], -6 * math.log(math.sqrt(8 / 3)), delta=1e-14)
        self.assertAlmostEqual(sphere.log_energy(octahedron, 6)[0], -(12 * math.log(math.sqrt(2)) + 3 * math.log(2)), delta=1e-14)
        rng = np.random.default_rng(8)
        x = rng.normal(size=(40, 3))
        unit = (x / np.linalg.norm(x, axis=1)[:, None]).tolist()
        expected = -math.fsum(0.5 * math.log(sum((a - b) ** 2 for a, b in zip(p, q))) for p, q in itertools.combinations(unit, 2))
        self.assertAlmostEqual(sphere.log_energy(x, 40)[0], expected, delta=1e-11)
        for bad in (np.vstack([x[:39], x[:1]]), np.vstack([x[:39], [[0.0, 0.0, 0.0]]]), np.vstack([x[:39], [[math.nan, 0.0, 0.0]]])):
            with self.assertRaises(CandidateError):
                sphere.log_energy(bad, 40)

    @unittest.skipUnless(HAVE_SCIPY, "covering_radius needs SciPy")
    def test_covering_radius(self):
        tetrahedron = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
        cube = np.array(list(itertools.product([-1, 1], repeat=3)), float)  # four cocircular points on every face
        self.assertAlmostEqual(sphere.covering_radius(tetrahedron, 4)[0], math.degrees(math.acos(1 / 3)), delta=1e-11)
        self.assertAlmostEqual(sphere.covering_radius(3.0 * cube, 8)[0], math.degrees(math.acos(1 / math.sqrt(3))), delta=1e-11)
        self.assertAlmostEqual(sphere.covering_radius(ICOSAHEDRON, 12)[0], 37.3773681406497, delta=1e-10)  # table: 37.3773682
        rng = np.random.default_rng(9)
        for _ in range(5):  # definition: the deepest holes are circumcentres of triples with an empty cap
            u = rng.normal(size=(14, 3))
            u /= np.linalg.norm(u, axis=1)[:, None]
            best = 0.0
            for i, j, k in itertools.combinations(range(14), 3):
                normal = np.cross(u[j] - u[i], u[k] - u[i])
                for centre in (normal / np.linalg.norm(normal), -normal / np.linalg.norm(normal)):
                    if (u @ centre).max() <= centre @ u[i] + 1e-12:
                        best = max(best, math.degrees(math.acos(centre @ u[i])))
            self.assertAlmostEqual(sphere.covering_radius(u, 14)[0], best, delta=1e-9)
        with self.assertRaises(CandidateError):
            sphere.covering_radius(np.abs(rng.normal(size=(10, 3))), 10)  # inside a hemisphere


class ClusterEnergies(unittest.TestCase):
    @staticmethod
    def icosahedron_minimum(energy):
        """Lowest energy of a centred 13-atom icosahedron over its radius (one free parameter under I_h)."""
        shell = ICOSAHEDRON / np.linalg.norm(ICOSAHEDRON[0])
        value = lambda r: energy(np.vstack([[0.0, 0.0, 0.0], r * shell]))
        low, high = 0.7, 1.3
        for _ in range(200):
            a, b = low + (high - low) / 3, high - (high - low) / 3
            low, high = (low, b) if value(a) < value(b) else (a, high)
        return value(0.5 * (low + high))

    def test_published_icosahedra(self):
        # Cambridge Cluster Database: Morse 13A at rho = 3, 6, 10, 14; Dzugutov N = 13 (I_h).
        for rho, published in ((3.0, -51.737046), (6.0, -42.439863), (10.0, -39.662975), (14.0, -37.258877)):
            self.assertAlmostEqual(self.icosahedron_minimum(lambda x: chem_clusters.morse_energy_gradient(x, rho)[0]), published, delta=6e-7)
        self.assertAlmostEqual(self.icosahedron_minimum(lambda x: chem_clusters.dzugutov_energy_gradient(x)[0]), -23.206834, delta=6e-7)

    def test_energies_and_gradients_match_the_formulas(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(8, 3)) * 0.9
        dist = [math.dist(p, q) for p, q in itertools.combinations(x.tolist(), 2)]
        morse = math.fsum(math.exp(6 * (1 - r)) * (math.exp(6 * (1 - r)) - 2) for r in dist)
        self.assertAlmostEqual(chem_clusters.morse_energy_gradient(x, 6.0)[0], morse, delta=1e-9 * abs(morse))
        density = [sum(math.dist(p, q) ** -6 for j, q in enumerate(x.tolist()) if j != i) for i, p in enumerate(x.tolist())]
        sutton_chen = sum(r ** -9 for r in dist) - 39.432 * sum(math.sqrt(d) for d in density)
        self.assertAlmostEqual(chem_clusters.sutton_chen_energy_gradient(x, "9-6")[0], sutton_chen, delta=1e-9 * abs(sutton_chen))
        pair = lambda r: ((5.82 * (r ** -16 - 1.28) * math.exp(1.1 / (r - 1.87)) if r < 1.87 else 0.0)
                          + (1.28 * math.exp(0.27 / (r - 1.94)) if r < 1.94 else 0.0))
        self.assertAlmostEqual(chem_clusters.dzugutov_energy_gradient(x)[0], sum(pair(r) for r in dist), delta=1e-9)
        for function in (lambda y: chem_clusters.morse_energy_gradient(y, 6.0), lambda y: chem_clusters.sutton_chen_energy_gradient(y, "9-6"),
                         chem_clusters.dzugutov_energy_gradient):
            gradient = function(x)[1]
            step = np.zeros_like(x)
            step[3, 1] = 1e-6
            numeric = (function(x + step)[0] - function(x - step)[0]) / 2e-6
            self.assertAlmostEqual(gradient[3, 1], numeric, delta=1e-5 * max(1.0, abs(numeric)))

    def test_score_is_relative_excess_over_the_references(self):
        far = lambda instance: np.arange(instance["n"] * 3, dtype=float).reshape(-1, 3) * 100.0  # no bonds: E = 0
        self.assertAlmostEqual(chem_clusters.cluster_set_score(far, "dzugutov")[0], 100.0, delta=1e-9)
        for bad in (lambda i: np.zeros((i["n"], 3)), lambda i: np.zeros((i["n"] + 1, 3)), lambda i: np.full((i["n"], 3), math.nan)):
            with self.assertRaises(CandidateError):
                chem_clusters.cluster_set_score(bad, "morse")


class PhysicsModels(unittest.TestCase):
    def test_reflectance_agrees_with_the_airy_recursion(self):
        def airy(indices, widths, wavelength):  # interfaces folded from the substrate upwards
            stack = [1.0] + list(indices) + [1.52]
            gamma = (stack[-2] - stack[-1]) / (stack[-2] + stack[-1])
            for k in range(len(indices), 0, -1):
                phase = cmath.exp(-4j * math.pi * stack[k] * widths[k - 1] / wavelength)
                r = (stack[k - 1] - stack[k]) / (stack[k - 1] + stack[k])
                gamma = (r + gamma * phase) / (1 + r * gamma * phase)
            return abs(gamma) ** 2
        rng = random.Random(11)
        for _ in range(50):
            layers = rng.randint(0, 20)
            indices = [rng.choice([1.38, 2.35]) for _ in range(layers)]
            widths = [rng.uniform(0, 400) for _ in range(layers)]
            for wavelength in (400.0, 537.0, 800.0):
                self.assertAlmostEqual(phys_coating.reflectance(indices, widths, [wavelength])[0], airy(indices, widths, wavelength), delta=1e-13)
        self.assertAlmostEqual(phys_coating.reflectance([], [], [550.0])[0], (0.52 / 2.52) ** 2, delta=1e-15)
        quarter = phys_coating.reflectance([1.38], [600 / (4 * 1.38)], [600.0])[0]  # textbook quarter-wave value
        self.assertAlmostEqual(quarter, ((1.52 - 1.38 ** 2) / (1.52 + 1.38 ** 2)) ** 2, delta=1e-15)
        self.assertNotAlmostEqual(phys_coating.reflectance([1.38, 2.35], [100, 60], [550.0])[0],
                                  phys_coating.reflectance([2.35, 1.38], [60, 100], [550.0])[0], delta=1e-3)

    def test_coating_constraints(self):
        self.assertEqual(len(phys_coating.check_design([["L", "H"], [100.0, 50.0]], 20, 1000.0, 400.0)[1]), 2)
        for bad in ([["L"] * 21, [10.0] * 21], [["L"], [400.1]], [["L"], [-1.0]], [["X"], [10.0]], [["L", "H", "L"], [400.0, 400.0, 200.001]],
                    [["L"], [math.nan]], [["L", "H"], [10.0]], [[], []], ["L"], None):
            with self.assertRaises(CandidateError):
                phys_coating.check_design(bad, 20, 1000.0, 400.0)

    def test_ame_table_parser_reads_the_fortran_columns(self):
        header = "1N-Z    N    Z   A  EL    O     MASS EXCESS       BINDING ENERGY/A        BETA-DECAY ENERGY         ATOMIC MASS"
        rows_2016 = ["   0    8    8   16 O         -4737.00135    0.00016   7976.206    0.000 B- -15412.184    8.265  15 994914.61960    0.00017",
                     "  14   18    4   22 Be  -nn   62220#       580#       4080#      26#    B-  27750#     800#     22 066800#       620#"]
        table = phys_nuclear.parse_table("\n".join([header] + rows_2016), (28, 41), (41, 52))
        self.assertEqual(table[(8, 8)], (16, "O", "-4737.00135", "0.00016"))
        self.assertIn("#", table[(18, 4)][2])  # estimated values keep their marker
        self.assertEqual(phys_nuclear.rms([3.0, 4.0]), math.sqrt(12.5))

    def test_nuclear_split(self):
        payload = prepared(phys_nuclear.load)
        training, targets = payload["training"], payload["targets"]
        self.assertEqual((len(training), len(targets)), (2408, 71))
        self.assertFalse({(r[0], r[1]) for r in training} & {(r[0], r[1]) for r in targets})
        self.assertTrue(all(r[0] >= 8 and r[1] >= 8 and r[0] + r[1] == r[2] for r in training + targets))
        estimates = [r[6] - r[4] for r in targets if r[6] is not None]
        self.assertEqual(len(estimates), 69)
        self.assertAlmostEqual(phys_nuclear.rms(estimates), 213.42, delta=0.01)  # the context value of the statement

    def test_spin_glass_cut_weight(self):
        instance = {"nodes": 4, "edges": [[0, 1, 5], [1, 2, -3], [2, 3, 7], [3, 0, 2]]}
        for spins in itertools.product([-1, 1], repeat=4):
            cut = sum(w for a, b, w in instance["edges"] if spins[a] != spins[b])
            self.assertEqual(phys_spinglass.cut_weight(instance, list(spins)), cut)
            energy = sum(w * spins[a] * spins[b] for a, b, w in instance["edges"])  # H = -sum J s s with J = -w
            self.assertEqual(energy, sum(e[2] for e in instance["edges"]) - 2 * cut)
        for bad in ([0, 1, 1, 1], [2, 1, 1, 1], [0.5, 1, 1, 1], [1, 1, 1], ["1", "1", "1", "1"]):
            with self.assertRaises(CandidateError):
                phys_spinglass.cut_weight(instance, bad)

    def test_spin_glass_instances_are_tori_with_the_tabulated_optima(self):
        for dimension in (2, 3):
            for name, instance in prepared(lambda: phys_spinglass.load_instances(dimension)):
                side, nodes = instance["side"], instance["nodes"]
                self.assertEqual((nodes, len(instance["edges"])), (side ** dimension, dimension * side ** dimension))
                self.assertEqual(instance["optimum"], phys_spinglass.OPTIMA[name][2])
                coords = lambda v: np.array(np.unravel_index(v, (side,) * dimension))
                for a, b, _ in instance["edges"]:
                    step = np.minimum((coords(a) - coords(b)) % side, (coords(b) - coords(a)) % side)
                    self.assertEqual(sorted(step.tolist()), [0] * (dimension - 1) + [1])
                self.assertEqual(len({(min(a, b), max(a, b)) for a, b, _ in instance["edges"]}), dimension * nodes)


@unittest.skipUnless(HAVE_SCIPY, "phys_quantum and phys_photonic import SciPy")
class QuantumAndPhotonics(unittest.TestCase):
    def test_known_circuits_reach_distance_zero_within_budget(self):
        from autoresearch_suite import phys_quantum
        targets = {name: (qubits, unitary, budget) for name, qubits, unitary, budget, _ in phys_quantum.TARGETS}
        h = lambda q: ["u3", q, math.pi / 2, 0.0, math.pi]
        t = lambda q, sign=1: ["u3", q, 0.0, 0.0, sign * math.pi / 4]
        circuits = {"cnot": [["cx", 0, 1]], "swap": [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]],
                    "toffoli": [h(2), ["cx", 1, 2], t(2, -1), ["cx", 0, 2], t(2), ["cx", 1, 2], t(2, -1), ["cx", 0, 2], t(1), t(2), h(2),
                                ["cx", 0, 1], t(0), t(1, -1), ["cx", 0, 1]]}
        for name, gates in circuits.items():
            qubits, unitary, budget = targets[name]
            matrix, cnots = phys_quantum.build_circuit(gates, qubits, budget)
            self.assertEqual(cnots, budget)
            self.assertLess(phys_quantum.distance(unitary, matrix), 1e-14)
        self.assertEqual(phys_quantum.distance(np.eye(4), phys_quantum.CNOT), 0.5)
        for bad in ([["cx", 0, 1]] * 2, [["cx", 0, 0]], [["cx", 0, 2]], [["u3", 0.0, 0, 0, 0]], [["u3", 0, math.nan, 0, 0]], [["cz", 0, 1]]):
            with self.assertRaises(CandidateError):
                phys_quantum.build_circuit(bad, 2, 1)

    def test_product_formula_applies_stage_one_first_on_pauli_operators(self):
        import scipy.linalg as sla
        from autoresearch_suite import phys_quantum
        pauli = {"x": np.array([[0, 1], [1, 0]], complex), "y": np.array([[0, -1j], [1j, 0]]), "z": np.diag([1.0 + 0j, -1.0])}
        site = lambda p, j: np.kron(np.kron(np.eye(2 ** j), pauli[p]), np.eye(2 ** (5 - j)))
        bond = lambda j: site("x", j) @ site("x", j + 1) + site("y", j) @ site("y", j + 1) + 0.8 * site("z", j) @ site("z", j + 1)
        a_part = bond(0) + bond(2) + bond(4) + 0.3 * (site("z", 0) + site("z", 2) + site("z", 4))
        b_part = bond(1) + bond(3) + 0.3 * (site("z", 1) + site("z", 3) + site("z", 5))
        groups = phys_quantum.heisenberg_groups(6, 0.8, 0.3)
        self.assertTrue(np.allclose(groups[0], a_part) and np.allclose(groups[1], b_part))
        rng = np.random.default_rng(12)
        a, b = rng.normal(size=12), rng.normal(size=12)
        a, b = a / a.sum(), b / b.sum()
        product = np.eye(64, dtype=complex)
        for ak, bk in zip(a, b):  # stage k acts on the state after stages 1..k-1; A factor before B factor
            product = sla.expm(-0.5j * bk * b_part) @ sla.expm(-0.5j * ak * a_part) @ product
        expected = np.linalg.norm(product - sla.expm(-0.5j * (a_part + b_part)), 2)
        self.assertAlmostEqual(phys_quantum.splitting_error(a, b, groups, [0.5])[0], expected, delta=1e-10)
        strang = phys_quantum.splitting_error([1 / 12] * 12, [1 / 6, 0.0] * 6, groups, (0.25, 0.5, 1.0))
        self.assertAlmostEqual(float(np.mean(strang)), 0.046156762743, delta=1e-10)

    def test_plane_wave_solver_on_the_empty_lattice_and_the_rod_crystal(self):
        from autoresearch_suite import phys_photonic
        uniform = np.ones((32, 32), dtype=int)  # homogeneous epsilon 12: omega = |k + G| / (2 pi sqrt(12))
        for k in ((math.pi, 0.0), (math.pi, math.pi), (0.3, 0.2)):
            exact = sorted(math.hypot(k[0] + 2 * math.pi * m, k[1] + 2 * math.pi * n) for m in range(-3, 4) for n in range(-3, 4))[:5]
            bands = phys_photonic.tm_bands(uniform, [k], 5)[0]
            self.assertTrue(np.allclose(bands, np.array(exact) / (2 * math.pi * math.sqrt(12.0)), atol=1e-9))
        rod = np.array([[1 if ((i + 0.5) / 32 - 0.5) ** 2 + ((j + 0.5) / 32 - 0.5) ** 2 < 0.04 else 0 for j in range(32)] for i in range(32)])
        self.assertTrue(np.array_equal(phys_photonic.check_cell(rod.tolist()), rod))
        samples = phys_photonic.brillouin_samples()
        self.assertEqual(len(samples), 30)
        coarse = phys_photonic.tm_bands(rod, samples, 5)
        fine = phys_photonic.tm_bands(rod, [samples[8], samples[16]], 5, cutoff=11)  # X and M with 529 plane waves
        self.assertTrue(np.all(coarse[[8, 16]] - fine >= -1e-9) and np.all(coarse[[8, 16]] - fine < 5e-4))  # variational, converged
        first, fourth = phys_photonic.gap_ratio(coarse, 1)[0], phys_photonic.gap_ratio(coarse, 4)[0]
        self.assertAlmostEqual(first, 0.3910, delta=5e-4)  # MPB tutorial, resolution 32: 0.3895
        self.assertAlmostEqual(fourth, 0.0431, delta=5e-4)  # MPB tutorial, resolution 32: 0.0384
        with self.assertRaises(CandidateError):
            phys_photonic.check_cell(np.triu(np.ones((32, 32), dtype=int)).tolist())


if __name__ == "__main__":
    unittest.main()
