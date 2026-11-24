"""Independent oracles, constructions and malformed inputs for group G."""
from __future__ import annotations

from fractions import Fraction
import itertools
from math import cos, gcd, lcm, pi, sin, sqrt
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"suite/_lib"))
sys.path.insert(0, str(ROOT/"tests"))
from autoresearch_suite import runtime
from autoresearch_suite import science_prize_g as v
from suite_test_support import specifications

# The specification list of the generator script is not part of this library: the same fields are read
# from the packages (seed, parameters, direction, verifier) and the catalog (label, reference value).
SPECS = specifications("prize_g")
FUNCTIONS = {spec["key"]: spec["function"] for spec in SPECS}


def run_seed(spec):
    namespace = {}
    exec(spec["seed"], namespace)
    return namespace["construct"]()


def lines_by_triples(points):
    """Group collinear points by brute force over pairs (independent of the verifier's routine)."""
    n, found = len(points), set()
    for i, j in itertools.combinations(range(n), 2):
        (x1, y1), (x2, y2) = points[i], points[j]
        found.add(frozenset(k for k in range(n) if (x2-x1)*(points[k][1]-y1) == (y2-y1)*(points[k][0]-x1)))
    return found


def twelve_torsion_points():
    """Rational points of order dividing 12 on a Tate-normal-form curve, moved to the affine plane."""
    tau = Fraction(2)
    m = (3*tau-3*tau**2-1)/(tau-1)
    f = m/(1-tau)
    d = m+tau
    c = f*(d-1)
    b = c*d
    a1, a2, a3 = 1-c, -b, -b

    def add(p, q):
        if p is None:
            return q
        (x1, y1), (x2, y2) = p, q
        if x1 == x2 and y1+y2+a1*x2+a3 == 0:
            return None
        lam = (3*x1*x1+2*a2*x1-a1*y1)/(2*y1+a1*x1+a3) if x1 == x2 else (y2-y1)/(x2-x1)
        x3 = lam*lam+a1*lam-a2-x1-x2
        return x3, -(lam+a1)*x3-(y1-lam*x1)-a3

    base, point, torsion = (Fraction(0), Fraction(0)), None, []
    for _ in range(11):
        point = add(point, base)
        torsion.append((point[0], point[1], Fraction(1)))
    assert add(point, base) is None                                   # the base point has order 12
    torsion.append((Fraction(0), Fraction(1), Fraction(0)))            # the point at infinity
    matrix = [[1, 2, 3], [-2, 1, 5], [3, -1, 7]]
    image = [[sum(matrix[r][k]*p[k] for k in range(3)) for r in range(3)] for p in torsion]
    affine = [(x/z, y/z) for x, y, z in image]
    scale = lcm(*[q.denominator for p in affine for q in p])
    return [[int(x*scale), int(y*scale)] for x, y in affine]


class Lines(unittest.TestCase):
    def test_line_counts_against_triples(self):
        rng = random.Random(1)
        for _ in range(40):
            pts = list({(rng.randrange(6), rng.randrange(6)) for _ in range(12)})[:12]
            if len(pts) < 12:
                continue
            lines = lines_by_triples(pts)
            self.assertEqual(v.orchard_lines([list(p) for p in pts])[0], sum(len(l) == 3 for l in lines))
            self.assertEqual(v.ordinary_lines([list(p) for p in pts])[0], sum(len(l) == 2 for l in lines))

    def test_rational_orchard_configuration_attains_19(self):
        points = twelve_torsion_points()
        self.assertEqual(v.orchard_lines(points, bound=10**30)[0], 19)

    def test_four_point_lines_and_general_position(self):
        grid = [[x, y] for x in range(4) for y in range(4)]
        self.assertEqual(v.four_point_lines(grid)[0], 10)
        with self.assertRaises(runtime.CandidateError):
            v.four_point_lines([[x, 0] for x in range(5)] + [[x, y] for x in range(4) for y in range(1, 4)][:11])
        rng = random.Random(2)
        for _ in range(20):
            pts = []
            while len(pts) < 9:
                p = (rng.randrange(5), rng.randrange(5))
                if p not in pts and all(len(l) <= 3 for l in lines_by_triples(pts+[p])):
                    pts.append(p)
            best = max(len(s) for r in range(10) for s in itertools.combinations(pts, r)
                       if all(len(l) <= 2 for l in lines_by_triples(list(s))))
            self.assertEqual(v.general_position_subset([list(p) for p in pts], n=9, bound=5)[0], best)


class EmptyPolygons(unittest.TestCase):
    def test_against_hull_enumeration(self):
        def hull(points):
            points = sorted(points)

            def half(seq):
                h = []
                for p in seq:
                    while len(h) >= 2 and v._cross(h[-2], h[-1], p) <= 0:
                        h.pop()
                    h.append(p)
                return h
            return half(points)[:-1]+half(points[::-1])[:-1]
        rng = random.Random(3)
        for _ in range(150):
            pts = list({(rng.randrange(40), rng.randrange(40)) for _ in range(rng.randint(5, 10))})
            if not v._no_three_collinear(pts):
                continue
            for k in (4, 5, 6):
                expected = False
                for subset in itertools.combinations(pts, k):
                    h = hull(list(subset))
                    if len(h) == k and not any(all(v._cross(h[i], h[(i+1) % k], p) > 0 for i in range(k)) for p in pts if p not in subset):
                        expected = True
                        break
                self.assertEqual(v.has_empty_convex(pts, k), expected)
        convex = [[round(1000*cos(2*pi*i/6)), round(1000*sin(2*pi*i/6))] for i in range(6)]
        with self.assertRaises(runtime.CandidateError):
            v.empty_hexagon_free(convex)


class IntegralSets(unittest.TestCase):
    def test_certificates(self):
        seed = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        self.assertEqual(v.integral_point_set({"D": 1, "points": [[[x, 1], [y, 1]] for x, y in seed]})[0], 5)
        triangle = {"D": 3, "points": [[[0, 1], [0, 1]], [[2, 1], [0, 1]], [[1, 1], [1, 1]]]}   # equilateral, side 2
        self.assertEqual(v.integral_point_set(triangle)[0], 3)
        rejects = [
            {"D": 1, "points": [[[0, 1], [0, 1]], [[3, 1], [0, 1]], [[0, 1], [4, 1]], [[3, 1], [4, 1]]]},   # rectangle: concyclic
            {"D": 1, "points": [[[0, 1], [0, 1]], [[3, 1], [0, 1]], [[6, 1], [0, 1]]]},                      # collinear
            {"D": 1, "points": [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]},                      # irrational distance
            {"D": 4, "points": [[[0, 1], [0, 1]], [[2, 1], [0, 1]], [[1, 1], [1, 1]]]},                      # D not squarefree
            {"D": 1, "points": [[[0, 0], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}]                      # zero denominator
        for value in rejects:
            with self.subTest(value=value), self.assertRaises(runtime.CandidateError):
                v.integral_point_set(value)


class Contacts(unittest.TestCase):
    def test_harborth_formula_on_small_lattice_windows(self):
        window = [(a, b) for a in range(-2, 3) for b in range(-2, 3) if abs(a+b) <= 2]
        for n in range(2, 8):
            best = max(sum((p[0]-q[0])**2+(p[0]-q[0])*(p[1]-q[1])+(p[1]-q[1])**2 == 1 for p, q in itertools.combinations(s, 2))
                       for s in itertools.combinations(window, n))
            self.assertEqual(best, int(3*n-sqrt(12*n-3)))
            self.assertEqual(v.penny_contacts([list(p) for p in window[:n]], n=n)[0],
                             sum((p[0]-q[0])**2+(p[0]-q[0])*(p[1]-q[1])+(p[1]-q[1])**2 == 1 for p, q in itertools.combinations(window[:n], 2)))


class Analysis(unittest.TestCase):
    def test_distance_product_normalization(self):
        for n in (4, 6, 8):
            regular = [[cos(2*pi*k/n), sin(2*pi*k/n)] for k in range(n)]
            self.assertAlmostEqual(v.distance_product(regular, n=n)[0], 1.0, places=9)
            moved = [[3*x+5, 3*y-2] for x, y in regular]
            self.assertAlmostEqual(v.distance_product(moved, n=n)[0], 1.0, places=9)
        with self.assertRaises(runtime.CandidateError):
            v.distance_product([[0.0, 0.0]]*8)

    def test_cosine_and_polynomial_extrema_against_dense_grids(self):
        rng = random.Random(4)
        for _ in range(5):
            freqs = rng.sample(range(1, 60), 16)
            grid = min(sum(cos(a*t) for a in freqs) for t in (2*pi*i/200000 for i in range(200000)))
            self.assertAlmostEqual(v.chowla_cosine(freqs, maximum=60)[0], grid/4, places=5)
            signs = [rng.choice((-1, 1)) for _ in range(24)]
            top = max(abs(sum(c*complex(cos(k*t), sin(k*t)) for k, c in enumerate(signs))) for t in (2*pi*i/60000 for i in range(60000)))
            self.assertAlmostEqual(v.littlewood_sup_norm(signs, n=24)[0], top/sqrt(24), places=5)
        best = [1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1]   # local exhaustive optimum
        self.assertAlmostEqual(v.littlewood_sup_norm(best, n=24)[0], 1.2583758, places=6)
        self.assertAlmostEqual(v.littlewood_sup_norm([1]*24, n=24)[0], sqrt(24), places=9)


class Specifications(unittest.TestCase):
    def test_seeds_are_valid_and_do_not_claim_records(self):
        for spec in SPECS:
            with self.subTest(task=spec["id"]):
                score = getattr(v, FUNCTIONS[spec["key"]])(run_seed(spec), **spec["parameters"])[0]
                reference = spec["sota"]["value"]
                if reference is not None:
                    self.assertTrue(score < reference if spec["direction"] == "maximize" else score > reference)

    def test_labels_follow_reference_values(self):
        for spec in SPECS:
            self.assertEqual(spec["contribution"] == "rediscovery", spec["sota"]["kind"] == "optimum", spec["id"])

    def test_malformed_inputs(self):
        for value in [None, [[0, 0]]*12, [[0.5, 0]]+[[i, i*i] for i in range(11)], [[True, 0]]+[[i, i*i] for i in range(1, 12)], [[0, 0, 0]]*12]:
            with self.subTest(value=value), self.assertRaises(runtime.CandidateError):
                v.orchard_lines(value)
        with self.assertRaises(runtime.CandidateError):
            v.ordinary_lines([[i, 0] for i in range(12)])          # all collinear
        with self.assertRaises(runtime.CandidateError):
            v.chowla_cosine(list(range(1, 16))+[1])                # repeated frequency
        with self.assertRaises(runtime.CandidateError):
            v.littlewood_sup_norm([1]*23+[0])
        with self.assertRaises(runtime.CandidateError):
            v.penny_contacts([[0, 0]]*32)
        with self.assertRaises(runtime.CandidateError):
            v.distance_product([[float("nan"), 0.0]]+[[i, 0.0] for i in range(1, 8)])


if __name__ == "__main__":
    unittest.main()
