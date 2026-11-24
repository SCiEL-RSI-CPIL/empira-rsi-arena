"""Graph, number-theory, geometry and analysis verifiers: corrected numerics and reference objects.

Three things are pinned here. The cosine minimum is certified even when many minima are equal, and
the distance product does not depend on the scale of the point set. Published optimal or
best-known constructions verify to their published values. Where a statement names an optimum of
a pinned instance, an explicit object attains it and the seed stays strictly worse.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
import json
from math import cos, pi, sin, sqrt
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "suite/_lib")]
from autoresearch_suite import runtime, science_prize_c, science_prize_g

# 16 frequencies with common factor 8: every minimum of the sum occurs 16 times on the circle
REPEATED_MINIMA = [8*b for b in (1, 2, 4, 6, 7, 9, 10, 11, 13, 14, 17, 18, 20, 22, 23, 24)]
# Best octagon found by local optimisation; Cambie, Decadt, Dong, Hu and Tang (arXiv:2603.07088, Table 1) list 1.250472
OCTAGON = [[0.2050993383, 0.943820548], [-0.454568023, 0.6392400819], [-0.9687308661, 0.1088156208],
           [-0.9574404324, -0.6836011189], [-0.163180553, -1.0219795237], [0.6997035962, -0.9940562871],
           [0.7097786138, 0.4211072825], [0.9970692067, -0.2594642648]]
# Overmars, Finding sets of points without empty convex 6-gons (UU-CS-2001-07), Figure 1
OVERMARS = [(1, 1260), (16, 743), (22, 531), (37, 0), (306, 592), (310, 531), (366, 552), (371, 487), (374, 525),
            (392, 575), (396, 613), (410, 539), (416, 550), (426, 526), (434, 552), (436, 535), (446, 565),
            (449, 518), (450, 498), (453, 542), (458, 526), (489, 537), (492, 502), (496, 579), (516, 467),
            (552, 502), (754, 697), (777, 194), (1259, 320)]
# Kreisel and Kurz (arXiv:0804.1303), Figure 1: points (x, y*sqrt(2002))
HEPTAGON = [[[0, 1], [0, 1]], [[22270, 1], [0, 1]], [[26127018, 2227], [932064, 2227]], [[245363, 17], [3144, 17]],
            [[17615968, 2227], [238464, 2227]], [[56068, 17], [3144, 17]], [[19079044, 2227], [-54168, 2227]]]
# OEIS A172992, best known integer configuration for 16 points (D. Kamenetsky's file): 14 four-point lines
FOUR_POINT_LINES = [(34, 35), (28, 11), (24, 25), (36, 13), (52, 17), (31, 23), (4, 5), (40, 29), (40, 59), (58, 41),
                    (12, 37), (10, 29), (20, 29), (16, 53), (28, 29), (22, 32)]
# A self-orthogonal Latin square of order 10 (cyclic over Z_9 with one fixed symbol): orthogonal to its transpose
SQUARE = [[0, 9, 1, 6, 5, 3, 8, 2, 4, 7], [5, 1, 9, 2, 7, 6, 4, 0, 3, 8], [4, 6, 2, 9, 3, 8, 7, 5, 1, 0],
          [2, 5, 7, 3, 9, 4, 0, 8, 6, 1], [7, 3, 6, 8, 4, 9, 5, 1, 0, 2], [1, 8, 4, 7, 0, 5, 9, 6, 2, 3],
          [3, 2, 0, 5, 8, 1, 6, 9, 7, 4], [8, 4, 3, 1, 6, 0, 2, 7, 9, 5], [9, 0, 5, 4, 2, 7, 1, 3, 8, 6],
          [6, 7, 8, 0, 1, 2, 3, 4, 5, 9]]
COPRIME_7372 = [1, 17, 29, 37, 41, 43, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
                139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 217, 223, 227, 229, 233, 235, 239,
                241, 243, 247, 251, 253, 256]
ROUGH_WITNESSES = [10008, 10019, 10038, 10057, 10063, 10068, None, 10081, 10092, 10097, None, 10109, 10117, None, 10140,
                   10147, 10153, None, None, None, None, 10183, 10201, 10217, 10229, None, 10249, None, 10261, None,
                   10272, 10277]
# An acyclic 4-colouring of the circulant C50(1,7); 3 colours are impossible for a 4-regular graph
# (three bicoloured forests have at most 2n-3 edges)
ACYCLIC_FOUR = [1, 3, 2, 1, 2, 0, 1, 2, 1, 0, 3, 0, 3, 0, 3, 0, 2, 1, 3, 1, 2, 1, 2, 1, 3, 2, 3, 0, 3, 1, 0, 1, 0, 1, 3, 2, 0, 3,
                2, 1, 2, 0, 3, 2, 0, 3, 2, 3, 2, 0]
LONG_PATH = [6, 36, 81, 10, 0, 80, 53, 68, 42, 48, 22, 33, 55, 24, 58, 21, 11, 18, 3, 12, 65, 60, 90, 69, 40, 97, 46,
             50, 59, 49, 19, 38, 35, 17]


def parameters(task):
    return json.loads((ROOT / "suite" / task / "problem.json").read_text(encoding="utf-8"))["parameters"]


def seed(task):
    namespace = {}
    exec((ROOT / "suite" / task / "initial_program.py").read_text(encoding="utf-8"), namespace)
    return namespace["construct"]()


def exact(number, task, candidate):
    return science_prize_c.evaluate(number, candidate, parameters(task))[0]


def boroczky_twelve():
    """Six vertices of an affinely regular hexagon and the six directions of its chords, made affine."""
    projective = [(x, y, 1) for x, y in ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))]
    projective += [(x, y, 0) for x, y in ((1, 2), (0, 1), (-1, 1), (1, 0), (2, 1), (1, 1))]
    moved = [(x, y, z+3*x-5*y) for x, y, z in projective]          # no point stays at infinity
    return [[840*x//z, 840*y//z] for x, y, z in moved]              # 840 = lcm of the third coordinates


class CosineMinimumTests(unittest.TestCase):
    def test_repeated_minima_do_not_hide_the_global_one(self):
        # one period of the sum, sampled finely and polished: independent of the library routine
        base = [a//8 for a in REPEATED_MINIMA]
        f = lambda s: sum(cos(b*s) for b in base)
        steps = 20000
        index = min(range(steps), key=lambda i: f(2*pi*i/steps))
        low, high = 2*pi*(index-1)/steps, 2*pi*(index+1)/steps
        for _ in range(200):
            first, second = low+(high-low)/3, high-(high-low)/3
            low, high = (low, second) if f(first) < f(second) else (first, high)
        expected = f((low+high)/2)
        value, details = science_prize_g.chowla_cosine(REPEATED_MINIMA, k=16, maximum=200)
        self.assertAlmostEqual(details["minimum"], expected, places=9)
        self.assertAlmostEqual(expected, -3.236006369791, places=9)
        # the twelve-minima refinement returned -3.235438605 here (score -0.808859651)
        self.assertEqual(value, -0.809001592)

    def test_value_is_bracketed_by_a_dense_grid(self):
        rng = random.Random(4)
        for _ in range(6):
            frequencies = rng.sample(range(1, 201), 16)
            steps = 12000
            grid = min(sum(cos(a*pi*i/steps) for a in frequencies) for i in range(steps+1))
            slack = sum(a*a for a in frequencies)*(pi/steps)**2/8
            value = science_prize_g.cosine_minimum(frequencies, 128*200)
            self.assertLessEqual(value, grid+1e-12)
            self.assertLessEqual(grid-value, slack)

    def test_seed_score_is_unchanged(self):
        self.assertEqual(science_prize_g.chowla_cosine(seed("science-chowla-cosine-sixteen-frequencies"), k=16, maximum=200)[0],
                         -1.023864576)


class DistanceProductTests(unittest.TestCase):
    def test_score_does_not_depend_on_scale_or_position(self):
        octagon = [(cos(2*pi*k/8), sin(2*pi*k/8)) for k in range(8)]
        for scale, shift in ((1.0, 0.0), (1e-10, 0.0), (1e-160, 0.0), (1e5, -7e5), (3.0, 5.0)):
            points = [[scale*x+shift, scale*y-shift] for x, y in octagon]
            self.assertAlmostEqual(science_prize_g.distance_product(points, n=8)[0], 1.0, places=9, msg=str(scale))
        small = [[1e-12*x, 1e-12*y] for x, y in OCTAGON]
        self.assertAlmostEqual(science_prize_g.distance_product(small, n=8)[0], science_prize_g.distance_product(OCTAGON, n=8)[0], places=9)

    def test_only_equal_points_are_not_distinct(self):
        with self.assertRaises(runtime.CandidateError):
            science_prize_g.distance_product([[0.0, 0.0]]*2+[[float(i), 1.0] for i in range(6)], n=8)
        close = [[0.0, 0.0], [1e-200, 0.0]]+[[float(i), 1.0] for i in range(6)]
        self.assertEqual(science_prize_g.distance_product(close, n=8)[0], 0.0)

    def test_published_values(self):
        kite = [[0.0, 0.0], [2.0, 0.0], [sqrt(3), 1.0], [sqrt(3), -1.0]]            # proven optimum for n = 4
        self.assertAlmostEqual(science_prize_g.distance_product(kite, n=4)[0], 16*(7-4*sqrt(3)), places=10)
        self.assertAlmostEqual(science_prize_g.distance_product(OCTAGON, n=8)[0], 1.250472, places=6)
        self.assertEqual(science_prize_g.distance_product(seed("science-distance-product-fixed-diameter-8"), n=8)[0], 1.0)


class GeometryReferenceTests(unittest.TestCase):
    def test_twelve_integer_points_with_six_ordinary_lines(self):
        points = boroczky_twelve()
        self.assertEqual(science_prize_g.ordinary_lines(points, **parameters("science-ordinary-lines-twelve-rational-points"))[0], 6)
        collinear = lambda a, b, c: (b[0]-a[0])*(c[1]-a[1]) == (b[1]-a[1])*(c[0]-a[0])
        ordinary = sum(not any(collinear(a, b, c) for c in points if c not in (a, b)) for a, b in combinations(points, 2))
        self.assertEqual(ordinary, 6)
        self.assertGreater(science_prize_g.ordinary_lines(seed("science-ordinary-lines-twelve-rational-points"), n=12)[0], 6)

    def test_best_known_integer_four_point_lines(self):
        task = "science-four-point-lines-sixteen-integer-points"
        self.assertEqual(science_prize_g.four_point_lines([list(p) for p in FOUR_POINT_LINES], **parameters(task))[0], 14)
        self.assertLess(science_prize_g.four_point_lines(seed(task), **parameters(task))[0], 14)

    def test_overmars_set_has_no_empty_hexagon_and_cannot_grow(self):
        task = "science-point-sets-without-empty-hexagon"
        self.assertEqual(science_prize_g.empty_hexagon_free([list(p) for p in OVERMARS], **parameters(task))[0], 29)
        with self.assertRaises(runtime.CandidateError):                      # every 30-point set has one
            science_prize_g.empty_hexagon_free([list(p) for p in OVERMARS]+[[600, 601]], **parameters(task))

    def test_integral_heptagon_and_its_perturbation(self):
        task = "science-integral-point-sets-general-position"
        self.assertEqual(science_prize_g.integral_point_set({"D": 2002, "points": HEPTAGON}, **parameters(task))[0], 7)
        moved = [row[:] for row in HEPTAGON]
        moved[3] = [[245364, 17], [3144, 17]]
        with self.assertRaises(runtime.CandidateError):
            science_prize_g.integral_point_set({"D": 2002, "points": moved}, **parameters(task))

    def test_penny_spiral_reaches_harborth(self):
        form = lambda a, b: (a[0]-b[0])**2+(a[0]-b[0])*(a[1]-b[1])+(a[1]-b[1])**2
        points = [(0, 0)]
        while len(points) < 32:                                              # greedy hexagonal spiral
            border = {(p[0]+d[0], p[1]+d[1]) for p in points for d in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))}
            points.append(max(border-set(points), key=lambda c: (sum(form(c, p) == 1 for p in points), -form(c, (0, 0)), c)))
        self.assertEqual(science_prize_g.penny_contacts([list(p) for p in points], n=32, bound=64)[0], 76)

    def test_flattest_polynomial_with_24_coefficients(self):
        signs = [1 if c == "+" else -1 for c in "+++++--+++----+--++-+-+-"]    # exhaustive optimum, recomputed in the audit
        self.assertAlmostEqual(science_prize_g.littlewood_sup_norm(signs, n=24)[0], 1.2583758, places=7)


class ReferenceValueTests(unittest.TestCase):
    """Objects that attain the reference values named in the statements (published, or exhaustive on the pinned instance)."""

    def test_published_constructions(self):
        self.assertEqual(exact(653, "science-distinct-subset-product-sets",
                               [5, 6, 7, 11, 12, 13, 17, 19, 21, 23, 25, 27, 28, 29, 31, 32, 37]), 17)      # OEIS A401472(40)
        self.assertEqual(exact(717, "science-pairwise-sum-triple-free-sets", list(range(16, 33))+list(range(64, 129))), 82)
        self.assertEqual(exact(697, "science-short-square-product-extension", [220, 221, 224, 231, 234, 238]), 28)
        self.assertEqual(exact(730, "science-maximum-sum-pairwise-coprime-integers", COPRIME_7372), 7372)
        self.assertEqual(exact(592, "science-mutually-orthogonal-latin-squares", [SQUARE, [list(r) for r in zip(*SQUARE)]]), 2)
        self.assertEqual(exact(572, "science-uniform-no-singleton-intersections",
                               [[0, 1, a, b] for a, b in combinations(range(2, 16), 2)]), 91)
        self.assertEqual(exact(610, "science-diameter-two-edge-critical-graphs", [[i, j] for i in range(8) for j in range(8, 16)]), 64)

    def test_optima_of_pinned_instances(self):
        self.assertEqual(exact(506, "science-cochromatic-homogeneous-partition",
                               [[0, 1, 7, 8], [2, 3, 5, 12], [4, 6, 9, 10], [11, 13]]), 4)
        self.assertEqual(exact(518, "science-ramsey-induced-degree-diversity", [0, 1, 2, 3, 4, 5, 6, 11]), 5)
        self.assertEqual(exact(521, "science-common-vertex-edge-disjoint-cycles",
                               [[(step*i) % 21 for i in range(21)] for step in (1, 2, 5)]), 3)
        self.assertEqual(exact(554, "science-rough-numbers-in-consecutive-prime-gaps", ROUGH_WITNESSES), 22)
        self.assertEqual(exact(578, "science-minimum-span-distinct-multiples",
                               [8226, 8230, 8218, 8220, 8232, 8228, 8216, 8217, 8225, 8229]), 16)
        self.assertEqual(exact(580, "science-distinct-multiple-worst-window", 121), 22)
        self.assertEqual(exact(614, "science-random-graph-hamiltonian-arrival",
                               [0, 2, 5, 15, 16, 10, 13, 21, 22, 19, 11, 4, 7, 6, 23, 1, 3, 14, 9, 18, 8, 20, 17, 12]), 45)
        self.assertEqual(exact(615, "science-random-hypergraph-matching-arrival",
                               [[0, 3, 5], [1, 10, 15], [2, 7, 9], [4, 8, 13], [6, 14, 17], [11, 12, 16]]), 31)
        self.assertEqual(exact(655, "science-acyclic-vertex-coloring", ACYCLIC_FOUR), 4)
        two_k33 = [[u+shift, w+shift] for shift in (0, 6) for u in range(3) for w in range(3, 6)]
        self.assertEqual(exact(619, "science-girth-degree-cycle-length-diversity", two_k33), 2)
        self.assertEqual(exact(748, "science-sparse-random-graph-long-path", LONG_PATH), 34)
        value, details = science_prize_c.evaluate(679, [9174353, 6116235], parameters("science-equal-divisor-sum-ratio-approximation"))
        self.assertEqual(Fraction(details["error_numerator"], details["error_denominator"]), Fraction(1, 12232470))
        self.assertEqual(value, 1/12232470)

    def test_cochromatic_number_of_the_pinned_graph_is_four(self):
        p = parameters("science-cochromatic-homogeneous-partition")
        n, neighbours = p["n"], [0]*p["n"]
        for u, v in p["edges"]:
            neighbours[u] |= 1 << v
            neighbours[v] |= 1 << u
        clique, independent = [True]*(1 << n), [True]*(1 << n)
        for s in range(1, 1 << n):
            low = s & -s
            rest, v = s ^ low, low.bit_length()-1
            clique[s] = clique[rest] and neighbours[v] & rest == rest
            independent[s] = independent[rest] and not neighbours[v] & rest
        # three homogeneous parts would need two parts that cover everything outside a third one
        two = [False]*(1 << n)
        for s in range(1 << n):
            sub = s
            while True:
                if (clique[sub] or independent[sub]) and (clique[s ^ sub] or independent[s ^ sub]):
                    two[s] = True
                    break
                if not sub:
                    break
                sub = (sub-1) & s
        full = (1 << n)-1
        self.assertFalse(any((clique[s] or independent[s]) and two[full ^ s] for s in range(1 << n)))

    def test_distinct_multiple_windows_match_oeis_a390246(self):
        # A390246(n) = f(n, n)+1 for n = 1..12
        self.assertEqual([science_prize_c.matching_window(n, n)+1 for n in range(1, 13)], [2, 3, 4, 6, 6, 9, 9, 11, 13, 15, 15, 17])

    def test_no_seed_attains_a_known_optimum(self):
        known = {"science-cochromatic-homogeneous-partition": (506, 4, "minimize"),
                 "science-ramsey-induced-degree-diversity": (518, 5, "maximize"),
                 "science-common-vertex-edge-disjoint-cycles": (521, 3, "maximize"),
                 "science-rough-numbers-in-consecutive-prime-gaps": (554, 22, "maximize"),
                 "science-uniform-no-singleton-intersections": (572, 91, "maximize"),
                 "science-minimum-span-distinct-multiples": (578, 16, "minimize"),
                 "science-distinct-multiple-worst-window": (580, 22, "maximize"),
                 "science-diameter-two-edge-critical-graphs": (610, 64, "maximize"),
                 "science-random-graph-hamiltonian-arrival": (614, 45, "minimize"),
                 "science-random-hypergraph-matching-arrival": (615, 31, "minimize"),
                 "science-girth-degree-cycle-length-diversity": (619, 2, "minimize"),
                 "science-distinct-subset-product-sets": (653, 17, "maximize"),
                 "science-acyclic-vertex-coloring": (655, 4, "minimize"),
                 "science-equal-divisor-sum-ratio-approximation": (679, 1/12232470, "minimize"),
                 "science-three-chromatic-triple-system": (688, 7, "minimize"),
                 "science-short-square-product-extension": (697, 28, "minimize"),
                 "science-pairwise-sum-triple-free-sets": (717, 82, "maximize"),
                 "science-maximum-sum-pairwise-coprime-integers": (730, 7372, "maximize"),
                 "science-sparse-random-graph-long-path": (748, 34, "maximize")}
        for task, (number, reference, direction) in known.items():
            with self.subTest(task=task):
                score = exact(number, task, seed(task))
                self.assertTrue(score > reference if direction == "minimize" else score < reference, (score, reference))
        self.assertEqual(exact(506, "science-cochromatic-homogeneous-partition", seed("science-cochromatic-homogeneous-partition")), 7)


if __name__ == "__main__":
    unittest.main()
