"""Regression tests of the evaluator audit for the number-theory and combinatorics construction tasks.

Each class pins one corrected defect or one reference value that the statements rely on. The oracles are
written here from the definitions; nothing is read from the network and only the standard library is used.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from math import gcd, isqrt
from pathlib import Path
from random import Random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "suite/_lib")]
from autoresearch_suite import runtime
from autoresearch_suite import science_prize_h as h
from autoresearch_suite import science_prize_i as i

LEHMER = [1, 1, 0, -1, -1, -1, -1, -1, 0, 1, 1]
PLASTIC = [-1, -1, 0, 1]


def multiply(a, b):
    out = [0]*(len(a)+len(b)-1)
    for j, x in enumerate(a):
        for k, y in enumerate(b):
            out[j+k] += x*y
    return out


def seed_of(task_id):
    namespace = {}
    exec((ROOT / "suite" / task_id / "initial_program.py").read_text(), namespace)
    return namespace["construct"]()


class MahlerMeasure(unittest.TestCase):
    """Repeated roots (cyclotomic or not) and monomials used to be scored wrongly or rejected."""

    def test_cyclotomic_factors_do_not_change_the_score(self):
        self.assertEqual(h.mahler_measure(LEHMER)[0], 1.1762808183)
        self.assertEqual(h.mahler_measure(multiply(LEHMER, [1, -2, 1]))[0], 1.1762808183)        # times (x-1)^2
        self.assertEqual(h.mahler_measure(multiply(PLASTIC, [-1, 3, -3, 1]))[0], 1.3247179572)    # times (x-1)^3
        self.assertEqual(h.mahler_measure(multiply(PLASTIC, multiply([1, -2, 1], [1, 2, 1])))[0], 1.3247179572)

    def test_repeated_noncyclotomic_factor(self):
        for g in ([-1, 1, 0, 1, 1, 1], PLASTIC, [-1, -1, 1], [1, 0, 1, 0, -1, 1]):
            square = multiply(g, g)
            self.assertTrue(all(abs(c) <= 3 for c in square), g)
            self.assertAlmostEqual(h.mahler_measure(square)[0], h.mahler_measure(g)[0]**2, delta=1e-9)

    def test_monomials(self):
        self.assertEqual(h.mahler_measure([0, 0, 0, 2])[0], 2.0)
        self.assertEqual(h.mahler_measure([0, -3])[0], 3.0)
        self.assertEqual(h.mahler_measure([0, 2, 2])[0], 2.0)                                    # 2x(x+1)
        for measure_one in ([0, 1], [0, 0, -1], [0, 1, 1], [1, 2, 1], [0, 0, -1, 1], [-1, 0, 0, 0, 1]):
            with self.assertRaises(runtime.CandidateError):
                h.mahler_measure(measure_one)

    def test_seed_and_bounds_unchanged(self):
        self.assertEqual(h.mahler_measure(seed_of("science-lehmer-small-mahler-measure-degree-12"))[0], 1.3247179572)
        for bad in ([1, 0], [4, 1], [1]*14, [1.0, 1], [True, 1]):
            with self.assertRaises(runtime.CandidateError):
                h.mahler_measure(bad)


class CoverCaps(unittest.TestCase):
    """The covered length may not pass the cap the statements give."""

    def test_double_cover_stops_at_the_cap(self):
        residues = [0, 0, 1, 1, 2, 3]                    # modulo 2, 3, 5, 7, 11, 13: 1, 2 and 3 are covered twice
        self.assertEqual(h.double_prime_cover(residues, largest_prime=13)[0], 3)
        score, report = h.double_prime_cover(residues, largest_prime=13, cap=2)
        self.assertEqual((score, report["capped"]), (2, True))

    def test_budget_cover_stops_at_the_cap(self):
        classes = [[2, 1], [3, 2], [5, 4]]               # covers 1..5 and not 6
        self.assertEqual(h.budget_prime_cover(classes, largest_prime=7, budget=Fraction(2))[0], 5)
        self.assertEqual(h.budget_prime_cover(classes, largest_prime=7, budget=Fraction(2), cap=4)[0], 4)

    def test_seeds_unchanged(self):
        self.assertEqual(h.double_prime_cover(seed_of("science-double-cover-primes-to-1000"))[0], 85)
        self.assertEqual(h.budget_prime_cover(seed_of("science-prime-covers-with-reciprocal-budget"))[0], 163)


class StrongEdgeColouring(unittest.TestCase):
    def test_conjectured_bound_for_odd_degree(self):
        score, report = i.strong_edge_colouring(seed_of("science-strong-edge-colouring-instances"))
        self.assertEqual(score, 61)
        self.assertEqual(report["erdos_nesetril_bounds"], [20, 29, 45])      # (5*25 - 10 + 1)/4 = 29 for degree 5


class ReferenceObjects(unittest.TestCase):
    """Published optimal or best-known constructions give their published values."""

    def test_extremal_two_bases_of_range_212(self):
        # Kohonen and Corander, J. Integer Seq. 17 (2014) 14.3.4, Table 1
        for text in ("1 3 4 6 10 13 15 21 29 37 45 53 61 69 77 85 91 93 96 100 102 103 105 106",
                     "1 3 4 6 10 13 15 21 29 37 45 53 61 69 77 85 93 97 99 102 103 104 106 108",
                     "1 3 4 6 10 13 15 21 29 37 45 53 61 69 77 85 93 97 99 102 103 106 108 112"):
            basis = [0]+[int(x) for x in text.split()]
            self.assertEqual(h.additive_basis(basis)[0], 25)
            with self.assertRaises(runtime.CandidateError):
                h.additive_basis(basis, n=213)

    def test_wichmann_ruler(self):
        marks = [0]
        for step in [1]*3+[4]+[7]*3+[15]*9+[8]*4+[1]*3:         # W(3, 9): 24 marks, length 198
            marks.append(marks[-1]+step)
        self.assertEqual(h.sparse_ruler(marks, n=198)[0], 24)
        self.assertEqual(h.sparse_ruler(marks+[200])[0], 25)
        with self.assertRaises(runtime.CandidateError):
            h.sparse_ruler(marks)

    def test_abbott_hanson_hypergraph(self):
        fano = [[0, 1, 2], [0, 3, 4], [0, 5, 6], [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]]
        edges = [e+[7+k, 12+k] for e in fano for k in range(5)]
        edges += [[(7 if c == 0 else 12)+k for k, c in enumerate(choice)] for choice in product((0, 1), repeat=5) if sum(choice) % 2 == 0]
        self.assertEqual(i.property_b({"n": 17, "edges": edges})[0], 51)
        masks = [sum(1 << v for v in e) for e in edges]
        full = (1 << 17)-1
        self.assertFalse(any(all(m & c and m & ~c & full for m in masks) for c in range(1 << 16)))    # every 2-colouring
        with self.assertRaises(runtime.CandidateError):
            i.property_b({"n": 17, "edges": edges[1:]})

    def test_published_sets(self):
        self.assertEqual(h.no_square_subset_sum([11, 18, 29, 30, 47, 58, 65, 76, 87, 94, 105, 123, 134])[0], 13)   # OEIS A372040
        family = [list(s) for k in range(9) for s in combinations(range(1, 9), k) if len(set(s)-{1}) >= 5 or len(s) <= 1]
        self.assertEqual(i.forbidden_intersection(family)[0], 67)                                              # Frankl-Furedi
        k4 = [[4*b+x, 4*b+y] for b in range(8) for x in range(4) for y in range(x+1, 4)]
        self.assertEqual(i.cycle_clique_ramsey({"n": 32, "edges": k4})[0], 32)
        fan = [[a, b] for a in range(10) for b in range(10, 16)]+[[2*k, 2*k+1] for k in range(5)]
        self.assertEqual(i.triangle_degree_sum({"n": 16, "edges": fan})[0], 24)


class KnownCeilings(unittest.TestCase):
    """Tasks whose maximum is fixed by a cap or by a short exact computation (stated as their reference)."""

    def test_totient_run_cap_is_reached(self):
        self.assertEqual(i.distinct_totient_run({"start": 1173985, "length": 200})[0], 200)
        with self.assertRaises(runtime.CandidateError):
            i.distinct_totient_run({"start": 1173985, "length": 201})

    def test_powerful_pair_cap_is_reached(self):
        # 5^3 13^2 17 29 37 41 53 61 73 89 is a sum of two positive squares in 1536 ways (products of Gaussian primes)
        def two_squares(p):
            return next((x, isqrt(p-x*x)) for x in range(1, isqrt(p)+1) if isqrt(p-x*x)**2 == p-x*x)

        def times(u, w):
            return (u[0]*w[0]-u[1]*w[1], u[0]*w[1]+u[1]*w[0])

        def power(u, e):
            out = (1, 0)
            for _ in range(e):
                out = times(out, u)
            return out
        factors = {5: 3, 13: 2, 17: 1, 29: 1, 37: 1, 41: 1, 53: 1, 61: 1, 73: 1, 89: 1}
        total, reps = 1, [(1, 0)]
        for p, e in factors.items():
            total *= p**e
            x, y = two_squares(p)
            reps = [times(r, times(power((x, y), k), power((x, -y), e-k))) for r in reps for k in range(e+1)]
        pairs = sorted({tuple(sorted((a*a, b*b))) for a, b in reps})
        self.assertEqual((total, len(pairs)), (331854559557386125, 1536))
        self.assertTrue(all(0 < a < b and a+b == total and isqrt(a)**2 == a and isqrt(b)**2 == b for a, b in pairs))
        # Squares are powerful, so any 1000 of the 1536 pairs reach the cap; the checker is run on a sample
        # (factoring all 2000 numbers takes a few seconds) and on one pair too many.
        self.assertEqual(i.powerful_pairs({"N": total, "pairs": [list(p) for p in pairs[::16]]})[0], 96)
        with self.assertRaises(runtime.CandidateError):
            i.powerful_pairs({"N": total, "pairs": [list(p) for p in pairs[:1001]]})

    def test_coprime_reciprocal_optimum(self):
        # Every a < 512 has at most one prime factor >= 23, so an exact dynamic programme over the subsets of
        # {2, ..., 19} gives the maximum of sum 1/(512 - a) over pairwise coprime sets.
        small = [2, 3, 5, 7, 11, 13, 17, 19]
        groups = {}
        for a in range(2, 512):
            mask, rest = 0, a
            for k, p in enumerate(small):
                if rest % p == 0:
                    mask |= 1 << k
                    while rest % p == 0:
                        rest //= p
            groups.setdefault(rest, []).append((mask, a))
        best = {0: (Fraction(1, 511), (1,))}
        for q in sorted(groups):
            stages = [groups[q]] if q != 1 else [[item] for item in groups[q]]     # one multiple of a large prime at most
            for stage in stages:
                update = dict(best)
                for mask, a in stage:
                    for used, (value, chosen) in best.items():
                        if not used & mask and (used | mask not in update or update[used | mask][0] < value+Fraction(1, 512-a)):
                            update[used | mask] = (value+Fraction(1, 512-a), chosen+(a,))
                best = update
        value, chosen = max(best.values())
        self.assertTrue(all(gcd(x, y) == 1 for x, y in combinations(chosen, 2)))
        self.assertEqual(round(float(value), 12), 2.762987911413)
        self.assertEqual(i.coprime_reciprocal_sum(list(chosen))[0], 2.762987911413)
        self.assertLess(i.coprime_reciprocal_sum(seed_of("science-coprime-reciprocal-distances-512"))[0], 2.762987911413)


def minimum_hitting_set(edges):
    """Exact branch and bound (bound: pairwise disjoint edges)."""
    best = [None]

    def search(rest, chosen):
        if not rest:
            if best[0] is None or len(chosen) < len(best[0]):
                best[0] = set(chosen)
            return
        used, bound = set(), 0
        for e in sorted(rest, key=len):
            if not e & used:
                used |= e
                bound += 1
        if best[0] is not None and len(chosen)+bound >= len(best[0]):
            return
        edge = min(rest, key=len)
        banned = set()
        for x in sorted(edge, key=lambda v: -sum(v in f for f in rest)):
            remaining = [f-banned for f in rest if x not in f]
            if all(remaining):
                search(remaining, chosen | {x})
            banned.add(x)
    search([frozenset(e) for e in edges], set())
    return best[0]


class InstanceTasks(unittest.TestCase):
    """Exact optima and attainable maxima of the fixed instances."""

    def test_clique_transversal_optimum(self):
        total, answer = 0, []
        for n, p, seed in i.TRANSVERSAL_INSTANCES:
            adj = [0]*n
            for u, v in i.random_graph(n, p, seed):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
            cliques = [{v for v in range(n) if c >> v & 1} for c in i.maximal_cliques(adj, n)]
            answer.append(sorted(minimum_hitting_set(cliques)))
            total += len(answer[-1])
        self.assertEqual([len(row) for row in answer], [12, 13, 19])
        self.assertEqual(i.clique_transversal(answer)[0], 44)

    def test_sum_free_optimum_is_attained(self):
        removed = [[-6293, -5370, -1250, -801, 3641],
                   [-8281, -6988, -5153, -2232, -1860, -1157, 169, 425, 1879, 2050, 2216, 2437, 3934, 4316, 4889, 5791, 9358],
                   [-9797, -8808, -8419, -8368, -8039, -7871, -7405, -7256, -7032, -7020, -6838, -6824, -6817, -6213, -4857,
                    -4777, -4710, -4590, -4387, -3888, -3353, -2958, -2649, -2250, -2010, -1938, -1932, -1361, -1319, -997,
                    -785, -440, -388, -336, -147, 377, 392, 483, 541, 581, 647, 782, 1014, 1224, 1326, 1510, 1606, 2456, 2528,
                    2753, 2892, 3285, 3330, 4191, 4338, 4431, 4545, 4792, 5745, 5956, 6738, 6955, 6961, 7266, 7615, 8348, 8699,
                    8988, 9447, 9656]]
        answer = [[x for x in i.sum_free_instance(*spec) if x not in set(out)] for spec, out in zip(i.SUM_FREE_INSTANCES, removed)]
        self.assertEqual(i.sum_free_subsets(answer)[0], 328)
        for (size, seed), expected in zip(i.SUM_FREE_INSTANCES[:2], (55, 103)):      # the third needs a linear-programming bound
            values = i.sum_free_instance(size, seed)
            members = set(values)
            triples = [{a, b, a+b} for k, a in enumerate(values) for b in values[k:] if a+b in members]
            self.assertEqual(size-len(minimum_hitting_set(triples)), expected)

    def test_colouring_maxima_are_attained_by_textbook_search(self):
        rng, colourings = Random(0), []
        for n, seed in i.TRIANGLE_CYCLE_INSTANCES:                 # min-conflicts
            adj = [[] for _ in range(n)]
            for u, v in i.cycle_plus_triangles(n, seed):
                adj[u].append(v)
                adj[v].append(u)
            colour = [rng.randrange(3) for _ in range(n)]
            for _ in range(200000):
                bad = [x for x in range(n) if any(colour[y] == colour[x] for y in adj[x])]
                if not bad:
                    break
                x = rng.choice(bad)
                counts = [sum(colour[y] == c for y in adj[x]) for c in range(3)]
                colour[x] = rng.choice([c for c in range(3) if counts[c] == min(counts)]) if rng.random() > 0.02 else rng.randrange(3)
            colourings.append(colour)
        self.assertEqual(i.triangle_cycle_colouring(colourings)[0], 5)
        colourings = []
        for splits, seed in i.LIST_INSTANCES:                      # smallest list first, no backtracking needed
            n, edges, lists = i.planar_bipartite_lists(splits, seed)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            colour, free = [None]*n, [set(row) for row in lists]
            for _ in range(n):
                x = min((v for v in range(n) if colour[v] is None), key=lambda v: (len(free[v]), -len(adj[v])))
                colour[x] = min(free[x]) if free[x] else lists[x][0]
                for y in adj[x]:
                    free[y].discard(colour[x])
            colourings.append(colour)
        self.assertEqual(i.list_colouring(colourings)[0], 4)


if __name__ == "__main__":
    unittest.main()
