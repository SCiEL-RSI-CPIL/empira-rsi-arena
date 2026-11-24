"""Independent oracles, published constructions and malformed inputs for group I."""
from __future__ import annotations

from fractions import Fraction
import itertools
from math import gcd, isqrt
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"suite/_lib"))
sys.path.insert(0, str(ROOT/"tests"))
from autoresearch_suite import runtime
from autoresearch_suite import science_prize_i as v
from suite_test_support import specifications

# The specification list of the generator script is not part of this library: the same fields are read
# from the packages (seed, parameters, direction, verifier) and the catalog (label, reference value).
SPECS = specifications("prize_i")
FUNCTIONS = {spec["key"]: spec["function"] for spec in SPECS}

# Seymour's 23-edge 4-uniform hypergraph without Property B (OEIS A392185, sets for a(4) = 23).
M4 = [[0, 1, 2, 3], [0, 1, 7, 8], [0, 2, 7, 9], [0, 3, 8, 9], [0, 4, 5, 10], [0, 4, 6, 10], [0, 5, 6, 10], [0, 7, 8, 9],
      [1, 2, 7, 10], [1, 3, 8, 10], [1, 4, 5, 9], [1, 4, 6, 9], [1, 5, 6, 9], [1, 7, 8, 10], [2, 3, 9, 10], [2, 4, 5, 8],
      [2, 4, 6, 8], [2, 5, 6, 8], [2, 7, 9, 10], [3, 4, 5, 7], [3, 4, 6, 7], [3, 5, 6, 7], [3, 8, 9, 10]]
FANO = [[0, 1, 3], [1, 2, 4], [2, 3, 5], [3, 4, 6], [4, 5, 0], [5, 6, 1], [6, 0, 2]]


def run_seed(spec):
    namespace = {}
    exec(spec["seed"], namespace)
    return namespace["construct"]()


def abbott_hanson_51():
    """Abbott-Hanson: Fano core plus A = 7..11, B = 12..16 (51 edges on 17 vertices)."""
    a, b = list(range(7, 12)), list(range(12, 17))
    edges = [e+[a[j], b[j]] for e in FANO for j in range(5)]
    for k in range(5):                                   # |K| = 1: A_K with the complementary B's
        edges.append([a[k]]+[b[i] for i in range(5) if i != k])
    for pair in itertools.combinations(range(5), 2):     # |K| = 2: the complementary A's with B_K
        edges.append([a[i] for i in range(5) if i not in pair]+[b[i] for i in pair])
    edges.append(a)
    return {"n": 17, "edges": edges}


def two_colourable(n, edges):
    return any(all(len({(mask >> x) & 1 for x in e}) == 2 for e in edges) for mask in range(1 << n))


def graph_value(n, edges):
    return {"n": n, "edges": [list(e) for e in edges]}


def adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, w in edges:
        adj[u].add(w)
        adj[w].add(u)
    return adj


def cycle_set(n, edges):
    """All cycle lengths by explicit enumeration of vertex sequences."""
    adj = adjacency(n, edges)
    found = set()
    for k in range(3, n+1):
        for combo in itertools.combinations(range(n), k):
            first = combo[0]
            for rest in itertools.permutations(combo[1:]):
                seq = (first,)+rest
                if all(seq[(i+1) % k] in adj[seq[i]] for i in range(k)):
                    found.add(k)
                    break
            if k in found:
                break
    return found


def backtrack_colouring(n, edges, lists):
    """Exhaustive search choosing the most constrained uncoloured vertex first."""
    adj = adjacency(n, edges)
    colour = {}

    def options(x):
        return [c for c in lists[x] if all(colour.get(y) != c for y in adj[x])]

    def go():
        if len(colour) == n:
            return True
        x = min((y for y in range(n) if y not in colour), key=lambda y: (len(options(y)), -len(adj[y])))
        for c in options(x):
            colour[x] = c
            if go():
                return True
            del colour[x]
        return False
    return [colour[x] for x in range(n)] if go() else None


class Hypergraphs(unittest.TestCase):
    def test_two_colouring_search_against_brute_force(self):
        rng = random.Random(1)
        for _ in range(300):
            n, k = rng.randint(4, 10), rng.choice((2, 3, 4))
            edges = [rng.sample(range(n), k) for _ in range(rng.randint(1, 25))]
            self.assertEqual(v.proper_two_colouring(n, edges) is not None, two_colourable(n, edges))

    def test_published_constructions(self):
        self.assertEqual(v.property_b({"n": 3, "edges": [[0, 1], [1, 2], [0, 2]]}, uniformity=2)[0], 3)
        self.assertEqual(v.property_b({"n": 7, "edges": FANO}, uniformity=3)[0], 7)
        self.assertEqual(v.property_b({"n": 11, "edges": M4}, uniformity=4)[0], 23)
        construction = abbott_hanson_51()
        self.assertFalse(two_colourable(17, construction["edges"]))
        self.assertEqual(v.property_b(construction)[0], 51)
        for i in range(7):                                  # every Fano edge is needed
            with self.assertRaises(runtime.CandidateError):
                v.property_b({"n": 7, "edges": FANO[:i]+FANO[i+1:]}, uniformity=3)

    def test_forbidden_intersections(self):
        published = {(3, 0): 4, (3, 1): 4, (3, 2): 7, (4, 0): 8, (4, 1): 6, (4, 2): 11, (4, 3): 15}   # A390645
        for (n, r), expected in published.items():
            subsets = [frozenset(c) for k in range(n+1) for c in itertools.combinations(range(1, n+1), k)]
            compatible = [sum(1 << j for j, b in enumerate(subsets) if j != i and len(a & b) != r) for i, a in enumerate(subsets)]
            best = max(c.bit_count() for c in v.maximal_cliques(compatible, len(subsets)))
            self.assertEqual(best, expected, (n, r))
        extremal = [list(c) for k in range(9) for c in itertools.combinations(range(1, 9), k)
                    if k <= 1 or len(set(c)-{1}) >= 5]
        self.assertEqual(v.forbidden_intersection(extremal)[0], 67)
        with self.assertRaises(runtime.CandidateError):
            v.forbidden_intersection([[1, 2, 3], [2, 3, 4]])


class Graphs(unittest.TestCase):
    def test_cycle_clique_ramsey(self):
        eight_k4 = [(4*b+i, 4*b+j) for b in range(8) for i in range(4) for j in range(i+1, 4)]
        self.assertEqual(v.cycle_clique_ramsey(graph_value(32, eight_k4))[0], 32)
        with self.assertRaises(runtime.CandidateError):
            v.cycle_clique_ramsey(graph_value(33, eight_k4))                 # a ninth isolated vertex
        with self.assertRaises(runtime.CandidateError):
            v.cycle_clique_ramsey(graph_value(32, eight_k4+[(0, 4), (1, 5)]))  # creates the 5-cycle 0-4-5-1-2
        rng = random.Random(2)
        for _ in range(60):
            n = rng.randint(5, 8)
            edges = [e for e in itertools.combinations(range(n), 2) if rng.random() < 0.35]
            has_c5 = 5 in cycle_set(n, edges)
            adj = [0]*n
            for a, b in edges:
                adj[a] |= 1 << b
                adj[b] |= 1 << a
            self.assertEqual(v.has_cycle(adj, 5), has_c5)

    def test_cycle_spectra(self):
        rng = random.Random(3)
        for _ in range(150):
            n = rng.randint(3, 7)
            edges = [e for e in itertools.combinations(range(n), 2) if rng.random() < 0.45]
            adj = [0]*n
            for a, b in edges:
                adj[a] |= 1 << b
                adj[b] |= 1 << a
            mask = v.cycle_lengths(adj, n)
            self.assertEqual({k for k in range(n+1) if mask >> k & 1}, cycle_set(n, edges))
        for n, expected in enumerate([1, 1, 1, 2, 4, 6, 11]):              # A399654
            pairs = list(itertools.combinations(range(n), 2))
            spectra = set()
            for bits in range(1 << len(pairs)):
                adj = [0]*n
                for i, (a, b) in enumerate(pairs):
                    if bits >> i & 1:
                        adj[a] |= 1 << b
                        adj[b] |= 1 << a
                spectra.add(v.cycle_lengths(adj, n))
            self.assertEqual(len(spectra), expected)
        with self.assertRaises(runtime.CandidateError):
            v.cycle_spectra([[[0, 0]]])

    def test_triangle_degree_sums(self):
        fan = [(u, w) for u in range(10) for w in range(10, 16)] + [(2*i, 2*i+1) for i in range(5)]
        self.assertEqual(v.triangle_degree_sum(graph_value(16, fan))[0], 24)
        rng = random.Random(4)
        for _ in range(20):
            edges = rng.sample(list(itertools.combinations(range(16), 2)), 65)
            adj = adjacency(16, edges)
            expected = max(len(adj[a])+len(adj[b])+len(adj[c]) for a, b, c in itertools.combinations(range(16), 3)
                           if b in adj[a] and c in adj[a] and c in adj[b])
            self.assertEqual(v.triangle_degree_sum(graph_value(16, edges))[0], expected)
        with self.assertRaises(runtime.CandidateError):
            v.triangle_degree_sum(graph_value(16, fan[:-1]))

    def test_rainbow_four_cycles(self):
        rng = random.Random(5)
        for _ in range(100):
            n = rng.randint(4, 8)
            edges = {e: rng.randrange(6) for e in itertools.combinations(range(n), 2) if rng.random() < 0.5}
            if not edges:
                continue
            adj = adjacency(n, edges)
            col = lambda a, b: edges[(min(a, b), max(a, b))]
            ok = all(len({col(a, b), col(b, c), col(c, d), col(d, a)}) == 4
                     for a, b, c, d in itertools.permutations(range(n), 4)
                     if b in adj[a] and c in adj[b] and d in adj[c] and a in adj[d])
            value = [[a, b, c] for (a, b), c in edges.items()]
            if ok:
                self.assertEqual(v.rainbow_c4(value, n=n, colours=6)[0], len(edges))
            else:
                with self.assertRaises(runtime.CandidateError):
                    v.rainbow_c4(value, n=n, colours=6)


class Arithmetic(unittest.TestCase):
    def test_coprime_reciprocals(self):
        self.assertAlmostEqual(v.coprime_reciprocal_sum([511, 510, 509], n=512)[0], 1+Fraction(1, 2)+Fraction(1, 3), places=12)
        with self.assertRaises(runtime.CandidateError):
            v.coprime_reciprocal_sum([510, 508])

    def test_powerful_pairs(self):
        powerful = [x for x in range(1, 3001) if all(e >= 2 for e in v.factor(x).values())]
        members = set(powerful)
        for total in range(2, 3001, 7):
            pairs = [[a, total-a] for a in powerful if 2*a <= total and total-a in members]
            if pairs:
                self.assertEqual(v.powerful_pairs({"N": total, "pairs": pairs})[0], len(pairs))
        with self.assertRaises(runtime.CandidateError):
            v.powerful_pairs({"N": 17, "pairs": [[2, 15]]})
        with self.assertRaises(runtime.CandidateError):
            v.powerful_pairs({"N": 17, "pairs": [[1, 16], [1, 16]]})

    def test_totient_runs_and_divisor_pairs(self):
        phi = list(range(5001))
        for p in range(2, 5001):
            if phi[p] == p:
                for m in range(p, 5001, p):
                    phi[m] -= phi[m]//p
        for start in range(1, 4000, 37):
            k = 0
            while start+k <= 5000 and phi[start+k] not in phi[start:start+k]:
                k += 1
            self.assertEqual(v.distinct_totient_run({"start": start, "length": k})[0], k)
            if start+k <= 5000:
                with self.assertRaises(runtime.CandidateError):
                    v.distinct_totient_run({"start": start, "length": k+1})
        for n in range(2, 3000):
            divisors = [d for d in range(1, n+1) if n % d == 0]
            self.assertEqual(v.coprime_adjacent_divisors(n)[0], sum(gcd(a, b) == 1 for a, b in zip(divisors, divisors[1:])))

    def test_b2_2_sets(self):
        rng = random.Random(6)
        for _ in range(200):
            s = sorted(rng.sample(range(1, 60), rng.randint(2, 12)))
            counts = {}
            for a, b in itertools.combinations_with_replacement(s, 2):
                counts[a+b] = counts.get(a+b, 0)+1
            if max(counts.values()) <= 2:
                self.assertEqual(v.b2g_set(s)[0], len(s))
            else:
                with self.assertRaises(runtime.CandidateError):
                    v.b2g_set(s)


class Instances(unittest.TestCase):
    def test_generators(self):
        for n, d, seed in v.STRONG_INSTANCES:
            adj = adjacency(n, v.bounded_degree_graph(n, d, seed))
            self.assertLessEqual(max(len(a) for a in adj), d)
        for n, k, p, seed in v.CHORDAL_INSTANCES:
            adj = adjacency(n, v.split_graph(n, k, p, seed))
            self.assertTrue(all(adj[a] >= set(range(k))-{a} for a in range(k)))              # clique part
            self.assertTrue(all(not (adj[a] & set(range(k, n))) for a in range(k, n)))        # independent part
        for n, seed in v.TRIANGLE_CYCLE_INSTANCES:
            adj = adjacency(n, v.cycle_plus_triangles(n, seed))
            self.assertTrue(all(len(a) == 4 for a in adj))
            if n <= 30:
                self.assertIsNotNone(backtrack_colouring(n, v.cycle_plus_triangles(n, seed), [[0, 1, 2]]*n))
        for splits, seed in v.LIST_INSTANCES:
            n, edges, lists = v.planar_bipartite_lists(splits, seed)
            adj = adjacency(n, edges)
            self.assertEqual(len(edges), 2*n-4)                                               # quadrangulation
            self.assertGreaterEqual(min(len(a) for a in adj), 3)
            side = {0: 0}
            stack = [0]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y not in side:
                        side[y] = 1-side[x]
                        stack.append(y)
                    self.assertNotEqual(side[y], side[x])
            if n <= 90:
                self.assertIsNotNone(backtrack_colouring(n, edges, lists))
        for n, p, seed in v.DENSE_INSTANCES:
            self.assertGreater(len(v.random_graph(n, p, seed)), n*n//4)

    def test_strong_colouring_and_transversals(self):
        rng = random.Random(7)
        n, d, seed = v.STRONG_INSTANCES[0]
        edges = v.bounded_degree_graph(n, d, seed)
        adj = adjacency(n, edges)
        good = []
        for e in edges:
            near = {x for x in e} | {y for x in e for y in adj[x]}
            taken = {c for f, c in zip(edges, good) if set(f) & near}
            good.append(next(c for c in range(len(edges)) if c not in taken))
        seed_value = run_seed(next(s for s in SPECS if s["key"] == "strong_edge_colouring"))
        self.assertEqual(seed_value[0], good)
        i, j = next((i, j) for i, j in itertools.combinations(range(len(edges)), 2) if set(edges[i]) & set(edges[j]))
        bad = list(good)
        bad[j] = bad[i]
        with self.assertRaises(runtime.CandidateError):
            v.strong_edge_colouring([bad]+seed_value[1:])
        for _ in range(40):
            m = rng.randint(4, 9)
            edges = [e for e in itertools.combinations(range(m), 2) if rng.random() < 0.5]
            adj = adjacency(m, edges)
            cliques = [set(c) for k in range(2, m+1) for c in itertools.combinations(range(m), k)
                       if all(b in adj[a] for a, b in itertools.combinations(c, 2))]
            maximal = {frozenset(c) for c in cliques if not any(c < o for o in cliques)}
            bits = [0]*m
            for a, b in edges:
                bits[a] |= 1 << b
                bits[b] |= 1 << a
            self.assertEqual({frozenset(x for x in range(m) if c >> x & 1) for c in v.maximal_cliques(bits, m)}, maximal)

    def test_partitions_and_colourings(self):
        seeds = {s["key"]: run_seed(s) for s in SPECS}
        self.assertEqual(v.chordal_clique_partition(seeds["chordal_clique_partition"])[0],
                         sum(len(v.split_graph(*spec)) for spec in v.CHORDAL_INSTANCES))
        n, k, p, seed = v.CHORDAL_INSTANCES[0]
        clique = [list(range(k))]
        rest = [[a, b] for a, b in v.split_graph(n, k, p, seed) if b >= k]
        self.assertEqual(v.chordal_clique_partition([clique+rest]+seeds["chordal_clique_partition"][1:])[0],
                         1+len(rest)+sum(len(v.split_graph(*spec)) for spec in v.CHORDAL_INSTANCES[1:]))
        for broken in ([[0, 1]]+[[0, 1]], [[0, k]] if (0, k) not in set(v.split_graph(n, k, p, seed)) else [[k, k+1]]):
            with self.assertRaises(runtime.CandidateError):
                v.chordal_clique_partition([broken]+seeds["chordal_clique_partition"][1:])
        colourings = []
        for n, seed in v.TRIANGLE_CYCLE_INSTANCES[:1]:
            colourings.append(backtrack_colouring(n, v.cycle_plus_triangles(n, seed), [[0, 1, 2]]*n))
        value = colourings+seeds["cycle_plus_triangles"][1:]
        self.assertGreaterEqual(v.triangle_cycle_colouring(value)[0], 1)
        with self.assertRaises(runtime.CandidateError):
            v.triangle_cycle_colouring([[3]*30]+value[1:])
        sets = [v.sum_free_instance(size, seed) for size, seed in v.SUM_FREE_INSTANCES]
        odd = [[x for x in s if x % 2] for s in sets]
        self.assertEqual(v.sum_free_subsets(odd)[0], sum(map(len, odd)))
        positive = [x for x in sets[0] if x > 0]
        doubled = next(([x, 2*x] for x in positive if 2*x in positive), None)
        if doubled:
            with self.assertRaises(runtime.CandidateError):
                v.sum_free_subsets([doubled]+odd[1:])


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
            if spec["contribution"] == "method":
                self.assertEqual(spec["sota"]["kind"], "none")

    def test_malformed_inputs(self):
        cases = [(v.property_b, {"n": 9, "edges": [[0, 1, 2, 3, 3]]}), (v.property_b, {"n": 30, "edges": [[0, 1, 2, 3, 4]]}),
                 (v.property_b, {"n": 5, "edges": [[0, 1, 2, 3, 4]]}),                       # 2-colourable
                 (v.forbidden_intersection, [[0]]), (v.forbidden_intersection, [[1], [1]]),
                 (v.cycle_clique_ramsey, {"n": 36, "edges": []}), (v.cycle_spectra, [[[0, 10]]]),
                 (v.coprime_reciprocal_sum, [512]), (v.powerful_pairs, {"N": 10**19, "pairs": [[1, 1]]}),
                 (v.distinct_totient_run, {"start": 1, "length": 3}), (v.coprime_adjacent_divisors, 1),
                 (v.triangle_degree_sum, {"n": 15, "edges": []}), (v.rainbow_c4, [[0, 1, 20]]),
                 (v.strong_edge_colouring, [[]]), (v.clique_transversal, [[], [], []]),
                 (v.chordal_clique_partition, [[[0, 1]]]), (v.triangle_cycle_colouring, [[0]]*5),
                 (v.dense_clique_partition, [[[0, 1]], [[0, 1]]]), (v.list_colouring, [[0]]*4),
                 (v.sum_free_subsets, [[0], [], []]), (v.b2g_set, [1, 2, 3, 4, 5, 6])]
        for function, value in cases:
            with self.subTest(function=function.__name__, value=str(value)[:40]), self.assertRaises(runtime.CandidateError):
                function(value)


if __name__ == "__main__":
    unittest.main()
