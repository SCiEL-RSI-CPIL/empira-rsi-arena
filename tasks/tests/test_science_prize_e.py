"""Independent oracles, published-construction checks and malformed inputs for group E."""
from __future__ import annotations

import itertools
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"suite/_lib"))
sys.path.insert(0, str(ROOT/"tests"))
from autoresearch_suite import runtime
from autoresearch_suite import science_prize_e as v
from suite_test_support import specifications

# The specification list of the generator script is not part of this library: the same fields are read
# from the packages (seed, parameters, direction, verifier) and the catalog (label, reference value).
SPECS = specifications("prize_e")

# Published extremal graphs used only as verifier fixtures (never as seeds), from
# B. D. McKay's Ramsey graph data, https://users.cecs.anu.edu.au/~bdm/data/ramsey.html
R45_24 = "W?CX@DDWc[PJUtlYnHDtAdeIhSYZDYn`iQeY[NWNKON|ef?"            # a (4,5)-graph on 24 vertices
R46_35 = ("b@QWoq@_`KHKQcHQFVeGwiSrCbNV_?ZTGMTWDT{ol[LCS]uPj@kGVFqgiYXaaXWXM[oIQQPDQycEoesbPpTMikPC]"
          "wDWWW~@SRKw_")                                                # Exoo's (4,6)-graphs on 35 vertices
REGSUB6_20 = "S??GW_RHPSpdO[cpgjPTzXZuEzy}nx~\\{"                       # no regular induced subgraph on >= 6 vertices


def graph6(text):
    data = [ord(c)-63 for c in text]
    n, bits = data[0], [x >> k & 1 for x in data[1:] for k in range(5, -1, -1)]
    pairs = [(u, j) for j in range(1, n) for u in range(j)]
    return {"n": n, "edges": [list(p) for p, b in zip(pairs, bits) if b]}


def adjacency(n, edges):
    adj = [0]*n
    for u, w in edges:
        adj[u] |= 1 << w
        adj[w] |= 1 << u
    return adj


def brute_clique(n, adj, independent=False):
    best = 0
    for size in range(n, 0, -1):
        for subset in itertools.combinations(range(n), size):
            if all((adj[a] >> b & 1) != independent for a, b in itertools.combinations(subset, 2)):
                return size
    return best


def pairings(items):
    if not items:
        yield []
        return
    for i in range(1, len(items)):
        for rest in pairings(items[1:i]+items[i+1:]):
            yield [(items[0], items[i])]+rest


def random_graph(rng, n, p):
    return [(u, w) for u in range(n) for w in range(u+1, n) if rng.random() < p]


def run_seed(spec):
    namespace = {}
    exec(spec["seed"], namespace)
    return namespace["construct"]()


def verifier(spec):
    return {"ramsey_3_10": v.ramsey_graph, "ramsey_4_6": v.ramsey_graph, "ramsey_4_5": v.ramsey_graph,
            "c4_versus_k11": v.c4_clique_ramsey, "c4_versus_star39": v.c4_star_ramsey, "r334": v.triangle_triangle_clique,
            "r3333": v.multicolor_triangle, "r4c6": v.multicolor_cycle, "r5c4": v.multicolor_cycle, "octahedron": v.octahedron_ramsey,
            "cube": v.cube_ramsey, "triple_ramsey": v.triple_ramsey, "schutte": v.schutte_tournament,
            "triangle_free_five_chromatic": v.triangle_free_five_chromatic,
            "monochromatic_triangle_packing": v.monochromatic_triangle_packing, "regular_induced": v.no_large_regular_induced,
            "four_clique_colours": v.four_clique_colour_count, "monochromatic_progressions": v.monochromatic_progressions}[spec["key"]]


class SearchOracles(unittest.TestCase):
    def test_clique_and_independence_against_enumeration(self):
        rng = random.Random(11)
        for _ in range(150):
            n = rng.randint(1, 11)
            adj = adjacency(n, random_graph(rng, n, rng.random()))
            omega, alpha = brute_clique(n, adj), brute_clique(n, adj, independent=True)
            for k in range(n+2):
                self.assertEqual(v.has_clique(adj, k), omega >= k)
                self.assertEqual(v.has_independent_set(adj, k), alpha >= k)

    def test_cycle_octahedron_and_cube_detectors(self):
        cube = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
        rng = random.Random(5)
        for _ in range(60):
            n = rng.randint(6, 8)
            edges = random_graph(rng, n, rng.uniform(0.4, 0.95))
            adj = adjacency(n, edges)
            has = lambda a, b: adj[a] >> b & 1
            for length in (4, 5, 6):
                expected = any(all(has(c[i], c[(i+1) % length]) for i in range(length))
                               for c in itertools.permutations(range(n), length) if c[0] == min(c))
                self.assertEqual(v.has_cycle(adj, length), expected)
            octahedron = any(all(has(x, y) for p, q in itertools.combinations(parts, 2) for x in p for y in q)
                             for subset in itertools.combinations(range(n), 6) for parts in pairings(list(subset)))
            self.assertEqual(v.has_octahedron(adj), octahedron)
            if n == 8:
                embedded = any(all(has(perm[a], perm[b]) for a, b in cube) for perm in itertools.permutations(range(8)) if perm[0] == 0)
                self.assertEqual(v.has_cube(adj), embedded)

    def test_four_colourability_against_enumeration(self):
        rng = random.Random(3)
        for _ in range(60):
            n = rng.randint(1, 7)
            edges = random_graph(rng, n, rng.uniform(0.3, 1.0))
            expected = any(all(c[a] != c[b] for a, b in edges) for c in itertools.product(range(4), repeat=n))
            self.assertEqual(v.four_colourable(adjacency(n, edges)), expected)

    def test_triangle_packing_against_enumeration(self):
        rng = random.Random(8)
        for _ in range(40):
            n = rng.randint(3, 6)
            edges = random_graph(rng, n, rng.uniform(0.5, 1.0))
            adj = adjacency(n, edges)
            triangles = [t for t in itertools.combinations(range(n), 3) if all(adj[a] >> b & 1 for a, b in itertools.combinations(t, 2))]
            best = 0
            # packings are closed under subsets, so the first infeasible size ends the scan
            while any(len({frozenset(p) for t in choice for p in itertools.combinations(t, 2)}) == 3*(best+1)
                      for choice in itertools.combinations(triangles, best+1)):
                best += 1
            self.assertEqual(v.triangle_packing(adj, n), best)
        for n, packing in [(7, 7), (8, 8), (9, 12), (10, 13), (11, 17)]:   # triangle packing numbers of K_n
            self.assertEqual(v.triangle_packing(adjacency(n, itertools.combinations(range(n), 2)), n), packing)


class Constructions(unittest.TestCase):
    def test_small_classical_ramsey_values(self):
        pentagon = {"n": 5, "edges": [[i, (i+1) % 5] for i in range(5)]}
        self.assertEqual(v.ramsey_graph(pentagon, 3, 3, 6)[0], 5)
        # R(3,3) = 6: every graph on six vertices has a triangle or an independent triple.
        pairs = list(itertools.combinations(range(6), 2))
        for mask in range(0, 1 << 15, 97):
            edges = [list(p) for i, p in enumerate(pairs) if mask >> i & 1]
            with self.assertRaises(runtime.CandidateError):
                v.ramsey_graph({"n": 6, "edges": edges}, 3, 3, 6)

    def test_published_graphs_reach_reference_values(self):
        self.assertEqual(v.ramsey_graph(graph6(R45_24), 4, 5, 25)[0], 24)
        self.assertEqual(v.ramsey_graph(graph6(R46_35), 4, 6, 40)[0], 35)
        self.assertEqual(v.no_large_regular_induced(graph6(REGSUB6_20))[0], 20)
        with self.assertRaises(runtime.CandidateError):   # independence number 5 is too large for R(4,5)
            v.ramsey_graph(graph6(R46_35), 4, 5, 40)

    def test_triple_colourings_against_enumeration(self):
        rng = random.Random(2)
        for _ in range(40):
            n = rng.randint(4, 8)
            triples = list(itertools.combinations(range(n), 3))
            colours = [rng.randrange(2) for _ in triples]
            colour = dict(zip(triples, colours))
            red = any(all(colour[t] == 0 for t in itertools.combinations(s, 3)) for s in itertools.combinations(range(n), 4))
            blue = any(all(colour[t] == 1 for t in itertools.combinations(s, 3)) for s in itertools.combinations(range(n), 5))
            if red or blue:
                with self.assertRaises(runtime.CandidateError):
                    v.triple_ramsey({"n": n, "colors": colours})
            else:
                self.assertEqual(v.triple_ramsey({"n": n, "colors": colours})[0], n)

    def test_tournament_domination_against_enumeration(self):
        def paley(p):
            squares = {x*x % p for x in range(1, p)}
            return {"n": p, "orientation": [0 if (j-i) % p in squares else 1 for i in range(p) for j in range(i+1, p)]}
        self.assertEqual(v.schutte_tournament(paley(7), k=2)[0], 7)
        self.assertEqual(v.schutte_tournament(paley(19), k=3)[0], 19)
        with self.assertRaises(runtime.CandidateError):
            v.schutte_tournament(paley(59), k=4)
        rng = random.Random(4)
        for _ in range(30):
            n = rng.randint(4, 9)
            bits = [rng.randrange(2) for _ in range(n*(n-1)//2)]
            beats = {}
            for (i, j), b in zip(itertools.combinations(range(n), 2), bits):
                beats[(i, j)], beats[(j, i)] = b == 0, b == 1
            expected = all(any(all(beats[(z, x)] for x in s) for z in range(n) if z not in s) for s in itertools.combinations(range(n), 2))
            if expected:
                self.assertEqual(v.schutte_tournament({"n": n, "orientation": bits}, k=2)[0], n)
            else:
                with self.assertRaises(runtime.CandidateError):
                    v.schutte_tournament({"n": n, "orientation": bits}, k=2)

    def test_counting_objectives(self):
        rng = random.Random(9)
        bits = [rng.randrange(2) for _ in range(40)]
        expected = sum(1 for a in range(40) for d in range(1, 14) if a+3*d < 40 and len({bits[a+i*d] for i in range(4)}) == 1)
        self.assertEqual(v.monochromatic_progressions(bits, n=40)[0], expected)
        with self.assertRaises(runtime.CandidateError):
            v.four_clique_colour_count([0]*120)
        self.assertEqual(v.four_clique_colour_count(list(range(120)))[0], 120)
        graph = {"n": 7, "edges": [[i, (i+1) % 7] for i in range(7)]}   # C7 is 2-regular on 7 vertices
        with self.assertRaises(runtime.CandidateError):
            v.no_large_regular_induced(graph)
        path = {"n": 5, "edges": [[i, i+1] for i in range(4)]}
        self.assertEqual(v.no_large_regular_induced(path)[0], 5)


class Specifications(unittest.TestCase):
    def test_seeds_are_valid_and_do_not_claim_records(self):
        for spec in SPECS:
            with self.subTest(task=spec["id"]):
                score = verifier(spec)(run_seed(spec), **spec["parameters"])[0]
                reference = spec["sota"]["value"]
                if reference is not None:
                    if spec["direction"] == "maximize":
                        self.assertLess(score, reference)
                    else:
                        self.assertGreater(score, reference)

    def test_labels_follow_reference_values(self):
        for spec in SPECS:
            kind = spec["sota"]["kind"]
            self.assertEqual(spec["contribution"] == "rediscovery", kind == "optimum", spec["id"])
            if spec["contribution"] == "method":
                self.assertEqual(kind, "none")
            if kind == "none":
                self.assertIn(spec["contribution"], {"open_problem", "method"})

    def test_malformed_inputs(self):
        good = {"n": 3, "edges": [[0, 1]]}
        bad_graphs = [None, [], {"n": 3}, {"n": True, "edges": []}, {"n": 3.0, "edges": []}, {"n": 3, "edges": [[0, 0]]},
                      {"n": 3, "edges": [[0, 1], [1, 0]]}, {"n": 3, "edges": [[0, 3]]}, {"n": 3, "edges": [[0, 1, 2]]},
                      {"n": 3, "edges": [[0, 1]], "extra": 1}, {"n": 0, "edges": []}]
        v.ramsey_graph(good, 3, 3, 6)
        for value in bad_graphs:
            with self.subTest(value=value), self.assertRaises(runtime.CandidateError):
                v.ramsey_graph(value, 3, 3, 6)
        for value in [{"n": 4, "colors": [0]*5}, {"n": 4, "colors": [0, 1, 2, 0, 1, 4]}, {"n": 4, "colors": [0, 1, 2, 0, 1, True]},
                      {"n": 63, "colors": [0]*(63*31)}]:
            with self.subTest(value=value), self.assertRaises(runtime.CandidateError):
                v.multicolor_triangle(value)
        with self.assertRaises(runtime.CandidateError):
            v.triple_ramsey({"n": 5, "colors": [0]*9})
        with self.assertRaises(runtime.CandidateError):
            v.schutte_tournament({"n": 4, "orientation": [0]*6})
        with self.assertRaises(runtime.CandidateError):
            v.monochromatic_triangle_packing({"n": 10, "colors": [0]*45})
        with self.assertRaises(runtime.CandidateError):
            v.monochromatic_progressions([0]*255)


if __name__ == "__main__":
    unittest.main()
