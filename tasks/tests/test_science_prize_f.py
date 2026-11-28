"""Independent oracles, published-graph checks and malformed inputs for group F."""
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
from autoresearch_suite import science_prize_f as v
from suite_test_support import specifications

# The specification list of the generator script is not part of this library: the same fields are read
# from the packages (seed, parameters, direction, verifier) and the catalog (label, reference value).
SPECS = specifications("prize_f")
FUNCTIONS = {spec["key"]: spec["function"] for spec in SPECS}

# Extremal graphs from the Afzaly-McKay tables (https://users.cecs.anu.edu.au/~bdm/data/extremal.html),
# used only as verifier fixtures: girth >= 5 on 45 vertices (145 edges), no C5/C6 on 42 vertices (104 edges),
# and a best known C6-free graph on 40 vertices (105 edges).
GIRTH5_45 = (":l_OGKGF?``W_UCE_`pCAEIaAggmQKEqPSGYNbBACK\\DFcgHCfFGCh?kgTbaacCWTcBQxa?MJP`x_tCHeRScUPMOaP[u\\`cAxi?PHEX?oe^"
             "barY@BEDDRcG[SPGwGs{`PpR@s}caBqq@Ie`AqXaQ^OGtC[aYQiG_sgZSq@PHKiT~")
C5C6_42 = ":i_SOOKHP_?YTGhQIDZ@]mWjeBKw]OHeho{_RLEwOSKFJo_gWMVMop?gUW`a@Oko^aAP_srCCb@hdBDCAPpQIGCbaaMKIEBqkWSKFdSk[ULGDgwkYOJHw?CCBACQPN"
C6_40 = ":g_P?[KHFBP_{gRHDQPSq_Mebxg{bMGRa@MeScaak{aQbaGw_pCAp?gYCRIOaHGx??aPOmCFCFBk?ACAegXSkVKoqhWmXOp`W{_ZbApx?ubeBPp_tKEbb@hJ??_b`s{^O^"


def sparse6(text):
    data = [ord(c)-63 for c in text[1:]]
    n, bits, k = data[0], [x >> i & 1 for x in data[1:] for i in range(5, -1, -1)], 1
    while (1 << k) < n:
        k += 1
    edges, current, i = [], 0, 0
    while i+1+k <= len(bits):
        step = bits[i]
        value = int("".join(map(str, bits[i+1:i+1+k])), 2)
        i += 1+k
        current += step
        if current >= n:
            break
        if value > current:
            current = value
        else:
            edges.append([value, current])
    return n, edges


def adjacency(n, edges):
    adj = [0]*n
    for u, w in edges:
        adj[u] |= 1 << w
        adj[w] |= 1 << u
    return adj


def random_edges(rng, n, p):
    return [[u, w] for u in range(n) for w in range(u+1, n) if rng.random() < p]


def run_seed(spec):
    namespace = {}
    exec(spec["seed"], namespace)
    return namespace["construct"]()


class GraphOracles(unittest.TestCase):
    def test_bipartization_against_all_cuts(self):
        rng = random.Random(1)
        for _ in range(40):
            n = rng.randint(3, 10)
            edges = random_edges(rng, n, rng.uniform(0.2, 0.7))
            adj = adjacency(n, edges)
            if any(adj[a] >> b & 1 and adj[b] >> c & 1 and adj[a] >> c & 1 for a, b, c in itertools.combinations(range(n), 3)):
                with self.assertRaises(runtime.CandidateError):
                    v.bipartization(edges, n=n)
                continue
            best = max(sum((mask >> a & 1) != (mask >> b & 1) for a, b in edges) for mask in range(1 << n))
            self.assertEqual(v.bipartization(edges, n=n)[0], len(edges)-best)
        blowup = [[a, b] for i in range(5) for a in range(4*i, 4*i+4) for b in range(4*((i+1) % 5), 4*((i+1) % 5)+4)]
        self.assertEqual(v.bipartization(blowup, n=20)[0], 16)      # balanced C5 blow-up needs k^2 deletions
        self.assertEqual(v.sparse_halves(blowup, n=20)[0], 8)       # and has n^2/50 edges in its sparsest half

    def test_sparse_halves_and_counts_against_enumeration(self):
        rng = random.Random(2)
        for _ in range(30):
            n = 2*rng.randint(2, 5)
            edges = random_edges(rng, n, rng.uniform(0.1, 0.4))
            adj = adjacency(n, edges)
            if not v.triangle_free(adj):
                continue
            expected = min(sum(adj[a] >> b & 1 for a, b in itertools.combinations(s, 2)) for s in itertools.combinations(range(n), n//2))
            self.assertEqual(v.sparse_halves(edges, n=n)[0], expected)
        for _ in range(30):
            n = rng.randint(4, 9)
            edges = random_edges(rng, n, rng.random())
            adj = adjacency(n, edges)
            triangles = sum(all(adj[a] >> b & 1 for a, b in itertools.combinations(t, 2)) for t in itertools.combinations(range(n), 3))
            self.assertEqual(v.triangle_count(edges, n=n, edges=len(edges))[0], triangles)
            squares = sum(sum(all(adj[c[i]] >> c[(i+1) % 4] & 1 for i in range(4)) for c in [(a, b, c_, d), (a, b, d, c_), (a, c_, b, d)])
                          for a, b, c_, d in itertools.combinations(range(n), 4))
            self.assertEqual(v.four_cycle_count(edges, n=n, edges=len(edges))[0], squares)
            if edges:
                book = max(bin(adj[a] & adj[b]).count("1") for a, b in edges)
                self.assertEqual(v.largest_book(edges, n=n, edges=len(edges))[0], book)
        kab = [[a, b] for a in range(8) for b in range(8, 16)] + [[8, 9], [10, 11], [12, 13], [14, 15]]
        self.assertEqual(v.triangle_count(kab)[0], 32)

    def test_cycles_colourings_and_forbidden_subgraphs(self):
        rng = random.Random(3)
        for _ in range(40):
            n = rng.randint(4, 8)
            edges = random_edges(rng, n, rng.uniform(0.2, 0.9))
            adj = adjacency(n, edges)
            has = lambda a, b: adj[a] >> b & 1
            for length in range(3, 7):
                expected = any(all(has(c[i], c[(i+1) % length]) for i in range(length))
                               for c in itertools.permutations(range(n), length) if c[0] == min(c))
                self.assertEqual(v.has_cycle(adj, length), expected)
            for k in (2, 3):
                expected = any(all(c[a] != c[b] for a, b in edges) for c in itertools.product(range(k), repeat=n))
                self.assertEqual(v.colourable(adj, k), expected)
            k23 = any(bin(adj[a] & adj[b]).count("1") >= 3 for a, b in itertools.combinations(range(n), 2))
            if k23:
                with self.assertRaises(runtime.CandidateError):
                    v.bipartite_free(edges, n=n)
            # largest triangle-free induced subgraph by enumeration
            if not any(all(has(a, b) for a, b in itertools.combinations(q, 2)) for q in itertools.combinations(range(n), 4)):
                best = max(len(s) for r in range(n+1) for s in itertools.combinations(range(n), r)
                           if not any(all(has(a, b) for a, b in itertools.combinations(t, 2)) for t in itertools.combinations(s, 3)))
                self.assertEqual(v.erdos_rogers(edges, n=n)[0], best)

    def test_published_extremal_graphs(self):
        for text, lengths, value in [(GIRTH5_45, [3, 4], 145), (C5C6_42, [5, 6], 104), (C6_40, [6], 105)]:
            n, edges = sparse6(text)
            self.assertEqual(v.cycle_free(edges, n, lengths)[0], value)
        n, edges = sparse6(GIRTH5_45)
        adj = adjacency(n, edges)
        u, w = next((a, b) for a in range(n) for b in range(a+1, n) if not adj[a] >> b & 1 and adj[a] & adj[b])
        with self.assertRaises(runtime.CandidateError):   # joining two vertices at distance two closes a triangle
            v.cycle_free(edges + [[u, w]], n, [3, 4])

    def test_diameter_and_cube(self):
        petersen = [[i, (i+1) % 5] for i in range(5)] + [[5+i, 5+(i+2) % 5] for i in range(5)] + [[i, 5+i] for i in range(5)]
        self.assertEqual(v.diameter_two_triangle_free(petersen, n=10)[0], 3)
        with self.assertRaises(runtime.CandidateError):
            v.diameter_two_triangle_free([[i, (i+1) % 10] for i in range(10)], n=10)
        cube = [[x, x ^ (1 << i)] for x in range(8) for i in range(3) if x < x ^ (1 << i)]
        with self.assertRaises(runtime.CandidateError):
            v.cube_free(cube, n=8)
        self.assertEqual(v.cube_free(cube[:-1], n=8)[0], 11)


class HypergraphOracles(unittest.TestCase):
    def test_matching_and_packing(self):
        rng = random.Random(4)
        for _ in range(30):
            n = rng.randint(6, 9)
            triples = [list(t) for t in itertools.combinations(range(n), 3) if rng.random() < 0.3]
            three = any(len(set(a) | set(b) | set(c)) == 9 for a, b, c in itertools.combinations(triples, 3))
            if three:
                with self.assertRaises(runtime.CandidateError):
                    v.bounded_matching(triples, n=n)
            else:
                self.assertEqual(v.bounded_matching(triples, n=n)[0], len(triples))
        self.assertEqual(v.bounded_matching([list(t) for t in itertools.combinations(range(12), 3) if 0 in t or 1 in t])[0], 100)
        with self.assertRaises(runtime.CandidateError):
            v.steiner_packing([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6]])
        self.assertEqual(v.steiner_packing([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 6, 7]])[0], 2)

    def test_tripartite_and_linear_systems(self):
        rng = random.Random(5)
        for _ in range(20):
            n = rng.randint(6, 8)
            triples = {t for t in itertools.combinations(range(n), 3) if rng.random() < 0.6}

            def pairings(items):
                if not items:
                    yield []
                    return
                for i in range(1, len(items)):
                    for rest in pairings(items[1:i]+items[i+1:]):
                        yield [(items[0], items[i])]+rest
            expected = any(all(tuple(sorted((a, b, c))) in triples for a in p[0] for b in p[1] for c in p[2])
                           for s in itertools.combinations(range(n), 6) for p in pairings(list(s)))
            if expected:
                with self.assertRaises(runtime.CandidateError):
                    v.tripartite_free([list(t) for t in triples], n=n)
            else:
                self.assertEqual(v.tripartite_free([list(t) for t in triples], n=n)[0], len(triples))
        fano = [[0, 1, 2], [0, 3, 4], [0, 5, 6], [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]]
        # the Fano plane is linear and every 5 of its 7 points contain a line, but 4 points can avoid all lines
        self.assertEqual(v.linear_independence(fano, n=7)[0], 4)
        with self.assertRaises(runtime.CandidateError):
            v.linear_independence([[0, 1, 2], [0, 1, 3]], n=7)

    def test_hypercube_faces(self):
        q3 = [[x, x ^ (1 << i)] for x in range(8) for i in range(3) if x < x ^ (1 << i)]
        with self.assertRaises(runtime.CandidateError):
            v.hypercube_c4_free(q3, dimension=3)
        with self.assertRaises(runtime.CandidateError):   # one missing edge breaks only two of the six faces
            v.hypercube_c4_free([e for e in q3 if e != [0, 1]], dimension=3)
        best = max(len(s) for r in range(13) for s in itertools.combinations(q3, r)
                   if not any(all(f in s for f in face) for face in
                              [[[a, b], [b, c], [d, c], [a, d]] for i, j in itertools.combinations(range(3), 2) for x in range(8)
                               if not x >> i & 1 and not x >> j & 1
                               for a, b, c, d in [(x, x | 1 << i, x | 1 << i | 1 << j, x | 1 << j)]]))
        self.assertEqual(best, 9)   # a(3) = 9 in OEIS A245762


class Decompositions(unittest.TestCase):
    def test_decompositions_and_packings(self):
        graphs = [v.instance_graph(*spec) for spec in v.GALLAI_INSTANCES]
        self.assertEqual(v.gallai_paths([[list(e) for e in g] for g in graphs])[0], sum(map(len, graphs)))
        with self.assertRaises(runtime.CandidateError):
            v.gallai_paths([[list(e) for e in g][1:] for g in graphs])            # uncovered edge
        with self.assertRaises(runtime.CandidateError):
            v.gallai_paths([[list(e) for e in g] + [list(g[0])] for g in graphs])  # reused edge
        n, p, seed = v.CYCLE_INSTANCES[0]
        edges = set(v.instance_graph(n, p, seed))
        triangle = next(t for t in itertools.combinations(range(n), 3) if all(tuple(sorted(e)) in edges for e in itertools.combinations(t, 2)))
        rest = [list(e) for e in sorted(edges - {tuple(sorted(e)) for e in itertools.combinations(triangle, 2)})]
        others = [[list(e) for e in v.instance_graph(*spec)] for spec in v.CYCLE_INSTANCES[1:]]
        self.assertEqual(v.cycle_edge_decomposition([rest+[list(triangle)]]+others)[0], len(edges)-2+sum(map(len, others)))
        trees = v.instance_trees(20, 611)
        self.assertEqual([len(t) for t in trees], list(range(20)))
        for k, t in enumerate(trees, 1):   # every generated tree is connected and acyclic
            parent = list(range(k))

            def root(x):
                while parent[x] != x:
                    x = parent[x]
                return x
            for a, b in t:
                self.assertNotEqual(root(a), root(b))
                parent[root(a)] = root(b)
        a, b = trees[19][0]
        with self.assertRaises(runtime.CandidateError):   # T2 placed on an edge already used by T20
            v.tree_packing([None, [a, b]] + [None]*17 + [list(range(20))])
        self.assertEqual(v.tree_packing([None]*19 + [list(range(20))])[0], 19)


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
            kind = spec["sota"]["kind"]
            self.assertEqual(spec["contribution"] == "rediscovery", kind == "optimum", spec["id"])
            if spec["contribution"] == "method":
                self.assertEqual(kind, "none")

    def test_malformed_inputs(self):
        for value in [None, [[0, 0]], [[0, 1], [1, 0]], [[0, 23]], [[0, 1.0]], [[True, 1]], [[0, 1, 2]]]:
            with self.subTest(value=value), self.assertRaises(runtime.CandidateError):
                v.bipartization(value)
        with self.assertRaises(runtime.CandidateError):
            v.triangle_count([[0, 1]])                      # wrong number of edges
        with self.assertRaises(runtime.CandidateError):
            v.bounded_matching([[0, 1, 1]])
        with self.assertRaises(runtime.CandidateError):
            v.bounded_matching([[0, 1, 2], [2, 1, 0]])
        with self.assertRaises(runtime.CandidateError):
            v.hypercube_c4_free([[0, 3]])
        with self.assertRaises(runtime.CandidateError):
            v.steiner_packing([[0, 1, 2, 3, 4, 4]])
        with self.assertRaises(runtime.CandidateError):
            v.tree_packing([None]*19)
        with self.assertRaises(runtime.CandidateError):
            v.tree_packing([None]*19 + [[0]*20])


if __name__ == "__main__":
    unittest.main()
