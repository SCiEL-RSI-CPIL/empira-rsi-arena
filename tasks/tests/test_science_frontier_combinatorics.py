"""Evaluators, seeds and catalog records of the frontier combinatorics collection."""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import frontier_combinatorics_codes as codes  # noqa: E402
from autoresearch_suite import frontier_combinatorics_designs as designs  # noqa: E402
from autoresearch_suite import frontier_combinatorics_graphs as graphs  # noqa: E402
from autoresearch_suite import frontier_combinatorics_qap as qap  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
import suite_test_support as support  # noqa: E402

FANO = [[0, 1, 2], [0, 3, 4], [0, 5, 6], [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]]
TETRACODE = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def registered():
    # The generator script is not part of this library: one generated package stands for each family.
    return support.registered("frontier_combinatorics")


def seed(name):
    module = types.ModuleType("seed_" + name)
    exec(compile(registered()[name]["seed"], name, "exec"), module.__dict__)
    return module


def lcf(n, jumps, repeats):
    edges = {(i, (i + 1) % n) for i in range(n)}
    for i in range(n):
        j = (i + jumps[i % len(jumps)]) % n
        edges.add((min(i, j), max(i, j)))
    assert len(jumps) * repeats == n
    return sorted((min(a, b), max(a, b)) for a, b in edges)


def petersen():
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    return outer + inner + [(i, 5 + i) for i in range(5)]


def random_regular(n, k, rng):
    while True:
        stubs = [v for v in range(n) for _ in range(k)]
        rng.shuffle(stubs)
        edges = [tuple(sorted(stubs[i:i + 2])) for i in range(0, len(stubs), 2)]
        if all(a != b for a, b in edges) and len(set(edges)) == len(edges):
            return edges


class CoveringDesignTest(unittest.TestCase):
    def test_fano_plane_is_an_optimal_covering(self):
        value, details = designs.covering_design(FANO, 7, 3, 2)
        self.assertEqual(value, 7)
        self.assertEqual(details["t_subsets"], 21)
        self.assertEqual(details["schoenheim_bound"], 7)

    def test_rejects_invalid_designs(self):
        cases = [FANO[:-1], FANO + [FANO[0]], FANO[:-1] + [[0, 1, 7]], FANO[:-1] + [[0, 1]],
                 FANO[:-1] + [[0, 0, 1]], FANO[:-1] + [[2, 4, 5.0]], FANO[:-1] + [[2, 4, True]], [], "0123"]
        for blocks in cases:
            with self.assertRaises(CandidateError):
                designs.covering_design(blocks, 7, 3, 2)

    def test_rank_and_unrank_follow_colexicographic_order(self):
        for v, t in ((9, 3), (8, 5)):
            colex = sorted(itertools.combinations(range(v), t), key=lambda c: c[::-1])
            for rank, subset in enumerate(colex):
                self.assertEqual(designs.unrank(rank, t), list(subset))

    def test_exhaustive_check_matches_brute_force(self):
        rng = random.Random(3)
        for _ in range(30):
            v, k, t = 9, 4, 3
            blocks = [sorted(rng.sample(range(v), k)) for _ in range(rng.randrange(8, 30))]
            blocks = [list(b) for b in dict.fromkeys(tuple(b) for b in blocks)]
            covered = {s for b in blocks for s in itertools.combinations(b, t)}
            full = len(covered) == math.comb(v, t)
            if full:
                self.assertEqual(designs.covering_design(blocks, v, k, t)[0], len(blocks))
            else:
                with self.assertRaises(CandidateError):
                    designs.covering_design(blocks, v, k, t)

    def test_schoenheim_values(self):
        self.assertEqual(designs.schoenheim(24, 7, 3), 69)
        self.assertEqual(designs.schoenheim(12, 6, 5), 132)


class DifferenceTriangleSetTest(unittest.TestCase):
    def test_golomb_ruler_and_two_rows(self):
        self.assertEqual(designs.difference_triangle_set([[0, 1, 4, 6]], 1, 3)[0], 6)
        self.assertEqual(designs.difference_triangle_set([[0, 1, 3], [0, 4, 9]], 2, 2)[0], 9)

    def test_rejects_invalid_sets(self):
        for rows in ([[0, 1, 3], [0, 2, 5]], [[1, 2, 5], [0, 4, 9]], [[0, 3, 1], [0, 4, 9]], [[0, 1, 3]],
                     [[0, 1, 3], [0, 4, 9.0]], [[0, 1, 3], [0, 4]], [[0, 1, 3], [0, 4, 2_000_000]]):
            with self.assertRaises(CandidateError):
                designs.difference_triangle_set(rows, 2, 2)


class CoveringCodeTest(unittest.TestCase):
    def test_perfect_codes(self):
        self.assertEqual(codes.covering_code(TETRACODE, 3, 4, 1)[0], 9)
        hamming = [w for w in itertools.product(range(2), repeat=7)
                   if all(sum(w[i] for i in range(7) if (i + 1) >> b & 1) % 2 == 0 for b in range(3))]
        self.assertEqual(codes.covering_code(hamming, 2, 7, 1)[0], 16)

    def test_radius_two_repetition_code(self):
        self.assertEqual(codes.covering_code([[0] * 5, [1] * 5], 2, 5, 2)[0], 2)
        with self.assertRaises(CandidateError):
            codes.covering_code([[0] * 5, [1] * 5], 2, 5, 1)

    def test_rejects_invalid_codes(self):
        for code in (TETRACODE[:-1], TETRACODE + [TETRACODE[0]], TETRACODE[:-1] + [[0, 1, 2, 3]],
                     TETRACODE[:-1] + [[0, 1, 2]], TETRACODE[:-1] + [[0, 1, 2, 1.0]], []):
            with self.assertRaises(CandidateError):
                codes.covering_code(code, 3, 4, 1)


class SuperpermutationTest(unittest.TestCase):
    def test_seed_reaches_known_minima_for_small_n(self):
        construct = seed("science_frontier_combinatorics_superpermutation").construct
        for n, minimal in ((3, 9), (4, 33), (5, 153)):
            self.assertEqual(codes.superpermutation(construct(n), n)[0], minimal)
        self.assertEqual(codes.superpermutation(construct(7), 7)[0], 5913)

    def test_accepts_lists_and_rejects_invalid_words(self):
        self.assertEqual(codes.superpermutation([1, 2, 3, 1, 2, 1, 3, 2, 1], 3)[0], 9)
        for word in ("123121", "1231213210", "12312a321", "١٢٣١٢١٣٢١", "0" + "123121321", "123" * 7):
            with self.assertRaises(CandidateError):
                codes.superpermutation(word, 3)


class CageTest(unittest.TestCase):
    def test_small_cages(self):
        self.assertEqual(graphs.regular_girth_graph(10, petersen(), 3, 5, 100)[0], 10)
        heawood = lcf(14, [5, -5], 7)
        self.assertEqual(graphs.regular_girth_graph(14, heawood, 3, 6, 100)[0], 14)
        tutte_coxeter = lcf(30, [-13, -9, 7, -7, 9, 13], 5)
        self.assertEqual(graphs.regular_girth_graph(30, tutte_coxeter, 3, 8, 100)[0], 30)
        self.assertEqual(graphs.shortest_cycle_below(self.neighbours(30, tutte_coxeter, 3), 20), 8)
        with self.assertRaises(CandidateError):
            graphs.regular_girth_graph(10, petersen(), 3, 6, 100)
        with self.assertRaises(CandidateError):
            graphs.regular_girth_graph(14, heawood, 3, 7, 100)

    @staticmethod
    def neighbours(n, edges, k):
        table = [[] for _ in range(n)]
        for a, b in edges:
            table[a].append(b)
            table[b].append(a)
        return np.array(table, dtype=np.int64).reshape(n, k)

    def test_walk_certificate_matches_breadth_first_search(self):
        rng = random.Random(7)
        for trial in range(60):
            k = 3 + trial % 2
            n = rng.choice([12, 16, 20, 30, 40])
            edges = random_regular(n, k, rng)
            girth = graphs.girth_reference(n, edges, limit=64)
            table = self.neighbours(n, edges, k)
            for g in range(3, 12):
                found = graphs.shortest_cycle_below(table, g, chunk_entries=97)
                self.assertEqual(found is None, girth >= g, (n, k, g, girth, found))
                if found is not None:
                    self.assertEqual(found, girth)

    def test_rejects_invalid_graphs(self):
        edges = petersen()
        cases = [(10, edges[:-1]), (10, edges[:-1] + [(0, 0)]), (10, edges[:-1] + [edges[0]]),
                 (10, edges[:-1] + [(0, 10)]), (10, edges[:-1] + [(0, 2.0)]), (9, edges), (11, edges),
                 (10, edges[:-1] + [(0, 2)])]
        for n, case in cases:
            with self.assertRaises(CandidateError):
                graphs.regular_girth_graph(n, case, 3, 5, 100)
        with self.assertRaises(CandidateError):
            graphs.regular_girth_graph(10, edges, 3, 5, 9)

    def test_moore_bound(self):
        self.assertEqual(graphs.moore_bound(3, 5), 10)
        self.assertEqual(graphs.moore_bound(3, 6), 14)
        self.assertEqual(graphs.moore_bound(7, 5), 50)
        self.assertEqual(graphs.moore_bound(4, 10), 242)


class QuadraticAssignmentTest(unittest.TestCase):
    def test_objective_matches_brute_force(self):
        rng = np.random.default_rng(5)
        a = rng.integers(0, 9, size=(6, 6))
        b = rng.integers(0, 9, size=(6, 6))
        costs = {}
        for p in itertools.permutations(range(6)):
            direct = sum(int(a[i, j]) * int(b[p[i], p[j]]) for i in range(6) for j in range(6))
            costs[p] = direct
            self.assertEqual(qap.objective(list(p), a, b)[0], direct)
        self.assertEqual(min(costs.values()), min(qap.objective(list(p), a, b)[0] for p in costs))

    def test_rejects_invalid_permutations(self):
        a = np.ones((4, 4), dtype=np.int64)
        for p in ([0, 1, 2], [0, 1, 2, 2], [0, 1, 2, 4], [0, 1, 2, 3.0], [0, 1, 2, True], "0123"):
            with self.assertRaises(CandidateError):
                qap.objective(p, a, a)

    def test_parse_checks_the_entry_count(self):
        self.assertEqual(qap.parse("2\n1 2\n3 4\n5 6 7 8\n")[1].tolist(), [[5, 6], [7, 8]])
        with self.assertRaises(ValueError):
            qap.parse("2\n1 2 3\n")

    def test_pinned_instances_when_prepared(self):
        lock = json.loads((ROOT / "catalog/frontier_combinatorics-data-lock.json").read_text())
        for entry in lock["files"]:
            if not (qap.data_dir() / entry["path"]).is_file():
                self.skipTest("QAPLIB data not prepared (python3 scripts/prepare_frontier_combinatorics.py)")
            a, b = qap.instance(entry["path"][:-4], entry["sha256"])
            self.assertEqual(a.shape, b.shape)
            with self.assertRaises(ValueError):
                qap.instance(entry["path"][:-4], "0" * 64)


class FamilyAndCatalogTest(unittest.TestCase):
    records = json.loads((ROOT / "catalog/science/frontier_combinatorics.json").read_text())
    # The evidence file is not part of this library; its entries are the science_relevance field of the catalog records.
    evidence = support.evidence("frontier_combinatorics")

    def test_records_are_complete_and_consistent(self):
        specs = registered()
        self.assertEqual(len(specs), 6)
        ids = [r["id"] for r in self.records]
        self.assertEqual(len(ids), len(set(ids)))
        for r in self.records:
            self.assertTrue(r["id"].startswith("science-frontier-combinatorics-"))
            self.assertIn(r["family"], specs)
            sota = r["state_of_the_art"]
            self.assertEqual(sota["kind"], "best_known")
            self.assertEqual(sota["attribution"], "human")
            self.assertEqual(sota["value"], r["parameters"]["published"])
            self.assertTrue(sota["comparability"] and sota["as_of"] and sota["sources"])
            # QAPLIB instances are standard benchmark instances (a method task); the others are mathematical constructions.
            expected = "method" if r["family"] == "science_frontier_combinatorics_qap" else "open_problem"
            self.assertEqual(r["labels"]["scientific_contribution"], expected)
            self.assertEqual(r["labels"]["reference"], "published_value")
            self.assertEqual(r["labels"]["verification"], "exact")
            self.assertTrue(all(s["url"].startswith("https://") for s in r["sources"]))
            self.assertEqual(self.evidence[r["id"]]["status"], "supported")
            if r["family"] == "science_frontier_combinatorics_qap":
                self.assertEqual(r["requires"], [f"file:data/frontier_combinatorics/{r['parameters']['instance']}.dat"])
        self.assertEqual(set(self.evidence), set(ids))

    def test_seeds_are_valid_and_leave_room_below_the_record(self):
        scorers = {
            "science_frontier_combinatorics_covering_design": lambda m, P: designs.covering_design(m.construct(P["v"], P["k"], P["t"]), P["v"], P["k"], P["t"]),
            "science_frontier_combinatorics_covering_code": lambda m, P: codes.covering_code(m.construct(P["q"], P["n"], P["radius"]), P["q"], P["n"], P["radius"]),
            "science_frontier_combinatorics_superpermutation": lambda m, P: codes.superpermutation(m.construct(P["n"]), P["n"]),
            "science_frontier_combinatorics_cage": lambda m, P: graphs.regular_girth_graph(*m.construct(P["k"], P["g"]), P["k"], P["g"], P["max_vertices"]),
            "science_frontier_combinatorics_dts": lambda m, P: designs.difference_triangle_set(m.construct(P["rows"], P["k"]), P["rows"], P["k"]),
        }
        for r in self.records:
            if r["family"] not in scorers:
                continue
            value, _ = scorers[r["family"]](seed(r["family"]), r["parameters"])
            self.assertGreater(value, r["parameters"]["published"], r["id"])

    def test_qap_seed_when_prepared(self):
        construct = seed("science_frontier_combinatorics_qap").solve
        for r in self.records:
            P = r["parameters"]
            if "instance" not in P:
                continue
            if not (qap.data_dir() / f"{P['instance']}.dat").is_file():
                self.skipTest("QAPLIB data not prepared")
            a, b = qap.instance(P["instance"], P["sha256"])
            value, _ = qap.objective(construct(a.copy(), b.copy()), a, b)
            self.assertGreater(value, P["published"])


if __name__ == "__main__":
    unittest.main()
