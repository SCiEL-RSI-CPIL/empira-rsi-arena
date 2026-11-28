"""Ramsey and extremal construction verifiers: instance seeds and reference objects at their published values."""
from __future__ import annotations

import ast
import itertools
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "suite/_lib")]
from autoresearch_suite import runtime, science_prize_e as e, science_prize_f as f

# A starter of Z_15: seven pairs covering 1..14 whose differences are 1..7 once each. Its translates, each with
# the pair {15, i} added, are fifteen perfect matchings of K16 no two of which contain a four-cycle.
K16_STARTER = ((1, 2), (3, 5), (4, 10), (6, 14), (7, 11), (8, 13), (9, 12))


def seed_source(task):
    return (ROOT / "suite" / task / "initial_program.py").read_text()


def run_seed(task):
    namespace = {}
    exec(seed_source(task), namespace)
    return namespace["construct"]()


def nested_function(task, name):
    """A helper defined inside the seed's construct(), compiled on its own."""
    for node in ast.walk(ast.parse(seed_source(task))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace = {}
            exec("from random import Random\n" + ast.unparse(node), namespace)
            return namespace[name]
    return None


def paley(p):
    return {x * x % p for x in range(1, p)}


def pair_colours(n, colour):
    return [colour(i, j) for i, j in itertools.combinations(range(n), 2)]


class InstanceSeeds(unittest.TestCase):
    """The fixed instances are defined by generators in the verifier; each seed must hand them to the search."""

    def test_tree_packing_seed_carries_the_trees(self):
        generator = nested_function("science-tree-packing-20", "instance_trees")
        self.assertIsNotNone(generator, "the seed does not contain the tree generator")
        self.assertEqual(generator(20, 611), f.instance_trees(20, 611))
        self.assertEqual(f.tree_packing(run_seed("science-tree-packing-20"))[0], 19)

    def test_decomposition_seeds_copy_the_graph_generator(self):
        for task, instances, verifier in (("science-gallai-path-decompositions", f.GALLAI_INSTANCES, f.gallai_paths),
                                          ("science-cycle-and-edge-decompositions", f.CYCLE_INSTANCES, f.cycle_edge_decomposition)):
            with self.subTest(task=task):
                generator = nested_function(task, "instance")
                self.assertIsNotNone(generator)
                graphs = [f.instance_graph(*spec) for spec in instances]
                self.assertEqual([generator(*spec) for spec in instances], graphs)
                self.assertEqual(verifier(run_seed(task))[0], sum(map(len, graphs)))


class ReferenceObjects(unittest.TestCase):
    """Constructions built from their published descriptions reach the values the statements quote."""

    def test_witt_design_is_a_perfect_packing(self):
        # S(5,6,12): images of {infinity} + squares mod 11 under PSL(2,11) on the projective line (point 11 = infinity).
        def image(a, b, c, d, x):
            if x == 11:
                return a * pow(c, -1, 11) % 11 if c else 11
            low = (c * x + d) % 11
            return (a * x + b) * pow(low, -1, 11) % 11 if low else 11
        blocks = {tuple(sorted(image(a, b, c, d, x) for x in (11, 1, 3, 4, 5, 9)))
                  for a, b, c, d in itertools.product(range(11), repeat=4) if (a * d - b * c) % 11 in paley(11)}
        self.assertEqual(f.steiner_packing([list(b) for b in sorted(blocks)])[0], 132)
        extra = next(list(b) for b in itertools.combinations(range(12), 6) if b not in blocks)
        with self.assertRaises(runtime.CandidateError):
            f.steiner_packing([list(b) for b in sorted(blocks)] + [extra])

    def test_hoffman_singleton_graph_has_degree_seven(self):
        # Robertson's construction: pentagons P_h, pentagrams Q_i, vertex j of P_h joined to vertex h*i+j of Q_i.
        edges = [[5 * h + j, 5 * h + (j + 1) % 5] for h in range(5) for j in range(5)]
        edges += [[25 + 5 * i + j, 25 + 5 * i + (j + 2) % 5] for i in range(5) for j in range(5)]
        edges += [[5 * h + j, 25 + 5 * i + (h * i + j) % 5] for h in range(5) for i in range(5) for j in range(5)]
        self.assertEqual(f.diameter_two_triangle_free(edges)[0], 7)
        self.assertEqual(f.cycle_free(edges, 50, [3, 4])[0], 175)
        with self.assertRaises(runtime.CandidateError):   # a Moore graph loses diameter two with any edge
            f.diameter_two_triangle_free(edges[1:])

    def test_bipartite_colouring_of_k11_has_six_disjoint_monochromatic_triangles(self):
        value = {"n": 11, "colors": pair_colours(11, lambda i, j: int((i < 5) == (j < 5)))}
        self.assertEqual(e.monochromatic_triangle_packing(value)[0], 6)     # K5 packs 2 triangles, K6 packs 4

    def test_paley_colourings(self):
        def tournament(p):
            return {"n": p, "orientation": pair_colours(p, lambda i, j: 0 if (j - i) % p in paley(p) else 1)}
        self.assertEqual(e.schutte_tournament(tournament(67))[0], 67)
        with self.assertRaises(runtime.CandidateError):
            e.schutte_tournament(tournament(59))
        squares = paley(29)
        value = {"n": 29, "colors": pair_colours(29, lambda i, j: 0 if (j - i) % 29 in squares else 1)}
        self.assertEqual(e.octahedron_ramsey(value)[0], 29)

    def test_five_colouring_of_k26_without_monochromatic_four_cycles(self):
        # Points of F_5^2 with colour x2 + y2 - x1*y1, and one more point joined to column c in colour c.
        points = [(a, b) for a in range(5) for b in range(5)]

        def colour(u, v):
            return points[u][0] if v == 25 else (points[u][1] + points[v][1] - points[u][0] * points[v][0]) % 5
        self.assertEqual(e.multicolor_cycle({"n": 26, "colors": pair_colours(26, colour)}, 5, 4, 29)[0], 26)
        with self.assertRaises(runtime.CandidateError):
            e.multicolor_cycle({"n": 26, "colors": pair_colours(26, lambda u, v: min(colour(u, v), 3))}, 4, 4, 29)

    def test_cyclic_colouring_of_k29_for_r334(self):
        distance = {d: c for c, ds in enumerate(([1, 3, 7, 12], [2, 5, 6, 14], [4, 8, 9, 10, 11, 13])) for d in ds}
        value = {"n": 29, "colors": pair_colours(29, lambda i, j: distance[min(j - i, 29 - j + i)])}
        self.assertEqual(e.triangle_triangle_clique(value)[0], 29)
        swapped = {"n": 29, "colors": [2 - c for c in value["colors"]]}    # the K4-free class is not triangle-free
        with self.assertRaises(runtime.CandidateError):
            e.triangle_triangle_clique(swapped)

    def test_polarity_graph_gives_45_vertices_against_the_star(self):
        # ER(7) on the 57 points of PG(2,7), minus twelve points that leave every degree at least six.
        points = [v for v in itertools.product(range(7), repeat=3) if any(v) and next(x for x in v if x) == 1]
        keep = [i for i in range(57) if i not in (0, 1, 5, 8, 15, 21, 25, 29, 40, 44, 50, 55)]
        edges = [[a, b] for a, b in itertools.combinations(range(45), 2)
                 if sum(x * y for x, y in zip(points[keep[a]], points[keep[b]])) % 7 == 0]
        value = {"n": 45, "edges": edges}
        self.assertEqual(e.c4_star_ramsey(value)[0], 45)
        with self.assertRaises(runtime.CandidateError):   # degree six is too small against K1,38
            e.c4_star_ramsey(value, star=38)

    def test_fifteen_colours_suffice_on_k16(self):
        def factorisation(starter):
            colour = {(i, 15): i for i in range(15)}
            for i in range(15):
                for x, y in starter:
                    a, b = (x + i) % 15, (y + i) % 15
                    colour[min(a, b), max(a, b)] = i
            return pair_colours(16, lambda u, v: colour[u, v])
        # Matchings give every K4 at least four colours, and five unless two of them close a four-cycle on it.
        self.assertEqual(e.four_clique_colour_count(factorisation(K16_STARTER))[0], 15)
        # The patterned starter {x, -x} is the usual n-1 colour construction; it fails when 3 divides n-1:
        # the K4s {15, a, a+5, a+10} get three colours.
        patterned = [(x, 15 - x) for x in range(1, 8)]
        with self.assertRaises(runtime.CandidateError):
            e.four_clique_colour_count(factorisation(patterned))
        self.assertEqual(e.four_clique_colour_count(run_seed("science-erdos-gyarfas-five-colours-k16"))[0], 17)


if __name__ == "__main__":
    unittest.main()
