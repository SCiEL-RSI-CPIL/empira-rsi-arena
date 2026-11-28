"""Arithmetic and extremal construction verifiers: seeds with headroom, optima of the solved instances."""
from __future__ import annotations

import bisect
from collections import deque
from fractions import Fraction
from itertools import combinations
import json
from math import factorial, gcd, lcm
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "suite/_lib")]
from autoresearch_suite import runtime, science_prize_b, science_prize_d

MODULES = {"science_prize_b_": science_prize_b, "science_prize_d_": science_prize_d}
# Witnesses of the optima below; each was found by an exact search and is checked here by the verifier.
CONVEX = [2, 3, 5, 7, 11, 17, 23, 29, 37, 47, 59, 71, 83, 97, 113, 131, 149, 173, 197, 223, 251, 281, 311, 347, 383,
          419, 461, 503, 547, 593, 641, 691, 743, 797, 853, 911, 971, 1031, 1097, 1163, 1229, 1297, 1367,
          1439, 1511, 1583, 1657, 1733, 1811, 1889, 1973, 2063, 2153, 2243, 2339, 2441, 2543, 2647, 2753,
          2861, 2969, 3079, 3191, 3319, 3449, 3581, 3727, 3877, 4027, 4177, 4327, 4481, 4637, 4793, 4951,
          5113, 5279, 5449, 5623, 5801, 5981, 6173, 6367, 6563, 6761, 6959, 7159, 7369, 7589, 7823, 8059,
          8297, 8537, 8779, 9029, 9281, 9533, 9787]
SQUARE_FREE = [3, 7, 11, 15, 19, 23, 27, 31, 35, 36, 39, 43, 44, 47, 51, 52, 55, 59, 60, 63, 67, 71, 75, 79, 80, 83,
               87, 88, 91, 95, 96, 99, 103, 104, 107, 111, 112, 115, 119, 120, 123, 124, 127, 131, 135,
               139, 140, 143, 147, 148, 151, 155, 156, 159, 163, 164, 167, 171, 172, 175, 179, 180,
               183, 187, 188, 191, 195, 199, 203, 207, 208, 211, 215, 216, 219, 223, 224, 227, 231,
               232, 235, 239, 240, 243, 247, 248, 251, 255, 256]
LCM_WEIGHT = [17, 19, 25, 29, 31, 37, 41, 52, 53, 54, 55, 56, 59, 61, 63, 65, 67, 69, 71, 73, 77, 79, 83, 86, 88, 89,
              90, 91, 92, 94, 96, 97, 98, 99, 101, 103, 105, 107, 109, 113, 115, 117, 120, 121, 127,
              128, 129, 131, 132, 135, 137, 139, 140, 141, 143, 144, 147, 149, 151, 157, 160, 161, 163,
              167, 169, 173, 179, 181, 191, 193, 197, 199, 211, 215, 223, 227, 229, 233, 235, 239, 241,
              243, 245, 251, 253]
SIGNED = [1, -1, 1, 0, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 0, 0, 1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 1, -1, 0, -1]
PACKING = [3, 7, 13, 17, 25, 26, 27, 28, 30, 31, 33, 38, 39, 43, 44, 46, 47]
BLOCKING = [2, 11, 15, 23, 30, 31, 32, 34, 37, 42, 47, 53]


def problem(task_id):
    return json.loads((ROOT / "suite" / task_id / "problem.json").read_text(encoding="utf-8"))


def score(task_id, candidate):
    """Score of a candidate under the task's own family function and parameters."""
    record = problem(task_id)
    prefix = record["family"][:len("science_prize_b_")]
    function = getattr(MODULES[prefix], record["family"][len(prefix):])
    return function(candidate, **record["parameters"])


def seed(task_id):
    namespace = {"N": problem(task_id)["parameters"].get("n")}
    exec((ROOT / "suite" / task_id / "initial_program.py").read_text(encoding="utf-8"), namespace)
    return namespace["construct"]()


def exact(result):
    return Fraction(result[1]["exact_numerator"], result[1]["exact_denominator"])


def min_hitting_set(groups):
    """Fewest elements meeting every group (branch on a group, bound by disjoint groups); by components."""
    groups = [frozenset(g) for g in groups]
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for g in groups:
        first = next(iter(g))
        for x in g:
            parent[find(x)] = find(first)
    parts = {}
    for g in groups:
        parts.setdefault(find(next(iter(g))), []).append(g)
    total = 0
    for part in parts.values():
        best = [len(set().union(*part))]

        def search(rest, used):
            if not rest:
                best[0] = min(best[0], used)
                return
            packed, disjoint, count = set(), 0, {}
            for g in rest:
                if packed.isdisjoint(g):
                    packed |= g
                    disjoint += 1
                for x in g:
                    count[x] = count.get(x, 0) + 1
            if used + disjoint >= best[0]:
                return
            # one element of the chosen group is in the cover; the most frequent elements are tried first
            group = min(rest, key=lambda g: (len(g), -max(count[x] for x in g)))
            banned = set()
            for x in sorted(group, key=lambda x: -count[x]):
                remaining = [g for g in rest if x not in g]
                if not any(g <= banned for g in remaining):
                    search([g - banned for g in remaining], used + 1)
                banned.add(x)
        search(part, 0)
        total += best[0]
    return total


def max_clique(adjacency):
    """Largest clique of a graph given as neighbour bitmasks (greedy colouring bound)."""
    best = [0]

    def expand(size, candidates):
        order, bound, rest, colours = [], [], candidates, 0
        while rest:
            colours += 1
            layer = rest
            while layer:
                bit = layer & -layer
                x = bit.bit_length() - 1
                layer &= ~adjacency[x] & ~bit
                rest &= ~bit
                order.append(x)
                bound.append(colours)
        for i in range(len(order) - 1, -1, -1):
            if size + bound[i] <= best[0]:
                return
            x = order[i]
            following = candidates & adjacency[x]
            if following:
                expand(size + 1, following)
            else:
                best[0] = max(best[0], size + 1)
            candidates &= ~(1 << x)
    expand(0, (1 << len(adjacency)) - 1)
    return best[0]


class ReplacedSeeds(unittest.TestCase):
    """Two seeds already attained the optimum of their instance; the replacements leave headroom."""

    def test_egyptian_long_optimum_is_ten_and_the_seed_is_longer(self):
        def two_terms(rest, previous):
            # 1/a + 1/b = rest with previous < a < b forces 1/rest < a <= 2/rest
            low = max(previous + 1, rest.denominator // rest.numerator + 1)
            for a in range(low, 2 * rest.denominator // rest.numerator + 1):
                other = rest - Fraction(1, a)
                if other > 0 and other.numerator == 1 and other.denominator > a:
                    return True
            return False

        def representable(terms, rest, previous):
            if terms == 2:
                return two_terms(rest, previous)
            low = max(previous + 1, -(-rest.denominator // rest.numerator))
            high = terms * rest.denominator // rest.numerator
            return any(representable(terms - 1, rest - Fraction(1, x), x) for x in range(low, high + 1)
                       if rest > Fraction(1, x))
        # 1/5 + ... + 1/12 > 1 > 1/5 + ... + 1/11, so fewer than eight terms cannot work.
        self.assertFalse(any(representable(terms, Fraction(1), 4) for terms in (8, 9)))
        task = "science-number-theory-egyptian-long"
        self.assertEqual(score(task, [5, 6, 8, 9, 10, 12, 15, 18, 20, 24])[0], 10)
        self.assertEqual(score(task, seed(task))[0], 38)

    def test_sidon_obstruction_seed_is_above_the_optimum(self):
        task = "science-number-theory-sidon-obstruction"
        self.assertEqual(score(task, list(range(12)))[0], 5)
        self.assertEqual(score(task, seed(task))[0], 9)
        # Smaller exhaustive case of the same kind: nine of the integers 0..11 always contain a Sidon 4-subset.
        self.assertEqual(min(science_prize_b.sidon_obstruction(list(s), 9, 11)[0]
                             for s in combinations(range(12), 9)), 4)


class SolvedInstances(unittest.TestCase):
    """Optima that a short exact routine establishes; every seed must stay strictly worse."""

    WITNESSES = {
        "science-number-theory-egyptian-long": ([5, 6, 8, 9, 10, 12, 15, 18, 20, 24], 10),
        "science-number-theory-sidon-obstruction": (list(range(12)), 5),
        "science-number-theory-power-residue-run": ({"start": 1, "roots": [1, 439, 149, 2, 244]}, 1),
        "science-number-theory-powerful-parts": (465124, 216340800500),
        "science-number-theory-smooth-adjacent": (10647, 13),
        "science-number-theory-smooth-parts": (129269, 14),
        "science-number-theory-smooth-decomposition": ([5808, 117649], 11),
        "science-number-theory-inverse-free-interval": ([2, 85], 85),
        "science-number-theory-factorial-valuation":
            ([2, 3, 5, 6, 7, 11, 12, 15, 16, 19, 21, 22, 24, 25, 27, 31, 32], 32),
        "science-number-theory-factorial-large-factors": ([20, 27, 28, 30, 32, 33], 33),
        "science-number-theory-coprime-sumset": ([[1, 7, 13], [4, 40, 100]], 9),
        "science-number-theory-convex-primes": (CONVEX, 98),
        "science-number-theory-dissociated-reciprocals":
            ([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 22], 16),
        "science-number-theory-reciprocal-subset-free": ([1, 2, 7] + list(range(9, 25)), 19),
        "science-number-theory-bounded-lcm":
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 28, 30, 36, 42, 48, 60, 72, 84], 26),
        "science-number-theory-noncoprime-forced": (list(range(3, 256, 3)), 85),
        "science-number-theory-egyptian-maximum": ([10, 30, 62, 155, 186], 186),
        "science-number-theory-totient-multiple": (143, 143),
        "science-number-theory-square-sum-free": (SQUARE_FREE, 89),
        "science-anti-ramsey-four-cycle-colors-16": (None, 20),
        "science-jacobsthal-five-prime-gaps": (2310, 13),
        "science-most-frequent-permutation-order-16": (12, 2428557331200),
        "science-oppositely-ordered-farey-pair-1000": ([[499, 1000], [500, 999]], 252),
        "science-projective-blocking-intersections-pg27": (BLOCKING, 4),
        "science-small-prime-primitive-roots": ([2, 5, 2, 11, 3, 3, 2, 3, 2, 2, 7, 5], 47),
    }

    def test_witnesses_attain_the_optima_and_seeds_do_not(self):
        blocks = [x // 3 for x in range(16)]        # five rainbow triangles and one more vertex
        colouring = [20 + 3 * blocks[u] + (u + w) % 3 if blocks[u] == blocks[w] else min(blocks[u], blocks[w])
                     for u, w in combinations(range(16), 2)]
        for task, (witness, optimum) in self.WITNESSES.items():
            with self.subTest(task=task):
                value = score(task, colouring if witness is None else witness)[0]
                self.assertEqual(value, optimum)
                start = score(task, seed(task))[0]
                if problem(task)["direction"] == "maximize":
                    self.assertLess(start, optimum)
                else:
                    self.assertGreater(start, optimum)

    def test_rational_optima(self):
        value = exact(score("science-number-theory-signed-harmonic", SIGNED))
        self.assertEqual(value, Fraction(1, 174594420))
        value = exact(score("science-number-theory-unit-fraction-packing", [x - 2 for x in PACKING]))
        self.assertEqual(value, Fraction(84529, 628964612926650))
        result = score("science-number-theory-lcm-reciprocal-weight", LCM_WEIGHT)
        self.assertEqual(exact(result), sum(Fraction(1, x) for x in LCM_WEIGHT))
        self.assertEqual(result[0], 0.969774168765093)
        for task, best in (("science-number-theory-signed-harmonic", 5.727559907126471e-09),
                           ("science-number-theory-unit-fraction-packing", 1.3439388840443045e-10)):
            self.assertGreater(score(task, seed(task))[0], best)
        self.assertLess(score("science-number-theory-lcm-reciprocal-weight",
                              seed("science-number-theory-lcm-reciprocal-weight"))[0], 0.969774168765093)

    def test_scans_of_the_single_integer_instances(self):
        limit = 1000025
        largest, powerful, smooth = [1] * (limit + 1), [1] * (limit + 1), [1] * (limit + 1)
        for p in range(2, limit + 1):
            if largest[p] == 1:                                    # p is prime
                largest[p::p] = [p] * (limit // p)                 # the last prime written is the largest
                power = p * p
                while power <= limit:                              # exponent >= 2: the full power of p
                    for m in range(power, limit + 1, power):
                        if m % (power * p):
                            powerful[m] *= power
                    power *= p
                power = p
                while p < 24 and power <= limit:
                    smooth[power::power] = [x * p for x in smooth[power::power]]
                    power *= p
        window = range(10000, 1000001)
        self.assertEqual(max(powerful[n] * powerful[n + 1] for n in window), 216340800500)
        self.assertEqual(min(max(largest[n], largest[n + 1]) for n in window), 13)
        self.assertEqual(max(n for n in range(1, 1000001) if max(largest[n], largest[n + 1]) <= 11), 9800)
        self.assertEqual(min(max(largest[a], largest[123457 - a]) for a in range(1, 123457)), 11)
        counts, best = {}, 24
        for x in range(1001, 1025):
            counts[smooth[x]] = counts.get(smooth[x], 0) + 1
        for n in range(1000, 1000001):               # sliding window n+1 .. n+24
            best = min(best, len(counts))
            leaving, entering = smooth[n + 1], smooth[n + 25]
            counts[leaving] -= 1
            if not counts[leaving]:
                del counts[leaving]
            counts[entering] = counts.get(entering, 0) + 1
        self.assertEqual(best, 14)

    def test_residue_instances_modulo_1009(self):
        p = 1009
        squares = {x * x % p for x in range(1, p)}
        self.assertTrue(all(x in squares for x in range(1, 6)))
        inverse = [0] + [pow(x, p - 2, p) for x in range(1, p)]
        longest = 0
        for start in range(1, p):
            members = set()
            for x in range(start, p):
                if inverse[x] == x or inverse[x] in members:
                    break
                members.add(x)
            longest = max(longest, len(members))
        self.assertEqual(longest, 85)
        totient = next(n for n in range(1, 200) if sum(gcd(n, x) == 1 for x in range(1, n + 1)) % 120 == 0)
        self.assertEqual(totient, 143)

    def test_factorial_instances(self):
        target = factorial(12)
        for bound, expected in ((32, False), (33, True)):
            products = {1}
            for x in range(13, bound + 1):
                products |= {y * x for y in products if target % (y * x) == 0}
            self.assertEqual(target in products, expected)
        # A partial sum that is not divisible by the 2-part of the next factorial keeps its valuation.
        valuation = lambda x: (x & -x).bit_length() - 1
        alive, best = [2], 0
        for i in range(3, 33):
            following = []
            for total in alive:
                for extended in (total, total + factorial(i)):
                    if i < 32 and valuation(extended) >= valuation(factorial(i + 1)):
                        following.append(extended)
                    else:
                        best = max(best, valuation(extended))
            alive = following
        self.assertEqual(best, 32)

    def test_pair_and_triple_constraint_instances(self):
        triples = [(b * c // (b + c), b, c) for b, c in combinations(range(1, 181), 2) if b * c % (b + c) == 0]
        self.assertEqual(180 - min_hitting_set(triples), 152)
        pairs = [(a, c) for a, c in combinations(range(1, 257), 2) if a * c % (a + c) == 0]
        self.assertEqual(256 - min_hitting_set(pairs), 188)
        triples = [t for t in combinations(range(1, 129), 3)
                   if lcm(t[0], t[1]) == lcm(t[0], t[2]) == lcm(t[1], t[2])]
        self.assertEqual(128 - min_hitting_set(triples), 95)
        adjacency = [sum(1 << (y - 1) for y in range(1, 513) if y != x and lcm(x, y) <= 512) for x in range(1, 513)]
        self.assertEqual(max_clique(adjacency), 26)
        allowed = [x for x in range(1, 255) if gcd(x, 255) > 1]
        adjacency = [sum(1 << j for j, y in enumerate(allowed) if y != x and gcd(x, y) > 1) for x in allowed]
        self.assertEqual(max_clique(adjacency) + 1, 85)

    def test_egyptian_maximum_lower_bound_from_residues(self):
        # The multiples 31 j used in a representation of 5/31 satisfy sum(1/j) = 5 modulo 31.
        inverse = {j: pow(j, 29, 31) for j in range(1, 31)}
        least = next(top for top in range(1, 31) for size in range(1, top + 1)
                     for chosen in combinations(range(1, top + 1), size)
                     if top in chosen and sum(inverse[j] for j in chosen) % 31 == 5)
        self.assertEqual(31 * least, 186)

    def test_convex_prime_optimum_by_dynamic_programming(self):
        primes = [p for p in range(2, 10001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        tables, best = [], 1
        for i, p in enumerate(primes):
            lengths = {}
            for j in range(i):                    # longest chain ending ... primes[j], p with last gap p - primes[j]
                gap = p - primes[j]
                gaps, longest = tables[j]
                k = bisect.bisect_right(gaps, gap)
                lengths[gap] = (longest[k - 1] if k else 1) + 1
            gaps, longest, top = sorted(lengths), [], 0
            for gap in gaps:
                top = max(top, lengths[gap])
                longest.append(top)
            tables.append((gaps, longest))
            best = max(best, top)
        self.assertEqual(best, 98)


class BankDerivedFacts(unittest.TestCase):
    def test_gaussian_moat_threshold(self):
        def is_prime(n):
            return n > 1 and all(n % q for q in range(2, int(n ** 0.5) + 1))

        def gaussian_prime(x, y):
            return is_prime(x * x + y * y) if x and y else is_prime(abs(x + y)) and abs(x + y) % 4 == 3
        primes = {(x, y) for x in range(-64, 65) for y in range(-64, 65) if gaussian_prime(x, y)}

        def path(limit):
            steps = [(a, c) for a in range(-3, 4) for c in range(-3, 4) if 0 < a * a + c * c <= limit]
            parent, queue = {(1, 1): None}, deque([(1, 1)])
            while queue:
                z = queue.popleft()
                if max(abs(z[0]), abs(z[1])) >= 50:
                    walk = [z]
                    while parent[walk[-1]]:
                        walk.append(parent[walk[-1]])
                    return walk[::-1]
                for a, c in steps:
                    w = (z[0] + a, z[1] + c)
                    if w in primes and w not in parent:
                        parent[w] = z
                        queue.append(w)
            return None
        self.assertIsNone(path(5))
        self.assertEqual(score("science-gaussian-prime-moat-path-50", path(8))[0], 8)

    def test_inverse_cover_needs_five_terms(self):
        fewest = {}
        for size in range(1, 11):
            for terms in combinations(range(1, 11), size):
                fewest.setdefault(sum(pow(x, 99, 101) for x in terms) % 101, list(terms))
        self.assertEqual(max(map(len, fewest.values())), 5)
        self.assertEqual(score("science-distinct-modular-inverse-cover-101", [fewest[r] for r in range(101)])[0], 5)

    def test_paley_tournament_and_order_counts(self):
        residues = {x * x % 23 for x in range(1, 23)}
        bits = [0 if (w - u) % 23 in residues else 1 for u, w in combinations(range(20), 2)]
        self.assertEqual(score("science-transitive-subtournament-minimum-20", bits)[0], 5)
        counts = science_prize_d.permutation_orders(16)
        self.assertEqual(sum(counts.values()), factorial(16))
        self.assertEqual(max(counts, key=counts.get), 12)


class MalformedCandidates(unittest.TestCase):
    def test_no_family_accepts_a_seed_with_a_non_integer_entry(self):
        def spoil(candidate, bad):
            if type(candidate) is int:
                return bad(candidate)
            if type(candidate) is dict:
                key = next(iter(candidate))
                return {**candidate, key: spoil(candidate[key], bad)}
            return [spoil(candidate[0], bad)] + list(candidate[1:])
        tasks = [row["id"] for row in json.loads((ROOT / "suite/index.json").read_text(encoding="utf-8"))["tasks"]
                 if (row.get("family") or "").startswith(tuple(MODULES))]
        self.assertGreaterEqual(len(tasks), 41)
        for task in tasks:
            candidate = seed(task)
            for bad in (float, bool, str, lambda x: None, lambda x: Fraction(x)):
                with self.subTest(task=task), self.assertRaises(runtime.CandidateError):
                    score(task, spoil(candidate, bad))
            for bad in (None, "1", [], {}, float("nan")):
                with self.subTest(task=task), self.assertRaises(runtime.CandidateError):
                    score(task, bad)


if __name__ == "__main__":
    unittest.main()
