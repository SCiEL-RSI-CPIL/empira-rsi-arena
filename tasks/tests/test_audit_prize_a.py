"""Construction verifiers of the first prize-bank group: seed headroom, published objects, recomputed optima.

The reference values the packages state are recomputed here from the definitions, and the verifiers
are run on constructions from the literature and against published OEIS terms. The two integer
programmes at the end need SciPy and minutes; they run only with PRIZE_A_SLOW=1.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import gcd, isqrt
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "suite/_lib")]
from autoresearch_suite import runtime, science_prize, science_prize_a

SPEEDS = [1, 3, 4, 7, 11, 18, 29]


def accepted(function, candidate, **parameters):
    try:
        return function(candidate, **parameters)[0]
    except runtime.CandidateError:
        return None


def largest(n, feasible):
    """Size of the largest subset of [1,n] with a hereditary property, the verifier being the oracle."""
    best = 0

    def grow(chosen, start):
        nonlocal best
        best = max(best, len(chosen))
        for x in range(start, n + 1):
            if len(chosen) + n - x + 1 <= best:
                break
            if feasible(chosen + [x]):
                grow(chosen + [x], x + 1)
    grow([], 1)
    return best


def divisor_sum_optimum(n=5040, target=4979):
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    return next(k for k in range(1, len(divisors) + 1) if any(sum(c) == target for c in combinations(divisors, k)))


class SeedHeadroomTests(unittest.TestCase):
    def test_divisor_sum_seed_is_valid_and_worse_than_the_exhaustive_optimum(self):
        # The first seed (largest divisor first) already had five terms, the optimum.
        self.assertEqual(divisor_sum_optimum(), 5)
        seed = runtime.load_program(ROOT / "suite/science-practical-divisor-sum-5040/initial_program.py")
        score = science_prize_a.practical_divisor_sum(seed.construct(), n=5040, target=4979)[0]
        self.assertGreater(score, 5)
        self.assertEqual(score, 49)


class PublishedObjectsTests(unittest.TestCase):
    def test_distinct_subset_sums_records(self):
        # OEIS A276661, comment of 27 November 2025: a(12) <= 1157, below the Conway-Guy value 1164.
        record = [1157, 1152, 1149, 1147, 1145, 1141, 1130, 1108, 1069, 993, 845, 554]
        self.assertEqual(science_prize_a.distinct_subset_sums(record)[0], 1157)
        u = [0, 1]
        for n in range(1, 12):
            u.append(2 * u[n] - u[n - int(0.5 + (2 * n) ** 0.5)])
        conway_guy = [u[12] - u[12 - i] for i in range(1, 13)]
        self.assertEqual(science_prize_a.distinct_subset_sums(conway_guy)[0], 1164)
        self.assertIsNone(accepted(science_prize_a.distinct_subset_sums, record[:-1] + [record[0] - record[1]]))

    def test_reyssat_triple_and_lonely_runner_time(self):
        score, details = science_prize_a.abc_quality([2, 3**10 * 109, 23**5])
        self.assertAlmostEqual(score, 1.6299116841, places=9)
        self.assertEqual(details["radical"], 2 * 3 * 23 * 109)
        score, details = science_prize_a.lonely_runner([1, 5])
        self.assertEqual((score, details["numerator"], details["denominator"]), (0.2, 1, 5))

    def test_record_gaps_and_their_neighbours(self):
        # OEIS A020754 and A297350: the record gaps that start below 10^7.
        self.assertEqual(science_prize_a.squarefree_gap([8870023, 8870033])[0], 10)
        self.assertEqual(science_prize_a.two_square_gap([9565544, 9565594])[0], 50)
        for function, a, b in ((science_prize_a.squarefree_gap, 8870023, 8870033),
                               (science_prize_a.two_square_gap, 9565544, 9565594)):
            for endpoints in ([a - 1, b], [a, b + 1], [a + 1, b], [a, b - 1]):
                self.assertIsNone(accepted(function, endpoints))

    def test_cluster_primes_at_the_first_exception(self):
        # OEIS A038134: 97 is the smallest odd prime that is not a cluster prime.
        self.assertEqual(science_prize_a.cluster_prime(89)[0], 89)
        self.assertIsNone(accepted(science_prize_a.cluster_prime, 97))
        self.assertEqual(science_prize_a.cluster_prime(101)[0], 101)

    def test_cerny_automaton_and_golomb_pair(self):
        cerny = [[(i + 1) % 12 for i in range(12)], list(range(11)) + [0]]
        self.assertEqual(science_prize.reset_threshold(cerny, 12)[0], 121)
        golomb = {"n": 23**3, "factorizations": [[[23, 3]], [[2, 3], [3, 2], [13, 2]]]}
        self.assertEqual(science_prize.powerful_counterexample(golomb)[0], 1)

    def test_permutation_without_monotone_three_term_progressions(self):
        def split(values):  # Davis, Entringer, Graham and Simmons: odd places, then even places
            return values if len(values) <= 2 else split(values[0::2]) + split(values[1::2])
        self.assertEqual(science_prize_a.permutation_progression(split(list(range(1, 129))))[0], 2)
        self.assertEqual(science_prize_a.permutation_progression(list(range(1, 129)))[0], 128)

    def test_property_p_counts_a_repeated_larger_term(self):
        self.assertEqual(science_prize_a.property_p(list(range(171, 257)))[0], 86)
        # In [20,30] the only violation is 20 | 30 + 30: a repeated larger term counts.
        self.assertIsNone(accepted(science_prize_a.property_p, list(range(20, 31)), n=30))
        self.assertEqual(science_prize_a.property_p(list(range(21, 31)), n=30)[0], 10)

    def test_paley_and_sylvester_hadamard_matrices(self):
        def paley(q):  # q prime, q = 3 mod 4
            chi = [0] + [1 if pow(a, (q - 1) // 2, q) == 1 else -1 for a in range(1, q)]
            return [[1] * (q + 1)] + [[-1] + [1 if i == j else chi[(i - j) % q] for j in range(q)] for i in range(q)]
        matrices = {q + 1: paley(q) for q in (3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83)}
        for n in (4, 8, 12, 16, 20, 24, 32, 44, 48):
            h = matrices[n]
            matrices.setdefault(2 * n, [row + row for row in h] + [row + [-x for x in row] for row in h])
        candidate = [matrices.get(n) for n in range(4, 101, 4)]
        built = sum(matrix is not None for matrix in candidate)
        self.assertEqual(built, 18)
        self.assertEqual(science_prize_a.hadamard_orders(candidate)[0], built / 25)
        candidate[2] = [row[:] for row in candidate[2]]  # order 12 with one sign flipped
        candidate[2][5][7] *= -1
        self.assertIsNone(accepted(science_prize_a.hadamard_orders, candidate))


class RecomputedOptimaTests(unittest.TestCase):
    def test_lonely_runner_maximum_is_one_fifth(self):
        # A local maximum of min_i ||t v_i|| is a peak of one term (t = k/(2 v_i)) or a crossing
        # of a rising and a falling term (t = k/(v_i + v_j)); equal slopes cross at k/(v_j - v_i).
        times = set()
        for i, v in enumerate(SPEEDS):
            times.update(Fraction(k, 2 * v) for k in range(1, 2 * v))
            for w in SPEEDS[i + 1:]:
                times.update(Fraction(k, v + w) for k in range(1, v + w))
                times.update(Fraction(k, w - v) for k in range(1, w - v))
        best = max(min(min(t * v % 1, 1 - t * v % 1) for v in SPEEDS) for t in times)
        self.assertEqual(best, Fraction(1, 5))

    def test_triple_free_maximum_on_the_smooth_grids(self):
        # x, 2x, 3x stay inside {2^a 3^b m} for one m coprime to 6; a row-by-row programme is exact.
        n, total = 4096, 0
        for m in range(1, n + 1):
            if m % 2 == 0 or m % 3 == 0:
                continue
            widths = []
            while m * 3 ** len(widths) <= n:
                widths.append((n // (m * 3 ** len(widths))).bit_length())
            states = {0: 0}  # chosen exponents of two in the row above -> best count so far
            for width in reversed(widths):
                rows = {}
                for above, count in states.items():
                    for mask in range(1 << width):
                        if not mask & (mask >> 1) & above:
                            rows[mask] = max(rows.get(mask, 0), count + bin(mask).count("1"))
                states = rows
            total += max(states.values())
        self.assertEqual(total, 3281)
        greedy = set()
        for x in range(n, 0, -1):
            if not (2 * x in greedy and 3 * x in greedy):
                greedy.add(x)
        self.assertEqual(science_prize_a.triple_free(sorted(greedy))[0], 3276)

    def test_smooth_antichain_total_is_220(self):
        def smooth(x):
            for p in (2, 3, 5):
                while x % p == 0:
                    x //= p
            return x == 1
        terms = [x for x in range(1, 129) if smooth(x)]
        rows = []
        for target in range(1, 129):
            rows.append(next(list(c) for k in range(1, 5) for c in combinations([x for x in terms if x <= target], k)
                             if sum(c) == target and all(b % a for a, b in combinations(c, 2))))
        self.assertEqual(sum(map(len, rows)), 220)
        self.assertEqual(science_prize_a.antichain_representations(rows)[0], 220)

    def test_two_colourings_without_a_four_term_progression_stop_at_34(self):
        longest, witness, colours = 0, None, [0]

        def extend():
            nonlocal longest, witness
            if len(colours) > longest:
                longest, witness = len(colours), colours[:]
            for colour in (0, 1):
                n = len(colours)
                if all(not colours[n - d] == colours[n - 2 * d] == colours[n - 3 * d] == colour for d in range(1, n // 3 + 1)):
                    colours.append(colour)
                    extend()
                    colours.pop()
        extend()
        self.assertEqual(longest, 34)
        self.assertEqual(science_prize_a.van_der_waerden(witness)[0], 34)

    def test_every_goldbach_and_unit_fraction_target_is_solvable(self):
        sieve = science_prize_a.primes(1000510)
        pairs = [next([p, t - p] for p in range(2, t) if sieve[p] and sieve[t - p]) for t in range(1000000, 1000512, 2)]
        self.assertEqual(science_prize_a.goldbach_decompositions(pairs)[0], 1)
        rows = []
        for n in range(3, 257):
            for x in range(n // 4 + 1, n):
                rest = Fraction(4, n) - Fraction(1, x)
                found = next(([x, y, int(1 / (rest - Fraction(1, y)))] for y in range(max(x + 1, int(1 / rest) + 1), int(2 / rest) + 2)
                              if rest > Fraction(1, y) and (rest - Fraction(1, y)).numerator == 1
                              and (rest - Fraction(1, y)).denominator > y), None)
                if found:
                    rows.append(found)
                    break
        self.assertEqual(len(rows), 254)
        self.assertEqual(science_prize_a.egyptian_four(rows)[0], 1)

    def test_small_cases_agree_with_published_sequences(self):
        v = science_prize_a
        cases = (  # OEIS terms from n = 1
            ("A389784", lambda c, n: accepted(v.nonaveraging, c, n=n, maximum=n), [1, 2, 2, 3, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 7, 7, 8]),
            ("A387704", lambda c, n: accepted(v.b3_set, c, n=n, maximum=n), [1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4]),
            ("A003003", lambda c, n: accepted(v.progression_free, c, n=n, k=4), [1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11]),
            ("A157282", lambda c, n: accepted(v.triple_free, c, n=n), [1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12, 12, 13, 14, 14]))
        for name, oracle, terms in cases:
            with self.subTest(sequence=name):
                self.assertEqual([largest(n, lambda c: oracle(c, n) is not None) for n in range(1, len(terms) + 1)], terms)

    def test_colour_and_congruence_minima_agree_with_published_sequences(self):
        def colourable(n, k):  # is there a colouring of [1,n] with k colours, three on every 4-progression?
            colours = []

            def extend(used):
                i = len(colours)
                if i == n:
                    return accepted(science_prize_a.three_color_progressions, colours, n=n) is not None
                for colour in range(min(used + 1, k)):
                    if all(len({colour, colours[i - d], colours[i - 2 * d], colours[i - 3 * d]}) >= 3 for d in range(1, i // 3 + 1)):
                        colours.append(colour)
                        if extend(max(used, colour + 1)):
                            return True
                        colours.pop()
                return False
            return extend(0)
        # OEIS A402186, h(n) of Erdos Problem 160, n = 1..16.
        self.assertEqual([next(k for k in range(1, n + 1) if colourable(n, k)) for n in range(1, 17)],
                         [1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4])

        def classes(n):  # most disjoint classes with distinct moduli 2..n
            best, chosen = 0, []

            def extend(m):
                nonlocal best
                if len(chosen) > best:
                    best = len(chosen)
                    self.assertEqual(science_prize_a.disjoint_congruences([list(c) for c in chosen], maximum=n)[0], best)
                if m > n or len(chosen) + n - m + 1 <= best:
                    return
                for r in range(m if chosen else 1):
                    if all((r - s) % gcd(m, k) for s, k in chosen):
                        chosen.append((r, m))
                        extend(m + 1)
                        chosen.pop()
                extend(m + 1)
            extend(2)
            return best
        # OEIS A389975, n = 2..20.
        self.assertEqual([classes(n) for n in range(2, 21)], [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6])


@unittest.skipUnless(os.environ.get("PRIZE_A_SLOW"), "exact integer programmes: set PRIZE_A_SLOW=1 (needs SciPy)")
class SlowExactOptimaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy
            from scipy import optimize, sparse
        except ImportError as error:
            raise unittest.SkipTest(f"SciPy is not installed: {error}")
        cls.np, cls.optimize, cls.sparse = numpy, optimize, sparse

    def solve(self, objective, rows, lower, upper, columns):
        index = [(i, j) for i, row in enumerate(rows) for j in row]
        matrix = self.sparse.csr_matrix((self.np.ones(len(index)), tuple(zip(*index))), shape=(len(rows), columns))
        return self.optimize.milp(objective, constraints=[self.optimize.LinearConstraint(matrix, lower, upper)],
                                  integrality=self.np.ones(columns), bounds=self.optimize.Bounds(0, 1))

    def test_prime_complement_of_256_to_512_needs_ten_elements(self):
        # Without the targets 256, 257, 258 the prime 2 is never used, so even targets need odd b
        # and odd targets even b: two independent covers, five elements each. The same split
        # with every target kept gives a cover with ten elements.
        def prime(x):
            return x >= 2 and all(x % d for d in range(2, isqrt(x) + 1))
        cover, bound = [], 0
        for parity in (0, 1):
            candidates = [b for b in range(257) if b % 2 != parity]
            for first in (259, 256):
                targets = [t for t in range(first, 513) if t % 2 == parity]
                rows = [[j for j, b in enumerate(candidates) if prime(t - b)] for t in targets]
                result = self.solve(self.np.ones(len(candidates)), rows, 1, self.np.inf, len(candidates))
                self.assertEqual((result.status, round(result.fun)), (0, 5))
            # from 259 on, no element of the same parity serves a target of this parity
            self.assertFalse(any(prime(t - b) for t in range(259, 513) if t % 2 == parity
                                 for b in range(257) if b % 2 == parity))
            bound += 5
            cover += [b for b, x in zip(candidates, result.x) if x > 0.5]
        self.assertEqual(bound, 10)
        self.assertEqual(science_prize_a.prime_complement(cover)[0], 10)

    def test_property_p_in_256_has_at_most_86_elements(self):
        # Split by the smallest element m: larger elements avoid residues r with 2r = 0 mod m and
        # use one class of each pair {r, m - r}. Where that count allows 87, an integer
        # programme over the larger elements shows that 87 is infeasible.
        n, size = 256, 87
        for m in range(1, n + 1):
            live = [b for b in range(m + 1, n + 1) if (2 * b) % m]
            count = [0] * m
            for b in live:
                count[b % m] += 1
            if 1 + sum(max(count[r], count[m - r]) for r in range(1, (m + 1) // 2)) < size:
                continue
            column = {b: j for j, b in enumerate(live)}
            rows, upper = [], []
            for a in [m] + live:
                for b in live:
                    # c >= b with a | b + c; c = b is the repeated larger term
                    for c in range(b + (-2 * b) % a, n + 1, a) if b > a else ():
                        if c in column:
                            rows.append(sorted({column[x] for x in (a, b, c) if x != m}))
                            upper.append(len(rows[-1]) - 1)
            rows.append(list(range(len(live))))
            lower = [-self.np.inf] * len(upper) + [size - 1]
            result = self.solve(self.np.zeros(len(live)), rows, lower, upper + [self.np.inf], len(live))
            self.assertEqual(result.status, 2, f"a set of {size} elements with smallest element {m}")


if __name__ == "__main__":
    unittest.main()
