"""Independent finite oracles, boundary cases, and all 30 construction starters."""
from __future__ import annotations

import itertools
import json
from fractions import Fraction
from math import factorial, gcd, isqrt, prod
from pathlib import Path
import random
import sys
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'suite/_lib'))
sys.path.insert(0, str(ROOT / 'tests'))
from autoresearch_suite import runtime, science_prize_b as v
from suite_test_support import registered, source_file


def subsets(values):
    for mask in range(1 << len(values)):
        yield [x for i, x in enumerate(values) if (mask >> i) & 1]


def accepted(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except runtime.CandidateError:
        return None


def oracle_lcm(a, b):
    # Independent elementary common-multiple definition (small cases only).
    return next(x for x in range(max(a, b), a * b + 1) if x % a == x % b == 0)


def corrupt_first_integer(candidate, boolean=False):
    if type(candidate) is int:
        return True if boolean else float(candidate)
    if type(candidate) is list:
        return [corrupt_first_integer(candidate[0], boolean)] + candidate[1:]
    if type(candidate) is dict:
        result = dict(candidate)
        first = next(iter(result))
        result[first] = corrupt_first_integer(result[first], boolean)
        return result
    raise AssertionError('uncovered candidate form')


class PrizeBOracleTests(unittest.TestCase):
    def test_reciprocal_subsets_against_fraction_enumeration(self):
        for values in subsets(list(range(1, 9))):
            if not values:
                continue
            reciprocals = [Fraction(1, x) for x in values]
            sums = [sum(s, Fraction()) for s in subsets(reciprocals)]
            expected_dissociated = len(sums) == len(set(sums))
            expected_free = all(Fraction(1, x) not in
                                [sum(s, Fraction()) for s in subsets([Fraction(1, y) for y in values if y != x])]
                                for x in values)
            self.assertEqual(accepted(v.dissociated_reciprocals, values, 8) is not None, expected_dissociated)
            self.assertEqual(accepted(v.reciprocal_subset_free, values, 8) is not None, expected_free)

    def test_reciprocal_schur_against_ordered_fraction_triples(self):
        for values in subsets(list(range(1, 9))):
            if values:
                expected = all(Fraction(1, a) != Fraction(1, b) + Fraction(1, c)
                               for a, b, c in itertools.permutations(values, 3))
                self.assertEqual(accepted(v.reciprocal_schur_free, values, 8) is not None, expected)
        # Repeated summands are deliberately excluded by the explicit contract.
        self.assertEqual(v.reciprocal_schur_free([1, 2], 8)[0], 2)

    def test_divisibility_and_lcm_families_exhaustively(self):
        for values in subsets(list(range(1, 9))):
            if not values:
                continue
            pair_bad = any(Fraction(a * b, a + b).denominator == 1
                           for a, b in itertools.combinations(values, 2))
            self.assertEqual(accepted(v.pair_sum_nondivisibility, values, 8) is not None, not pair_bad)
            bounded = all(oracle_lcm(a, b) <= 8 for a, b in itertools.combinations(values, 2))
            self.assertEqual(accepted(v.bounded_lcm, values, 8) is not None, bounded)
            lcm_free = all(len({oracle_lcm(a, b), oracle_lcm(a, c), oracle_lcm(b, c)}) > 1
                           for a, b, c in itertools.combinations(values, 3))
            self.assertEqual(accepted(v.equal_lcm_free, values, 8) is not None, lcm_free)
            gcd_free = all(not (gcd(a, b) == gcd(a, c) == gcd(b, c))
                           for a, b, c in itertools.combinations(values, 3))
            self.assertEqual(accepted(v.equal_gcd_free, values, 8, 3) is not None, gcd_free)
            noncoprime = 6 in values and all(gcd(a, b) > 1 for a, b in itertools.combinations(values, 2))
            self.assertEqual(accepted(v.noncoprime_forced, values, 8, 6) is not None, noncoprime)
            separated = 1 not in values and all(oracle_lcm(a, b) > 8 for a, b in itertools.combinations(values, 2))
            result = accepted(v.lcm_reciprocal_weight, values, 8)
            self.assertEqual(result is not None, separated)
            if result:
                self.assertEqual(Fraction(result[1]['exact_numerator'], result[1]['exact_denominator']),
                                 sum((Fraction(1, x) for x in values), Fraction()))

    def test_square_pair_sums_include_diagonals(self):
        squares = {x * x for x in range(5)}
        for values in subsets(list(range(1, 9))):
            if values:
                expected = not any(a + b in squares for a in values for b in values)
                self.assertEqual(accepted(v.square_sum_free, values, 8) is not None, expected)
        self.assertIsNone(accepted(v.square_sum_free, [2], 8))

    def test_target_sum_against_all_subsets(self):
        for values in subsets(list(range(1, 9))):
            if values:
                expected = 12 not in {sum(s) for s in subsets(values)}
                self.assertEqual(accepted(v.target_subsum_free, values, 8, 12) is not None, expected)

    def test_block_functions_against_prefix_differences_and_ratios(self):
        for values in subsets(list(range(2, 10))):
            if not values:
                continue
            sums, products = [0], [1]
            for x in values:
                sums.append(sums[-1] + x)
                products.append(products[-1] * x)
            ds = [sums[j] - sums[i] for i, j in itertools.combinations(range(len(sums)), 2)]
            dp = [products[j] // products[i] for i, j in itertools.combinations(range(len(products)), 2)]
            self.assertEqual(accepted(v.unique_interval_sums, values, 10) is not None, len(ds) == len(set(ds)))
            self.assertEqual(accepted(v.unique_block_products, values, 10) is not None, len(dp) == len(set(dp)))
        self.assertIsNone(accepted(v.unique_interval_sums, [2, 1], 10))
        self.assertIsNone(accepted(v.unique_block_products, [3, 2], 10))

    def test_egyptian_exact_targets_and_boundaries(self):
        self.assertEqual(v.egyptian_long([2, 3, 6], threshold=1, maximum=6)[0], 3)
        self.assertIsNone(accepted(v.egyptian_long, [2, 3, 7], threshold=1, maximum=7))
        self.assertIsNone(accepted(v.egyptian_long, [2, 2], threshold=1, maximum=6))
        self.assertIsNone(accepted(v.egyptian_long, [2, 3, 6], threshold=2, maximum=6))
        self.assertEqual(v.egyptian_maximum([8, 28, 1736])[0], 1736)
        self.assertIsNone(accepted(v.egyptian_maximum, [7, 55, 11935]))
        # A finite enumeration oracle distinguishes term-count and max-denominator scores.
        for values in subsets(list(range(2, 10))):
            if not values:
                continue
            total = sum((Fraction(1, x) for x in values), Fraction())
            result = accepted(v.egyptian_maximum, values, 1, 2, 9)
            self.assertEqual(result is not None, total == Fraction(1, 2))
            if result:
                self.assertEqual(result[0], max(values))

    def test_packing_strictness_and_all_subsets(self):
        ds = [2, 3, 4, 5, 6, 7]
        for indices in subsets(list(range(len(ds)))):
            if not indices:
                continue
            total = sum((Fraction(1, ds[i]) for i in indices), Fraction())
            result = accepted(v.unit_fraction_packing, indices, ds)
            self.assertEqual(result is not None, total < 1)
            if result:
                self.assertEqual(Fraction(result[1]['exact_numerator'], result[1]['exact_denominator']), 1 - total)
        self.assertIsNone(accepted(v.unit_fraction_packing, [0, 1, 4], ds))

    def test_signed_harmonic_exhaustive_fraction_oracle(self):
        for coefficients in itertools.product((-1, 0, 1), repeat=6):
            exact = sum((Fraction(x, i) for i, x in enumerate(coefficients, 1)), Fraction())
            result = accepted(v.signed_harmonic, coefficients, 6)
            self.assertEqual(result is not None, exact != 0)
            if result:
                self.assertEqual(Fraction(result[1]['exact_numerator'], result[1]['exact_denominator']), abs(exact))

    def test_factorizations_totients_and_powerful_parts(self):
        primes = [p for p in range(2, 502) if all(p % d for d in range(2, p))]
        for n in range(2, 500):
            actual = v.factors(n)
            self.assertEqual(prod(p**e for p, e in actual.items()), n)
            self.assertTrue(all(p in primes for p in actual))
            # Divisor-based powerful-part oracle, independent of exponent extraction.
            def powerful_divisor(d):
                return all(d % (p * p) == 0 for p in primes if d % p == 0)
            parts = [max(d for d in range(1, x + 1) if x % d == 0 and powerful_divisor(d))
                     for x in (n, n + 1)]
            self.assertEqual(v.powerful_parts(n, 2, 500)[0], prod(parts))
            largest = max(p for p in primes if (n * (n + 1)) % p == 0)
            self.assertEqual(v.smooth_adjacent(n, 2, 500)[0], largest)
            phi = sum(gcd(n, x) == 1 for x in range(1, n + 1))
            result = accepted(v.totient_multiple, n, 12, 500)
            self.assertEqual(result is not None, phi % 12 == 0)
            if result:
                self.assertEqual(result[1]['totient'], phi)

    def test_factorial_valuation_fixed_minimum_and_exact_divisibility(self):
        for tail in subsets(list(range(3, 11))):
            values = [2] + tail
            total = sum(factorial(x) for x in values)
            # v2 equals the index of the least set bit, not repeated division.
            expected = (total & -total).bit_length() - 1
            self.assertEqual(v.factorial_valuation(values, n=10)[0], expected)
        self.assertEqual(v.factorial_valuation([2, 3, 5])[0], 7)
        self.assertIsNone(accepted(v.factorial_valuation, [3, 5]))

    def test_convex_primes_all_small_subsequences(self):
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        for values in subsets(primes):
            if values:
                # Discrete second differences, independent of the gap-list predicate.
                expected = all(values[i] + values[i + 2] >= 2 * values[i + 1]
                               for i in range(len(values) - 2))
                self.assertEqual(accepted(v.convex_primes, values, 20) is not None, expected)
        self.assertIsNone(accepted(v.convex_primes, [2, 4, 7], 20))

    def test_smooth_parts_against_largest_smooth_divisor(self):
        for n in range(20, 50):
            for length in (3, 5, 7):
                primes = [p for p in range(2, length) if all(p % d for d in range(2, p))]
                def smooth(d):
                    for p in primes:
                        while d % p == 0:
                            d //= p
                    return d == 1
                parts = [max(d for d in range(1, x + 1) if x % d == 0 and smooth(d))
                         for x in range(n + 1, n + length + 1)]
                score, detail = v.smooth_parts(n, 20, 50, length)
                self.assertEqual(detail['smooth_parts'], parts)
                self.assertEqual(score, len(set(parts)))
        # Prime equal to the interval length is excluded.
        self.assertEqual(v.smooth_parts(24, 24, 24, 5)[1]['smooth_parts'][0], 1)

    def test_polynomial_square_by_independent_integer_evaluation(self):
        rng = random.Random(1209)
        for _ in range(40):
            coefficients = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(6)]
            score, detail = v.sparse_polynomial_square(coefficients, terms=6, degree=5, coefficient_bound=3)
            result = detail['square_coefficients']
            # Equality on 2*degree+1 points proves equality of degree<=2*degree polynomials.
            for x in range(-5, 6):
                self.assertEqual(sum(a * x**i for i, a in enumerate(coefficients))**2,
                                 sum(a * x**i for i, a in enumerate(result)))
            self.assertEqual(score, sum(a != 0 for a in result))
        self.assertEqual(v.sparse_polynomial_square([1, 2, -2], 3, 2, 2)[0], 4)
        self.assertIsNone(accepted(v.sparse_polynomial_square, [1, 2, -2, 0], 3, 3, 2))

    def test_sidon_obstruction_against_unique_positive_differences(self):
        for values in itertools.combinations(range(9), 6):
            # Over integers, strong Sidon is equivalent to all positive pair differences being distinct.
            expected = max(len(s) for s in subsets(list(values))
                           if len([b - a for a, b in itertools.combinations(s, 2)]) ==
                           len({b - a for a, b in itertools.combinations(s, 2)}))
            result, detail = v.sidon_obstruction(values, 6, 9)
            self.assertEqual(result, expected)
            self.assertEqual(len(detail['maximum_sidon_subset']), expected)

    def test_smooth_decompositions_against_divisor_oracle(self):
        for target in range(3, 60):
            for a in range(1, target):
                b = target - a
                prime_divisors = [p for p in range(2, target) if
                                  (a % p == 0 or b % p == 0) and all(p % d for d in range(2, p))]
                self.assertEqual(v.smooth_decomposition([a, b], target)[0], max(prime_divisors, default=1))
        self.assertIsNone(accepted(v.smooth_decomposition, [0, 10], 10))
        self.assertIsNone(accepted(v.smooth_decomposition, [5, 6], 10))

    def test_factorial_factorization_exact_product_and_distinctness(self):
        # 6! = 8*9*10: distinct factors, all exceeding6.
        self.assertEqual(v.factorial_large_factors([8, 9, 10], 6)[0], 10)
        self.assertEqual(v.factorial_large_factors([720], 6)[0], 720)
        self.assertIsNone(accepted(v.factorial_large_factors, [6, 120], 6))
        self.assertIsNone(accepted(v.factorial_large_factors, [8, 9, 11], 6))
        self.assertIsNone(accepted(v.factorial_large_factors, [8, 8, 9], 6))
        # Enumerate all short factor sets at a smaller index; compare plain multiplication.
        for size in (1, 2, 3):
            for values in itertools.combinations(range(5, 25), size):
                result = accepted(v.factorial_large_factors, values, 4)
                self.assertEqual(result is not None, prod(values) == 24)
                if result:
                    self.assertEqual(result[0], max(values))

    def test_coprime_sumsets_independent_common_divisor_test(self):
        for a in itertools.combinations(range(1, 8), 2):
            for b in itertools.combinations(range(1, 8), 2):
                sums = set(x + y for x in a for y in b)
                expected = not any(x % d == y % d == 0
                                   for x, y in itertools.combinations(sums, 2)
                                   for d in range(2, min(x, y) + 1))
                result = accepted(v.coprime_sumset, [list(a), list(b)], 7, 2)
                self.assertEqual(result is not None, expected)
                if result:
                    self.assertEqual(result[0], len(sums))
        self.assertEqual(v.coprime_sumset([[1, 7, 13], [4, 40, 100]])[0], 9)

    def test_residue_run_certificates_against_residue_set(self):
        for p in (11, 17, 29):
            for exponent in (2, 3):
                residues = {pow(x, exponent, p): x for x in range(1, p)}
                for start in range(1, p - 2):
                    certificate = {'start': start, 'roots': [residues.get(start + i, 1) for i in range(3)]}
                    expected = all(start + i in residues for i in range(3))
                    self.assertEqual(accepted(v.power_residue_run, certificate, p, exponent, 3) is not None, expected)
        self.assertIsNone(accepted(v.power_residue_run, {'start': 1005, 'roots': [1] * 5}))

    def test_inverse_intervals_by_all_ordered_products(self):
        for p in (7, 11, 17, 29):
            for start in range(1, p):
                for length in range(1, p - start + 1):
                    points = range(start, start + length)
                    expected = all(x * y % p != 1 for x in points for y in points)
                    self.assertEqual(accepted(v.inverse_free_interval, [start, length], p) is not None, expected)
        self.assertIsNone(accepted(v.inverse_free_interval, [1, 1], 7))
        self.assertIsNone(accepted(v.inverse_free_interval, [6, 1], 7))
        self.assertIsNone(accepted(v.inverse_free_interval, [5, 3], 7))


class PrizeBStartersAndMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads((ROOT / 'catalog/science/prize_b.json').read_text())
        # The registrations of the generator script are not part of this library: seed, direction and scoring body
        # of each family are read from its package.
        cls.families = registered('prize_b')

    def test_all_30_seeds_and_runtime_envelopes(self):
        self.assertEqual(len(self.records), 30)
        for record in self.records:
            with self.subTest(record=record['id']):
                family = self.families[record['family']]
                ns = {}
                exec(family['seed'], ns)
                fn = getattr(v, record['family'].removeprefix('science_prize_b_'))
                score, detail = fn(ns['construct'](), **record['parameters'])
                result = runtime.success(score, family['direction'], **detail)
                self.assertTrue(result['valid'])
                self.assertEqual(result['combined_score'], score if family['direction'] == 'maximize' else -score)
                body = 'def evaluate_candidate(module, P):\n' + '\n'.join('    ' + line for line in family['body'].splitlines())
                namespace = {'runtime': runtime, 'science_prize_b': v}
                exec(body, namespace)
                actual = namespace['evaluate_candidate'](SimpleNamespace(construct=ns['construct']), record['parameters'])
                self.assertFalse(actual['lean_verified'])
                self.assertEqual(actual['score'], result['score'])

    def test_all_families_reject_wrong_container_type_and_exact_integer_impostors(self):
        for record in self.records:
            fn = getattr(v, record['family'].removeprefix('science_prize_b_'))
            ns = {}
            exec(self.families[record['family']]['seed'], ns)
            candidate = ns['construct']()
            for bad in (None, {}, '1', True, float('nan'), float('inf')):
                with self.subTest(record=record['id'], bad=bad), self.assertRaises(runtime.CandidateError):
                    fn(bad, **record['parameters'])
            for boolean in (False, True):
                bad = corrupt_first_integer(candidate, boolean)
                with self.subTest(record=record['id'], boolean=boolean), self.assertRaises(runtime.CandidateError):
                    fn(bad, **record['parameters'])

    def test_every_family_has_specific_invalid_control(self):
        controls = {
            'egyptian_long': [5], 'reciprocal_subset_free': [2, 3, 6],
            'reciprocal_schur_free': [2, 3, 6], 'egyptian_maximum': [31],
            'unit_fraction_packing': [0, 1, 4], 'signed_harmonic': [0] * 30,
            'dissociated_reciprocals': [2, 3, 6], 'pair_sum_nondivisibility': [3, 6],
            'unique_interval_sums': [1, 2, 3], 'target_subsum_free': [99, 100, 1],
            'powerful_parts': 1000001, 'smooth_adjacent': 9999,
            'factorial_valuation': [3, 5], 'unique_block_products': [2, 3, 6],
            'square_sum_free': [2], 'bounded_lcm': [511, 512],
            'convex_primes': [3, 7, 11, 13], 'totient_multiple': 1,
            'smooth_parts': 1000001, 'sparse_polynomial_square': [1] * 11,
            'sidon_obstruction': list(range(11)), 'noncoprime_forced': [2, 255],
            'equal_gcd_free': [2, 3, 5], 'equal_lcm_free': [6, 10, 15],
            'lcm_reciprocal_weight': [2, 3], 'smooth_decomposition': [123456, 2],
            'factorial_large_factors': [12, 39916800], 'coprime_sumset': [[1, 2, 3], [1, 2, 3]],
            'power_residue_run': {'start': 1, 'roots': [1] * 5}, 'inverse_free_interval': [1, 30]}
        self.assertEqual(len(controls), 30)
        for record in self.records:
            key = record['family'].removeprefix('science_prize_b_')
            with self.subTest(record=record['id']), self.assertRaises(runtime.CandidateError):
                getattr(v, key)(controls[key], **record['parameters'])

    def test_distinct_sources_evidence_and_attribution(self):
        selections = json.loads(source_file(self, 'validation/prize-expansion/b-selection.json').read_text())
        evidence = json.loads(source_file(self, 'validation/prize-expansion/b-evidence.json').read_text())
        self.assertEqual(len({x['planning_family'] for x in selections}), 30)
        self.assertEqual(set(evidence), {r['id'] for r in self.records})
        for record in self.records:
            self.assertEqual(record['provenance']['category'], 'independent_reimplementation')
            self.assertFalse(record['provenance']['upstream_evaluator_reused'])
            self.assertTrue(any(x['type'] == 'paper' for x in record['sources']))
            self.assertTrue(any('#JSP-' in x['url'] for x in record['sources']))
            self.assertNotEqual(record['labels']['scientific_contribution'], 'open_problem')


if __name__ == '__main__':
    unittest.main()
