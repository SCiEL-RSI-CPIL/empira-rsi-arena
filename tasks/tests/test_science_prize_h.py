"""Independent oracles, published values and malformed inputs for group H."""
from __future__ import annotations

from fractions import Fraction
import itertools
from math import cos, factorial, gcd, log, pi, prod, sin, isqrt
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"suite/_lib"), str(ROOT/"tests")]
from autoresearch_suite import runtime
from autoresearch_suite import science_prize_h as v
from prize_h_oracles import phi_table, sieve, sigma, tau, trial_factor
from suite_test_support import specifications

SPECS = specifications("prize_h")  # read from the packages and the catalog: the generator script is not in this library
FUNCTIONS = {spec["key"]: spec["function"] for spec in SPECS}


def run_seed(spec):
    namespace = {}
    exec(spec["seed"], namespace)
    return namespace["construct"]()


class Arithmetic(unittest.TestCase):
    def test_primality_and_factorization(self):
        flags = sieve(100000)
        self.assertEqual([n for n in range(100001) if v.is_prime(n)], [n for n in range(100001) if flags[n]])
        for n in (3215031751, 2152302898747, 3474749660383, 341550071728321, 3825123056546413051):
            self.assertFalse(v.is_prime(n), n)          # strong pseudoprimes to several small bases
        for n in (2**61-1, 10**18+9, 999999999999999989):
            self.assertTrue(v.is_prime(n), n)
        rng = random.Random(1)
        for _ in range(500):
            n = rng.randrange(2, 10**9)
            self.assertEqual(v.factor(n), trial_factor(n))
        for _ in range(50):
            a, b = rng.randrange(10**8, 10**9), rng.randrange(10**8, 10**9)
            f = v.factor(a*b)
            self.assertEqual(prod(p**e for p, e in f.items()), a*b)
            self.assertTrue(all(v.is_prime(p) for p in f))


class Polynomials(unittest.TestCase):
    def test_mahler_measure(self):
        lehmer = [1, 1, 0, -1, -1, -1, -1, -1, 0, 1, 1]
        self.assertAlmostEqual(v.mahler_measure(lehmer)[0], 1.1762808183, places=10)
        self.assertAlmostEqual(v.mahler_measure([-2, 0, 1])[0], 2.0, places=9)              # x^2 - 2
        self.assertAlmostEqual(v.mahler_measure([2, 0, 2])[0], 2.0, places=9)               # 2(x^2 + 1)
        self.assertAlmostEqual(v.mahler_measure([1, 2])[0], 2.0, places=9)                  # 2x + 1
        for cyclotomic in ([1, 1, 1, 1, 1], [1, 1, 1, 1], [0, 0, -1, 1], [1, 0, 0, 0, 0, 0, 1], [1, 2, 1]):
            with self.assertRaises(runtime.CandidateError):
                v.mahler_measure(cyclotomic)
        rng = random.Random(2)
        for _ in range(12):
            coeffs = [rng.randint(-3, 3) for _ in range(rng.randint(3, 12))] + [rng.choice((1, 2, -1, 3))]
            # Jensen's formula: log M(P) is the mean of log|P| on the unit circle
            samples = 20000
            mean = sum(log(abs(sum(c*complex(cos(k*t), sin(k*t)) for k, c in enumerate(coeffs))) or 1e-300)
                       for t in (2*pi*(i+0.5)/samples for i in range(samples)))/samples
            if any(all(x == 0 for x in v._poly_divmod(coeffs, v.cyclotomic(m))[1]) for m in range(1, 40)
                   if len(v.cyclotomic(m)) <= len(coeffs)):
                continue                                  # roots on the circle slow the quadrature
            measure = v.mahler_measure(coeffs)[0]
            self.assertAlmostEqual(log(measure), mean, delta=2e-3)


class Primes(unittest.TestCase):
    def test_consecutive_prime_progressions(self):
        flags = sieve(20000)
        primes = [p for p in range(20001) if flags[p]]
        runs = []
        for i in range(len(primes)-1):
            j = i+1
            while j+1 < len(primes) and primes[j+1]-primes[j] == primes[i+1]-primes[i]:
                j += 1
            runs.append((j-i+1, -primes[i], primes[i+1]-primes[i]))
        length, start, step = max(runs)
        self.assertEqual((length, -start, step), (4, 251, 6))             # A006560(4) = 251
        self.assertEqual(v.consecutive_prime_progression([251, 6, 4])[0], 4)
        self.assertEqual(v.consecutive_prime_progression([121174811, 30, 6])[0], 6)
        for bad in ([7, 30, 3], [3, 2, 4], [251, 6, 5], [121174811, 30, 7]):
            with self.assertRaises(runtime.CandidateError):
                v.consecutive_prime_progression(bad)

    def test_jacobsthal_small_cases(self):
        published = [2, 4, 6, 10, 14]                     # OEIS A048670
        for count in range(1, 6):
            primes = [2, 3, 5, 7, 11][:count]
            best = 0
            for residues in itertools.product(*[range(p) for p in primes]):
                length = 0
                while any((length+1-r) % p == 0 for r, p in zip(residues, primes)):
                    length += 1
                best = max(best, length)
                self.assertEqual(v.jacobsthal_cover(list(residues), count=count)[0], length)
            self.assertEqual(best, published[count-1]-1)

    def test_prime_gap_strings(self):
        flags = sieve(10**5)
        primes = [p for p in range(10**5+1) if flags[p]]
        gaps = [b-a for a, b in zip(primes, primes[1:])]
        for i in range(0, 2000, 37):
            j, seen = i, set()
            while gaps[j] not in seen:
                seen.add(gaps[j])
                j += 1
            self.assertEqual(v.distinct_prime_gaps({"start": primes[i], "length": j-i})[0], j-i)
            with self.assertRaises(runtime.CandidateError):
                v.distinct_prime_gaps({"start": primes[i], "length": j-i+1})
        self.assertEqual(v.distinct_prime_gaps({"start": 69931991, "length": 21})[0], 21)

    def test_prime_plus_power_of_two(self):
        flags = sieve(1000)
        published = [0, 0, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 3, 0, 1, 2, 3, 1, 4, 0, 2, 1, 2, 0, 3, 0, 1, 1]   # A109925
        for n in range(3, 31):
            self.assertEqual(v.prime_plus_power_of_two(n)[0], published[n-1])
        for n in range(3, 1000):
            self.assertEqual(v.prime_plus_power_of_two(n)[0], sum(flags[n-2**k] for k in range(n.bit_length()) if n-2**k >= 0))
        self.assertEqual(v.prime_plus_power_of_two(53999715)[0], 19)

    def test_sums_of_prime_cubes(self):
        flags = sieve(1000)
        primes = [p for p in range(1000) if flags[p]]
        for target, expected in ((185527, 2), (8627527, 3), (999979163, 4)):
            triples = [[p, q, r] for p in primes for q in primes for r in primes
                       if p <= q <= r and r**3 <= target and p**3+q**3+r**3 == target]
            self.assertEqual(len(triples), expected)
            self.assertEqual(v.prime_cube_sums({"N": target, "triples": triples})[0], expected)
        with self.assertRaises(runtime.CandidateError):
            v.prime_cube_sums({"N": 185527, "triples": [[13, 43, 47], [13, 43, 47]]})
        with self.assertRaises(runtime.CandidateError):
            v.prime_cube_sums({"N": 3*9**3, "triples": [[9, 9, 9]]})

    def test_taxicab_representations(self):
        ta5 = [[38787, 365757], [107839, 362753], [205292, 342952], [221424, 336588], [231518, 331954]]
        self.assertTrue(all(a**3+b**3 == 48988659276962496 for a, b in ta5))
        self.assertEqual(v.two_cube_sums({"m": 48988659276962496, "pairs": ta5}, bound=5*10**16)[0], 5)
        self.assertEqual(v.two_cube_sums({"m": 87539319, "pairs": [[167, 436], [228, 423], [255, 414]]}, bound=5*10**16)[0], 3)
        for bad in ({"m": 1729, "pairs": [[1, 12], [1, 12]]}, {"m": 1729, "pairs": [[12, 1]]}, {"m": 1730, "pairs": [[1, 12]]},
                    {"m": 6*10**16, "pairs": [[1, 1]]}):
            with self.assertRaises(runtime.CandidateError):
                v.two_cube_sums(bad, bound=5*10**16)


class Divisors(unittest.TestCase):
    def test_distinct_divisor_count_runs(self):
        published = [1, 1, 4, 9, 45, 76, 270, 2204]        # A048892
        for n, start in enumerate(published, 1):
            first = next(s for s in range(1, 5000) if len({tau(x) for x in range(s, s+n)}) == n)
            self.assertEqual(first, start)
            self.assertEqual(v.divisor_count_run({"start": start, "length": n})[0], n)
        self.assertEqual(v.divisor_count_run({"start": 1745175039, "length": 14})[0], 14)
        with self.assertRaises(runtime.CandidateError):
            v.divisor_count_run({"start": 2204, "length": 9})

    def test_powerful_counts(self):
        powerful = [n for n in range(1, 301**2) if all(e >= 2 for e in trial_factor(n).values())]
        for k in range(1, 300):
            expected = sum(1 for p in powerful if k*k < p < (k+1)**2)
            self.assertEqual(v.powerful_between_squares(k)[0], expected)
        for count, k in enumerate([1, 2, 5, 31, 234, 1822, 3611, 17329, 1511067, 524827]):   # A119242
            self.assertEqual(v.powerful_between_squares(k)[0], count)
        self.assertEqual(v.powerful_between_squares(472532614)[0], 11)

    def test_runs_with_large_prime_factors(self):
        published = [2, 5, 13, 19, 55, 65, 113, 151, 151, 226]   # A327909
        for n, start in enumerate(published, 1):
            first = next(s for s in range(1, 1000) if all(max(trial_factor(x), default=1) > n for x in range(s, s+n)))
            self.assertEqual(first, start)
        self.assertEqual(v.large_prime_factor_run({"start": 6961, "length": 30})[0], 30)
        with self.assertRaises(runtime.CandidateError):
            v.large_prime_factor_run({"start": 6961, "length": 31})

    def test_totient_multiplicities(self):
        counts = {}
        for value in phi_table(3000)[1:]:
            counts[value] = counts.get(value, 0)+1
        for m in range(1, 200):
            self.assertEqual(v.totient_preimages(m), counts.get(m, 0))
        self.assertEqual(v.totient_preimages(1360800), 1000)
        self.assertEqual(v.inverse_totient_multiplicity(1360800)[0], 1360800)
        with self.assertRaises(runtime.CandidateError):
            v.inverse_totient_multiplicity(1360801)

    def test_k_sigma_k(self):
        preimages = {}
        for k in range(1, 1000):
            preimages.setdefault(k*sigma(k), []).append(k)
        for m, ks in preimages.items():
            if m < 10**6:
                self.assertEqual(v.k_sigma_k({"M": m, "k": ks})[0], len(ks))
        self.assertEqual(preimages[333312], [336, 372, 434])
        for bad in ({"M": 333312, "k": [336, 336]}, {"M": 333312, "k": [335]}, {"M": 333312, "k": [336, 372, 434, 1000]}):
            with self.assertRaises(runtime.CandidateError):
                v.k_sigma_k(bad)

    def test_factorial_factorizations(self):
        def narrowest(n):
            target = factorial(n)
            divisors = [d for d in range(1, isqrt(target)+1) if target % d == 0]

            def product_of_subset(items, rest):
                if rest == 1:
                    return True
                return bool(items) and ((rest % items[0] == 0 and product_of_subset(items[1:], rest//items[0]))
                                        or product_of_subset(items[1:], rest))
            for span in range(1, 50):
                for a in divisors:                        # the smallest factor is below sqrt(n!)
                    b = a+span
                    if target % (a*b) == 0 and product_of_subset([d for d in range(a+1, b) if target % d == 0], target//(a*b)):
                        return span
        for n, expected in {2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 4, 9: 6, 10: 7}.items():   # A388302
            self.assertEqual(narrowest(n), expected, n)
        optimum = list(range(9, 31)) + [32, 35, 36]
        self.assertEqual(v.factorial_factorization_span(optimum)[0], 27)
        with self.assertRaises(runtime.CandidateError):
            v.factorial_factorization_span(list(range(9, 31)) + [32, 35])


class AdditiveSets(unittest.TestCase):
    def best_subset(self, universe, valid):
        """Largest valid subset for a hereditary property (subsets of valid sets are valid)."""
        best = 0
        for r in range(1, len(universe)+1):
            if not any(valid(list(c)) for c in itertools.combinations(universe, r)):
                break
            best = r
        return best

    def test_square_subset_sums(self):
        def valid(s):
            return not any(isqrt(sum(c))**2 == sum(c) for r in range(1, len(s)+1) for c in itertools.combinations(s, r))
        self.assertEqual(self.best_subset(list(range(1, 13)), valid), 5)        # A372040: a(5) = 12
        self.assertEqual(self.best_subset(list(range(1, 12)), valid), 4)
        rng = random.Random(3)
        for _ in range(200):
            s = sorted(rng.sample(range(1, 135), rng.randint(1, 7)))
            if valid(s):
                self.assertEqual(v.no_square_subset_sum(s)[0], len(s))
            else:
                with self.assertRaises(runtime.CandidateError):
                    v.no_square_subset_sum(s)

    def test_sidon_squares(self):
        def valid(s):
            sums = [x*x+y*y for i, x in enumerate(s) for y in s[i:]]
            return len(sums) == len(set(sums))
        for n, expected in enumerate([1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 9, 9], 1):   # A390813
            self.assertEqual(self.best_subset(list(range(1, n+1)), valid), expected)
        with self.assertRaises(runtime.CandidateError):
            v.sidon_squares([1, 7, 5, 5])
        with self.assertRaises(runtime.CandidateError):
            v.sidon_squares([1, 5, 7])          # 1 + 49 = 25 + 25

    def test_squarefree_sums(self):
        sf = [n > 0 and all(e == 1 for e in trial_factor(n).values()) for n in range(200)]

        def valid(s):
            return all(sf[x+y] for i, x in enumerate(s) for y in s[i:])
        published = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]   # A392164
        for n in range(1, 25):
            self.assertEqual(self.best_subset(list(range(1, n+1)), valid), published[n-1])
        with self.assertRaises(runtime.CandidateError):
            v.squarefree_sumset([2])              # 2 + 2 = 4

    def test_one_repeated_sum(self):
        def valid(s):
            counts = {}
            for i, x in enumerate(s):
                for y in s[i:]:
                    counts[x+y] = counts.get(x+y, 0)+1
            return sum(c > 1 for c in counts.values()) <= 1
        for n, expected in enumerate([1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6], 1):   # A389182
            self.assertEqual(self.best_subset(list(range(1, n+1)), valid), expected)
        self.assertEqual(v.one_repeated_sum([1, 2, 4, 5])[0], 4)          # only 6 = 1 + 5 = 2 + 4 repeats
        with self.assertRaises(runtime.CandidateError):
            v.one_repeated_sum([1, 2, 3, 4])                           # 5 and 6 both repeat

    def test_no_element_divides_two_others(self):
        def valid(s):
            return all(sum(1 for y in s if y != x and y % x == 0) <= 1 for x in s)
        for n, expected in enumerate([1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11], 1):   # A038372
            self.assertEqual(self.best_subset(list(range(1, n+1)), valid), expected)
        self.assertEqual(v.no_divisor_of_two(list(range(15, 45)))[0], 30)
        with self.assertRaises(runtime.CandidateError):
            v.no_divisor_of_two([3, 6, 9])

    def test_rulers_and_bases(self):
        def ruler(n):
            return next(k for k in range(1, n+2) for s in itertools.combinations(range(n+1), k)
                        if {b-a for a in s for b in s if b > a} >= set(range(1, n+1)))

        def basis(n):
            return next(k for k in range(1, n+2) for s in itertools.combinations(range(n+1), k)
                        if {a+b for a in s for b in s} >= set(range(n+1)))
        for n, expected in enumerate([1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]):     # A046693
            self.assertEqual(ruler(n), expected)
        for n, expected in enumerate([1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]):     # A066063
            self.assertEqual(basis(n), expected)
        self.assertEqual(v.sparse_ruler([0, 1, 4, 6], n=6)[0], 4)
        self.assertEqual(v.additive_basis([0, 1, 3, 4], n=8)[0], 4)
        with self.assertRaises(runtime.CandidateError):
            v.sparse_ruler([0, 1, 4, 6], n=7)
        with self.assertRaises(runtime.CandidateError):
            v.additive_basis([0, 1, 3, 4], n=9)

    def test_van_der_waerden(self):
        def valid(colours, long):
            n = len(colours)
            for length, colour in ((3, 0), (long, 1)):
                for d in range(1, n):
                    for a in range(n-(length-1)*d):
                        if all(colours[a+i*d] == colour for i in range(length)):
                            return False
            return True
        for long, w in ((3, 9), (4, 18)):
            longest = 0
            stack = [[]]
            while stack:
                c = stack.pop()
                longest = max(longest, len(c))
                for colour in (0, 1):
                    if valid(c+[colour], long):
                        stack.append(c+[colour])
            self.assertEqual(longest, w-1)
        with self.assertRaises(runtime.CandidateError):
            v.van_der_waerden([0, 1, 0, 1, 0])

    def test_distinct_multiples(self):
        published = [2, 3, 4, 6, 6, 9, 9, 11, 13, 15, 15, 17, 16, 19, 20, 25, 24, 27, 26, 29]   # A390246

        def matching(n, k):
            match = {}

            def augment(i, seen):
                for x in range((n//i+1)*i, n+k, i):
                    if x not in seen:
                        seen.add(x)
                        if x not in match or augment(match[x], seen):
                            match[x] = i
                            return True
                return False
            if all(augment(i, set()) for i in range(n, 0, -1)):
                return [x for i in range(1, n+1) for x, j in match.items() if j == i]
            return None
        for n in range(1, 21):
            k = next(k for k in range(1, 10*n) if matching(n, k))
            self.assertEqual(k, published[n-1])
            self.assertEqual(v.distinct_multiples(matching(n, k), n=n)[0], k)
        with self.assertRaises(runtime.CandidateError):
            v.distinct_multiples([4, 4, 6], n=3)

    def test_square_products(self):
        def valid(s):
            for q in itertools.combinations_with_replacement(sorted(s), 4):
                p = prod(q)
                if isqrt(p)**2 == p and q[0]*q[3] != q[1]*q[2]:
                    return False
            return True
        for n, expected in enumerate([1, 2, 3, 3, 4, 5, 6, 6, 6, 7, 8], 1):     # A387584
            self.assertEqual(self.best_subset(list(range(1, n+1)), valid), expected)
            rng = random.Random(n)
            for _ in range(20):
                s = sorted(rng.sample(range(1, n+1), rng.randint(1, n)))
                if valid(s):
                    self.assertEqual(v.square_product_condition(s, n=n)[0], len(s))
                else:
                    with self.assertRaises(runtime.CandidateError):
                        v.square_product_condition(s, n=n)
        primes = [p for p in range(2, 201) if v.is_prime(p)]
        semiprimes = [p*q for p in primes for q in primes if p < q and p*q <= 200]
        self.assertEqual(v.square_product_condition(primes+semiprimes)[0], 102)

    def test_progressions_and_sidon_variants(self):
        rng = random.Random(4)
        for _ in range(50):
            s = sorted(rng.sample(range(1, 513), 32))
            members = set(s)
            expected = len({b-a for a in s for b in s if b > a and 2*b-a in members})
            self.assertEqual(v.three_progression_differences(s)[0], expected)
        self.assertEqual(v.sidon_extension([1, 2, 5, 11, 24])[0], 5)
        for bad in ([1, 2, 5, 11, 12], [1, 2, 5, 20]):
            with self.assertRaises(runtime.CandidateError):
                v.sidon_extension(bad)
        self.assertEqual(v.unique_differences([0, 1, 4, 9, 11], covered=5)[0], 11)     # a Golomb ruler
        self.assertEqual(v.unique_differences([0, 1, 3], covered=3)[0], 3)
        for bad, covered in (([0, 1, 4, 9, 11], 6), ([0, 1, 2], 2), ([1, 2, 4], 3), ([0, 1, 3, 4], 3)):
            with self.assertRaises(runtime.CandidateError):
                v.unique_differences(bad, covered=covered)

    def test_smooth_runs_after_prime_squares(self):
        for p in [q for q in range(2, 600) if v.is_prime(q)]:
            k = 0
            while max(trial_factor(p*p+k+1)) <= p:
                k += 1
            if k:
                self.assertEqual(v.smooth_after_prime_square({"p": p, "k": k})[0], k)
            with self.assertRaises(runtime.CandidateError):
                v.smooth_after_prime_square({"p": p, "k": k+1})


class Congruences(unittest.TestCase):
    def test_primorial_gaps(self):
        for count, first_missing in ((4, 12), (5, 16), (6, 20)):              # A389839
            m = prod([2, 3, 5, 7, 11, 13][:count])
            units = [x for x in range(1, 2*m+2) if gcd(x, m) == 1]
            first = {}
            for a, b in zip(units, units[1:]):
                first.setdefault(b-a, a)
            complete = 0
            while complete+2 in first:
                complete += 2
            self.assertEqual(complete+2, first_missing)
            self.assertEqual(v.primorial_gaps([[x, g] for g, x in first.items()], count=count)[0], complete)
        with self.assertRaises(runtime.CandidateError):
            v.primorial_gaps([[1, 2]], count=4)

    def test_odd_covering_minimum_for_45(self):
        base, divisors = 45, [3, 5, 9, 15, 45]
        masks = [[sum(1 << x for x in range(r, base, d)) for r in range(d)] for d in divisors]
        best = base
        for choice in itertools.product(*masks):
            covered = 0
            for mask in choice:
                covered |= mask
            best = min(best, base-bin(covered).count("1"))
        self.assertEqual(best, 16)                                           # A399152(22)
        self.assertEqual(v.odd_covering([[3, 0], [5, 0], [15, 1]], base=15)[0], 7)
        for bad in ([[2, 0]], [[3, 0], [3, 1]], [[7, 0]]):
            with self.assertRaises(runtime.CandidateError):
                v.odd_covering(bad, base=45)

    def test_double_and_budget_covers(self):
        primes = [2, 3, 5, 7]
        for residues in itertools.product(*[range(p) for p in primes]):
            length = 0
            while sum((length+1) % p == r for p, r in zip(primes, residues)) >= 2:
                length += 1
            self.assertEqual(v.double_prime_cover(list(residues), largest_prime=7)[0], length)
            classes = [[p, r] for p, r in zip(primes, residues) if p > 2]
            length = 0
            while any((length+1) % p == r for p, r in classes):
                length += 1
            self.assertEqual(v.budget_prime_cover(classes, largest_prime=7)[0], length)
        with self.assertRaises(runtime.CandidateError):
            v.budget_prime_cover([[2, 0], [3, 1], [5, 2]], largest_prime=7)         # 1/2 + 1/3 + 1/5 > 1
        with self.assertRaises(runtime.CandidateError):
            v.budget_prime_cover([[9, 0]], largest_prime=10)

    def test_disjoint_classes(self):
        rng = random.Random(5)
        for _ in range(300):
            classes = [[m, rng.randrange(m)] for m in rng.sample(range(20, 41), 3)]
            disjoint = not any(any(x % m2 == r2 for x in range(r1, m1*m2, m1))
                               for (m1, r1), (m2, r2) in itertools.combinations(classes, 2))
            if disjoint:
                self.assertAlmostEqual(v.disjoint_classes_mass(classes)[0], sum(1/m for m, _ in classes), places=12)
            else:
                with self.assertRaises(runtime.CandidateError):
                    v.disjoint_classes_mass(classes)


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

    def test_malformed_inputs(self):
        cases = [(v.mahler_measure, [0]), (v.mahler_measure, [1, 0, 0]), (v.mahler_measure, [1.5, 1]), (v.mahler_measure, [1]*14),
                 (v.consecutive_prime_progression, [3, 2]), (v.jacobsthal_cover, [0]*29), (v.jacobsthal_cover, [2]+[0]*29),
                 (v.divisor_count_run, {"start": 0, "length": 3}), (v.divisor_count_run, {"start": 10**10, "length": 2}),
                 (v.powerful_between_squares, 0), (v.inverse_totient_multiplicity, 10**8+1),
                 (v.factorial_factorization_span, [factorial(30)]), (v.sparse_ruler, [0, 0, 200]),
                 (v.additive_basis, [True, 1]), (v.distinct_multiples, list(range(51, 101))),
                 (v.van_der_waerden, [2]), (v.primorial_gaps, [[1, 3]]), (v.odd_covering, [[3, 3]]),
                 (v.double_prime_cover, [0]*167), (v.unique_differences, [0]+list(range(40, 80))),
                 (v.sidon_extension, [1, 2, 5]), (v.three_progression_differences, list(range(1, 32))),
                 (v.smooth_after_prime_square, {"p": 4, "k": 1}), (v.k_sigma_k, {"M": 10**19, "k": [1]}),
                 (v.budget_prime_cover, [[1009, 0]]), (v.disjoint_classes_mass, [[19, 0]]), (v.two_cube_sums, {"m": 9})]
        for function, value in cases:
            with self.subTest(function=function.__name__, value=str(value)[:40]), self.assertRaises(runtime.CandidateError):
                function(value)


if __name__ == "__main__":
    unittest.main()
