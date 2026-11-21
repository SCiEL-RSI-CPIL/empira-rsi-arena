"""Independent exact verifiers for number-theory and additive-combinatorics constructions.

All scores are exact integer computations except the Mahler measure, which is
evaluated from polynomial roots after an exact test that the measure exceeds one.
Reference values and sources are in catalog/science/prize_h.json.
"""
from fractions import Fraction
from functools import lru_cache
from math import factorial, isqrt, prod

from .runtime import require
from .science_prize import integer, sequence
from .science_prize_h_arithmetic import (  # noqa: F401 - part of this module's interface
    _report, is_prime, _pollard_brent, factor, divisor_count, primes_up_to, squarefree_table, int_set,
    run_certificate,
)
from .science_prize_h_sets import (  # noqa: F401 - part of this module's interface
    no_square_subset_sum, sidon_squares, squarefree_sumset, one_repeated_sum, no_divisor_of_two,
    sparse_ruler, additive_basis, square_product_condition, van_der_waerden, distinct_multiples,
    primorial_gaps, three_progression_differences, sidon_extension, unique_differences,
    smooth_after_prime_square, _classes, odd_covering, _initial_run, double_prime_cover, budget_prime_cover,
    disjoint_classes_mass, two_cube_sums,
)


# --- polynomials --------------------------------------------------------------

def _poly_divmod(num, den):
    """Exact division of integer polynomials (coefficient lists, constant term first)."""
    num = list(num)
    q = [0]*(len(num)-len(den)+1) if len(num) >= len(den) else [0]
    for i in range(len(num)-len(den), -1, -1):
        c = Fraction(num[i+len(den)-1], den[-1])
        q[i] = c
        for j, d in enumerate(den):
            num[i+j] -= c*d
    while len(num) > 1 and num[-1] == 0:
        num.pop()
    return q, num


@lru_cache(maxsize=None)
def cyclotomic(m):
    poly = [-1]+[0]*(m-1)+[1]
    for d in range(1, m):
        if m % d == 0:
            poly, rem = _poly_divmod(poly, cyclotomic(d))
            assert all(r == 0 for r in rem)
    return tuple(int(c) for c in poly)


def _roots(coeffs):
    """Aberth-Ehrlich iteration for all complex roots (constant term first)."""
    n = len(coeffs)-1
    lead = coeffs[-1]
    monic = [c/lead for c in coeffs]
    radius = 1+max(abs(c) for c in monic[:-1])
    roots = [radius*complex(0.4, 0.9)**k for k in range(n)]

    def value(z):
        p, dp = 0j, 0j
        for c in reversed(monic):
            dp = dp*z+p
            p = p*z+c
        return p, dp

    for _ in range(500):
        worst = 0.0
        for i in range(n):
            p, dp = value(roots[i])
            if p == 0:
                continue
            ratio = p/dp if dp != 0 else p
            s = sum(1/(roots[i]-roots[j]) for j in range(n) if j != i and roots[i] != roots[j])
            step = ratio/(1-ratio*s)
            roots[i] -= step
            worst = max(worst, abs(step))
        if worst < 1e-15:
            break
    return roots


def mahler_measure(value, max_degree=12, max_coefficient=3):
    coeffs = sequence(value, "coefficients (constant term first)", 2, max_degree+1)
    for c in coeffs:
        integer(c, "coefficient", -max_coefficient, max_coefficient)
    require(coeffs[-1] != 0, "leading coefficient must be nonzero")
    poly = list(coeffs)
    while poly[0] == 0:
        poly.pop(0)
    require(len(poly) >= 2, "polynomial must have a nonzero root")
    rest = list(poly)
    for m in range(1, 64):
        phi = cyclotomic(m)
        if len(phi)-1 > len(poly)-1:
            continue
        while len(rest) >= len(phi):
            q, r = _poly_divmod(rest, phi)
            if any(x != 0 for x in r) or any(Fraction(x).denominator != 1 for x in q):
                break
            rest = [int(x) for x in q]
    # Kronecker: an integer polynomial has measure one iff it is +-x^k times cyclotomic factors.
    require(not (len(rest) == 1 and abs(rest[0]) == 1), "measure equals one (product of cyclotomic polynomials)")
    measure = abs(poly[-1])*prod(max(1.0, abs(z)) for z in _roots(poly))
    return round(measure, 10), _report(degree=len(coeffs)-1, measure=measure)


# --- primes -------------------------------------------------------------------

def consecutive_prime_progression(value, bound=10**9):
    start, step, length = sequence(value, "[start prime, difference, length]", 3, 3)
    integer(start, "start", 2, bound)
    integer(step, "difference", 1, 5000)
    integer(length, "length", 2, 64)
    require(start+(length-1)*step <= bound, "progression exceeds the bound")
    terms = [start+i*step for i in range(length)]
    require(all(is_prime(t) for t in terms), "every term must be prime")
    for a, b in zip(terms, terms[1:]):
        require(not any(is_prime(x) for x in range(a+1, b)), "terms must be consecutive primes")
    return length, _report(first=start, difference=step, length=length)


def jacobsthal_cover(value, count=30):
    primes = primes_up_to(1000)[:count]
    residues = sequence(value, "one residue per prime", count, count)
    for r, p in zip(residues, primes):
        integer(r, f"residue modulo {p}", 0, p-1)
    length = 0
    while any((length+1-r) % p == 0 for r, p in zip(residues, primes)):
        length += 1
    return length, _report(primes=count, largest_prime=primes[-1], covered_length=length, jacobsthal=length+1)


def distinct_prime_gaps(value, bound=10**8):
    start, length = run_certificate(value, bound, 200)
    require(is_prime(start), "start must be prime")
    primes = [start]
    while len(primes) < length+1:
        x = primes[-1]+1
        while not is_prime(x):
            x += 1
        require(x <= bound, "primes exceed the bound")
        primes.append(x)
    gaps = [b-a for a, b in zip(primes, primes[1:])]
    require(len(set(gaps)) == len(gaps), "gaps must be pairwise distinct")
    return length, _report(first_prime=start, gaps=gaps)


def prime_cube_sums(value, bound=10**10):
    require(type(value) is dict and set(value) == {"N", "triples"}, "return keys N and triples only")
    n = integer(value["N"], "N", 1, bound)
    rows = sequence(value["triples"], "prime triples", 1, 100)
    seen = set()
    for row in rows:
        p, q, r = sequence(row, "triple", 3, 3)
        for x in (p, q, r):
            integer(x, "prime", 2, 3000)
            require(is_prime(x), "entries must be prime")
        require(p <= q <= r and p**3+q**3+r**3 == n, "triple must be sorted and sum to N")
        require((p, q, r) not in seen, "duplicate triple")
        seen.add((p, q, r))
    return len(seen), _report(N=n, representations=len(seen))


def prime_plus_power_of_two(value, bound=10**8):
    n = integer(value, "n", 3, bound)
    count = sum(1 for k in range(n.bit_length()) if n-(1 << k) >= 2 and is_prime(n-(1 << k)))
    return count, _report(n=n, representations=count)


# --- divisors, totients and powerful numbers --------------------------------

def divisor_count_run(value, bound=10**10):
    start, length = run_certificate(value, bound, 40)
    counts = [divisor_count(x) for x in range(start, start+length)]
    require(len(set(counts)) == length, "divisor counts must be pairwise distinct")
    return length, _report(start=start, divisor_counts=counts)


def powerful_between_squares(value, bound=10**9):
    k = integer(value, "k", 1, bound)
    low, high = k*k, (k+1)**2
    cube_limit = int(round(high**(1/3)))+2
    squarefree = squarefree_table(cube_limit)
    count = 0
    for b in range(1, cube_limit+1):
        c = b**3
        if c >= high:
            break
        if not squarefree[b]:
            continue
        a_min = isqrt(low//c)
        while a_min*a_min*c <= low:
            a_min += 1
        a_max = isqrt((high-1)//c)
        count += max(0, a_max-a_min+1)
    return count, _report(k=k, powerful_numbers=count)


def large_prime_factor_run(value, bound=10**4):
    start, length = run_certificate(value, bound+200, 200)
    require(start <= bound, "the run must start at most at the bound")
    for x in range(start, start+length):
        require(max(factor(x) or {1: 1}) > length, f"{x} has no prime factor larger than the run length")
    return length, _report(start=start, length=length)


def totient_preimages(m):
    divisors = [d for d in _divisors(m)]
    primes = sorted((d+1 for d in divisors if is_prime(d+1)), reverse=True)

    @lru_cache(maxsize=None)
    def count(n, i):
        total = 1 if n == 1 else 0
        for j in range(i, len(primes)):
            p = primes[j]
            if n % (p-1):
                continue
            rest = n//(p-1)
            while True:
                total += count(rest, j+1)
                if rest % p:
                    break
                rest //= p
        return total

    return count(m, 0)


def _divisors(n):
    divisors = [1]
    for p, e in factor(n).items():
        divisors = [d*p**k for d in divisors for k in range(e+1)]
    return sorted(divisors)


def inverse_totient_multiplicity(value, k=1000, maximum=10**8):
    m = integer(value, "m", 1, maximum)
    count = totient_preimages(m)
    require(count == k, f"phi(x) = m has {count} solutions, not {k}")
    return m, _report(m=m, solutions=count)


def sigma(n):
    return prod((p**(e+1)-1)//(p-1) for p, e in factor(n).items())


def k_sigma_k(value, bound=10**18, max_solutions=1000):
    """Certificate: M and distinct k with k*sigma(k) = M. Score = number of certified k (a lower bound for f(M))."""
    require(type(value) is dict and set(value) == {"M", "k"}, "return keys M and k only")
    m = integer(value["M"], "M", 1, bound)
    ks = int_set(value["k"], "k", 1, isqrt(m), 1, max_solutions)
    for k in ks:
        require(k*sigma(k) == m, f"{k}*sigma({k}) != M")
    return len(ks), _report(M=m, solutions=len(ks))


# --- factorials -----------------------------------------------------------------

def factorial_factorization_span(value, n=30):
    parts = sequence(value, "increasing factors", 2, n)
    for x in parts:
        integer(x, "factor", 1, factorial(n))
    require(all(a < b for a, b in zip(parts, parts[1:])), "factors must be strictly increasing")
    require(prod(parts) == factorial(n), f"product must equal {n}!")
    return parts[-1]-parts[0], _report(n=n, factors=len(parts), span=parts[-1]-parts[0])
