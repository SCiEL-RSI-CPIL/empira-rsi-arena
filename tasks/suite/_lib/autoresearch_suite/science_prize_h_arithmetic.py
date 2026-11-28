"""Exact arithmetic helpers shared by the number-theory construction verifiers.

Primality, factorization, sieves and the argument checks used by science_prize_h and
science_prize_h_sets. Reference values and sources are in catalog/science/prize_h.json.
"""
from functools import lru_cache
from math import gcd, isqrt, prod

from .runtime import require
from .science_prize import integer, sequence


def _report(**extra):
    return {"lean_verified": False, **extra}


# --- arithmetic helpers -----------------------------------------------------

def is_prime(n):
    """Deterministic Miller-Rabin for n < 3.18e23 (the first twelve primes as bases)."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n-1, 0
    while d % 2 == 0:
        d, s = d//2, s+1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n-1):
            continue
        for _ in range(s-1):
            x = x*x % n
            if x == n-1:
                break
        else:
            return False
    return True


def _pollard_brent(n):
    """A nontrivial factor of the odd composite n (Brent's variant of Pollard's rho)."""
    for c in range(1, 200):
        y, m, g, r, q = 2, 128, 1, 1, 1
        f = lambda v: (v*v+c) % n
        while g == 1:
            x = y
            for _ in range(r):
                y = f(y)
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r-k)):
                    y = f(y)
                    q = q*abs(x-y) % n
                g = gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = f(ys)
                g = gcd(abs(x-ys), n)
        if g != n:
            return g
    raise ArithmeticError("factorization failed")


def factor(n):
    """Exact factorization: trial division by small primes, then Pollard-Brent with
    deterministic Miller-Rabin (valid for every n below 3.18e23)."""
    out = {}
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        while n % p == 0:
            out[p] = out.get(p, 0)+1
            n //= p
    p = 41
    while n > 1 and p <= 1000 and p*p <= n:
        while n % p == 0:
            out[p] = out.get(p, 0)+1
            n //= p
        p += 2
    stack = [n] if n > 1 else []
    while stack:
        m = stack.pop()
        if is_prime(m):
            out[m] = out.get(m, 0)+1
        else:
            d = _pollard_brent(m)
            stack += [d, m//d]
    return out


def divisor_count(n):
    return prod(e+1 for e in factor(n).values())


def primes_up_to(n):
    sieve = bytearray([1])*(n+1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, isqrt(n)+1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(range(p*p, n+1, p)))
    return [i for i in range(n+1) if sieve[i]]


@lru_cache(maxsize=4)
def squarefree_table(n):
    table = bytearray([1])*(n+1)
    for p in range(2, isqrt(n)+1):
        table[p*p::p*p] = bytearray(len(range(p*p, n+1, p*p)))
    return bytes(table)


def int_set(value, name, lower, upper, minimum=0, maximum=None):
    rows = sequence(value, name, minimum, upper-lower+1 if maximum is None else maximum)
    for x in rows:
        integer(x, name+" element", lower, upper)
    require(len(set(rows)) == len(rows), name+" elements must be distinct")
    return sorted(rows)


def run_certificate(value, bound, max_length):
    require(type(value) is dict and set(value) == {"start", "length"}, "return keys start and length only")
    start = integer(value["start"], "start", 1, bound)
    length = integer(value["length"], "length", 1, max_length)
    require(start+length-1 <= bound, "run exceeds the bound")
    return start, length
