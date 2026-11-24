"""Independent arithmetic oracles of test_science_prize_h.py.

They are the functions of the source repository's test module, unchanged, kept in their own file so that the test
module stays within the source-length limit of this repository.
"""
from __future__ import annotations

from math import isqrt, prod


def sieve(n):
    flags = [True]*(n+1)
    flags[0:2] = [False, False]
    for i in range(2, isqrt(n)+1):
        if flags[i]:
            flags[i*i::i] = [False]*len(range(i*i, n+1, i))
    return flags


def trial_factor(n):
    out, d = {}, 2
    while d*d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0)+1
            n //= d
        d += 1
    if n > 1:
        out[n] = out.get(n, 0)+1
    return out


def tau(n):
    return prod(e+1 for e in trial_factor(n).values())


def sigma(n):
    return sum(d for d in range(1, n+1) if n % d == 0)


def phi_table(n):
    phi = list(range(n+1))
    for p in range(2, n+1):
        if phi[p] == p:
            for m in range(p, n+1, p):
                phi[m] -= phi[m]//p
    return phi
