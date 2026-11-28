"""Exact checks for q-ary covering codes and superpermutations.

Covering codes: every word of Z_q^n must lie within Hamming distance R of a
codeword; the check marks the union of all radius-R balls over the whole
space. Superpermutations: every permutation of n symbols must occur as a
contiguous block; every window of the word is inspected. Scores (code size,
word length) are the tabulated quantities K_q(n, R) and the superpermutation
length, so beating a recorded value is a new upper bound.
"""
from __future__ import annotations

import math

import numpy as np

from .frontier_combinatorics_designs import integer, sequence
from .runtime import require


# --- covering codes ----------------------------------------------------------

def covering_code(codewords, q, n, radius):
    """Return |C| for a q-ary code of length n with covering radius at most ``radius``.

    Each codeword is a sequence of n integers in 0..q-1; codewords must be
    distinct. The union of Hamming balls is built exactly by ``radius`` rounds
    of one-coordinate changes over all q**n words.
    """
    q, n, radius = integer(q, "q"), integer(n, "n"), integer(radius, "radius")
    require(2 <= q <= 9 and 1 <= n <= 20 and q ** n <= 1_000_000, "require 2 <= q <= 9, 1 <= n <= 20 and q**n <= 1,000,000")
    require(0 <= radius <= n, "require 0 <= radius <= n")
    space = q ** n
    codewords = sequence(codewords, "codewords")
    require(1 <= len(codewords) <= space, f"return 1..{space} codewords")
    weights = q ** np.arange(n, dtype=np.int64)
    encoded = []
    for index, word in enumerate(codewords):
        word = sequence(word, f"codeword {index}")
        require(len(word) == n, f"codeword {index} must have length n = {n}")
        digits = [integer(x, "symbol") for x in word]
        require(all(0 <= x < q for x in digits), f"symbols must be integers in 0..{q - 1}")
        encoded.append(int(np.dot(digits, weights)))
    require(len(set(encoded)) == len(encoded), "duplicate codewords are not permitted")
    covered = np.zeros(space, dtype=bool)
    covered[np.asarray(encoded, dtype=np.int64)] = True
    words = np.arange(space, dtype=np.int64)
    digits = [(words // weights[i]) % q for i in range(n)]
    for _ in range(radius):
        grown = covered.copy()
        for i in range(n):
            for delta in range(1, q):
                grown |= covered[words + (((digits[i] + delta) % q) - digits[i]) * weights[i]]
        covered = grown
    missing = int(space - covered.sum())
    if missing:
        first = int(np.flatnonzero(~covered)[0])
        require(False, f"{missing} of {space} words are farther than {radius} from the code; first: "
                       f"{[int(d[first]) for d in digits]}")
    ball = sum(math.comb(n, j) * (q - 1) ** j for j in range(radius + 1))
    return len(encoded), {"codewords": len(encoded), "space_size": space, "ball_volume": ball,
                          "sphere_covering_bound": -(-space // ball),
                          "verification": "exhaustive_ball_union"}


# --- superpermutations -------------------------------------------------------

def superpermutation(word, n, max_length=None):
    """Return the length of a word over {1..n} that contains all n! permutations.

    ``word`` is a string of the digits 1..n (n <= 9) or a sequence of integers
    in 1..n. Every length-n window with n distinct symbols is recorded; the
    word is valid when all n! permutations occur.
    """
    n = integer(n, "n")
    require(2 <= n <= 9, "require 2 <= n <= 9")
    total = math.factorial(n)
    limit = n * total if max_length is None else integer(max_length, "max_length")
    if isinstance(word, str):
        require(all(c in "0123456789" for c in word), "a string word may contain only the ASCII digits 1..n")
        symbols = [int(c) for c in word]
    else:
        word = sequence(word, "word")
        symbols = [integer(x, "symbol") for x in word]
    require(n <= len(symbols) <= limit, f"the word must have length {n}..{limit}")
    require(all(1 <= x <= n for x in symbols), f"symbols must be 1..{n}")
    found = set()
    for i in range(len(symbols) - n + 1):
        window = tuple(symbols[i:i + n])
        if len(set(window)) == n:
            found.add(window)
    missing = total - len(found)
    require(missing == 0, f"{missing} of {total} permutations do not occur as contiguous blocks")
    # n! + (n-1)! + (n-2)! + n - 3 is the published lower bound (Houston, Pantone and Vatter), shown for orientation.
    lower = total + math.factorial(n - 1) + math.factorial(n - 2) + n - 3
    return len(symbols), {"length": len(symbols), "permutations": total, "waste": len(symbols) - total - n + 1,
                          "lower_bound_formula": lower, "verification": "exhaustive_windows"}
