"""Seeded input generators for the additional held-out QuixBugs tests.

QuixBugs ships 3 to 13 JSON test cases per program, so a patch that passes
them may still be wrong. At preparation time each generator below draws
inputs that satisfy the program's documented precondition (the QuixBugs
docstrings), the reference fix computes the expected outputs, and the pairs
become additional held-out tests (differential testing against the reference
fix). Inputs whose correct output is not unique under the documented
specification are avoided (unique values for searches, unique longest common
subsequences), so a repair is never failed for choosing another valid answer.
Sizes are small because several reference programs are exponential.
"""
from __future__ import annotations

import random
import zlib

COUNT = 40
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _ints(rng, n, lo, hi):
    return [rng.randint(lo, hi) for _ in range(n)]


def _unique(rng, n, lo, hi):
    return rng.sample(range(lo, hi + 1), n)


def _word(rng, n, alphabet="abc"):
    return "".join(rng.choice(alphabet) for _ in range(n))


def _nested(rng, depth):
    out = []
    for _ in range(rng.randint(0, 4)):
        roll = rng.random()
        if depth < 3 and roll < 0.35:
            out.append(_nested(rng, depth + 1))
        elif roll < 0.8:
            out.append(rng.randint(-9, 99))
        else:
            out.append(_word(rng, rng.randint(1, 3)))
    return out


def _almost_sorted(rng, n, k):
    """A permutation of n unique sorted values in which every element is at most k places from its sorted index."""
    values = sorted(_unique(rng, n, -50, 50))
    while True:
        order = sorted(range(n), key=lambda i: i + rng.uniform(0, k + 0.999))
        if all(abs(pos - i) <= k for pos, i in enumerate(order)):
            return [values[i] for i in order]


def _palindrome(rng):
    n = rng.randint(1, 7)
    if rng.random() < 0.25:
        return [9] * n
    half = [rng.randint(0, 9) for _ in range((n + 1) // 2)]
    half[0] = rng.randint(1, 9)
    return half + half[: n // 2][::-1]


def _rpn(rng, depth=0):
    if depth >= 3 or rng.random() < 0.3:
        return [float(rng.randint(1, 9))]
    return _rpn(rng, depth + 1) + _rpn(rng, depth + 1) + [rng.choice("+-*/")]


def _rpn_value(tokens):
    stack = []
    for token in tokens:
        if isinstance(token, float):
            stack.append(token)
            continue
        a, b = stack.pop(), stack.pop()
        if token == "/" and a == 0:
            return None
        stack.append({"+": b + a, "-": b - a, "*": b * a, "/": b / a if a else 0.0}[token])
    return stack.pop()


def _infix(rng):
    tokens = [rng.randint(0, 99)]
    for _ in range(rng.randint(0, 5)):
        tokens += [rng.choice("+-*/"), rng.randint(0, 99)]
    return tokens


def _text(rng):
    words = [_word(rng, rng.randint(1, 9), LETTERS) for _ in range(rng.randint(1, 14))]
    return " ".join(words)


def all_lcs(a, b):
    """Set of all longest common subsequences of two short strings (dynamic programming plus backtracking)."""
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            table[i][j] = table[i + 1][j + 1] + 1 if a[i] == b[j] else max(table[i + 1][j], table[i][j + 1])
    memo = {}

    def back(i, j):
        if (i, j) not in memo:
            if i == n or j == m:
                memo[i, j] = {""}
            elif a[i] == b[j]:
                memo[i, j] = {a[i] + rest for rest in back(i + 1, j + 1)}
            else:
                out = set()
                if table[i + 1][j] == table[i][j]:
                    out |= back(i + 1, j)
                if table[i][j + 1] == table[i][j]:
                    out |= back(i, j + 1)
                memo[i, j] = out
        return memo[i, j]
    return back(0, 0)


def _lcs_pair(rng):
    while True:
        a, b = _word(rng, rng.randint(0, 7), "abcd"), _word(rng, rng.randint(0, 7), "abcd")
        if len(all_lcs(a, b)) == 1:
            return [a, b]


def _sorted_search(rng, unique):
    n = rng.randint(0, 12)
    arr = sorted(_unique(rng, n, -30, 30) if unique else _ints(rng, n, -5, 5))
    x = rng.choice(arr) if arr and rng.random() < 0.7 else rng.randint(-32, 32)
    return [arr, x]


def _knapsack(rng):
    items = [[rng.randint(1, 30), rng.randint(1, 60)] for _ in range(rng.randint(0, 8))]
    return [rng.randint(0, 80), items]


def _hanoi(rng):
    start, end = rng.sample([1, 2, 3], 2)
    return [rng.randint(0, 6), start, end]


def _next_permutation(rng):
    while True:
        perm = _unique(rng, rng.randint(2, 7), 0, 20)
        if perm != sorted(perm, reverse=True):
            return [perm]


def _subsequences(rng):
    a = rng.randint(-2, 8)
    return [a, a + rng.randint(-1, 8), rng.randint(1, 4)]


def _kheapsort(rng):
    n = rng.randint(1, 10)
    k = rng.randint(0, n - 1)
    return [_almost_sorted(rng, n, k), k]


def _coins(rng):
    return [_unique(rng, rng.randint(0, 4), 1, 12), rng.randint(-3, 30)]


GENERATORS = {
    "bitcount": lambda r: [_bits(r)],
    "bucketsort": lambda r: (lambda k: [_ints(r, r.randint(0, 15), 0, k - 1), k])(r.randint(1, 12)),
    "find_first_in_sorted": lambda r: _sorted_search(r, unique=False),
    "find_in_sorted": lambda r: _sorted_search(r, unique=True),
    "flatten": lambda r: [_nested(r, 0)],
    "gcd": lambda r: [r.randint(0, 10 ** 6), r.randint(0, 10 ** 6)] if r.random() < 0.7 else [r.randint(0, 50), r.randint(0, 50)],
    "get_factors": lambda r: [r.randint(1, 5000)],
    "hanoi": _hanoi,
    "is_valid_parenthesization": lambda r: [_word(r, r.randint(0, 12), "()")],
    "kheapsort": _kheapsort,
    "knapsack": _knapsack,
    "kth": lambda r: (lambda arr: [arr, r.randint(0, len(arr) - 1)])(_ints(r, r.randint(1, 12), -9, 9)),
    "lcs_length": lambda r: [_word(r, r.randint(0, 10), "abcd"), _word(r, r.randint(0, 10), "abcd")],
    "levenshtein": lambda r: [_word(r, r.randint(0, 5), "abc"), _word(r, r.randint(0, 5), "abc")],
    "lis": lambda r: [_unique(r, r.randint(0, 12), -40, 40)],
    "longest_common_subsequence": _lcs_pair,
    "max_sublist_sum": lambda r: [_ints(r, r.randint(0, 12), -10, 10)],
    "mergesort": lambda r: [_ints(r, r.randint(0, 16), -20, 20)],
    "next_palindrome": lambda r: [_palindrome(r)],
    "next_permutation": _next_permutation,
    "pascal": lambda r: [r.randint(1, 30)],
    "possible_change": _coins,
    "powerset": lambda r: [list(r.sample("abcdefg", r.randint(0, 5)))],
    "quicksort": lambda r: [_ints(r, r.randint(0, 16), -6, 6)],
    "rpn_eval": lambda r: [_rpn(r)],
    "shunting_yard": lambda r: [_infix(r)],
    "sieve": lambda r: [r.randint(1, 300)],
    "sqrt": lambda r: [round(r.uniform(1.0, 500.0), 3), r.choice([0.1, 0.01, 0.001])],
    "subsequences": _subsequences,
    "to_base": lambda r: [r.randint(1, 10 ** 7), r.randint(2, 36)],
    "wrap": lambda r: [_text(r), r.randint(4, 30)],
}


def _bits(rng):
    roll = rng.random()
    if roll < 0.2:
        return 1 << rng.randint(0, 40)
    if roll < 0.3:
        return (1 << rng.randint(1, 40)) - 1
    return rng.randint(0, 1 << 40)


def generate(name, count=COUNT):
    """Seeded, de-duplicated argument lists for one program (precondition-respecting)."""
    rng = random.Random(zlib.crc32(name.encode()))
    seen, out = set(), []
    for _ in range(count * 20):
        args = GENERATORS[name](rng)
        if name == "rpn_eval" and _rpn_value(args[0]) is None:
            continue
        key = repr(args)
        if key not in seen:
            seen.add(key)
            out.append(args)
        if len(out) == count:
            break
    return out
