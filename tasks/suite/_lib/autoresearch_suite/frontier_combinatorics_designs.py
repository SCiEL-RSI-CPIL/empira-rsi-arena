"""Exact checks for covering designs and difference triangle sets.

Both evaluators recompute every property of the candidate's object with
integer arithmetic; nothing the candidate reports is trusted. The scores are
the quantities tabulated in the literature (number of blocks of a covering,
scope of a difference triangle set), so a lower score than the recorded best
known value is a new upper bound for that open problem.
"""
from __future__ import annotations

import itertools
import math
import numbers

import numpy as np

from .runtime import require


def integer(value, name):
    """A Python or NumPy integer (booleans and floats are rejected)."""
    require(isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)),
            f"{name} must be an integer (floats and booleans are not accepted)")
    return int(value)


def sequence(value, name):
    require(isinstance(value, (list, tuple, np.ndarray)), f"{name} must be a list, tuple or array")
    if isinstance(value, np.ndarray):
        require(value.ndim >= 1, f"{name} must not be a scalar array")
    return value


# --- covering designs --------------------------------------------------------

def covering_design(blocks, v, k, t):
    """Return the number of blocks of a (v, k, t) covering design.

    ``blocks`` is a sequence of distinct k-subsets of {0, ..., v-1}, each given
    as a sequence of k distinct integers. The design is valid when every
    t-subset of the points lies in at least one block. Every t-subset of every
    block is ranked in the combinatorial number system and marked; the check is
    exhaustive over all C(v, t) t-subsets.
    """
    v, k, t = integer(v, "v"), integer(k, "k"), integer(t, "t")
    require(1 <= t <= k <= v <= 64, "require 1 <= t <= k <= v <= 64")
    total = math.comb(v, t)
    blocks = sequence(blocks, "blocks")
    require(1 <= len(blocks) <= total, f"return 1..{total} blocks (C(v,t) blocks always suffice)")
    rows, seen = [], set()
    for index, block in enumerate(blocks):
        block = sequence(block, f"block {index}")
        require(len(block) == k, f"block {index} must contain exactly k = {k} points")
        points = sorted(integer(x, "point") for x in block)
        require(all(0 <= x < v for x in points), f"points must be integers in 0..{v - 1}")
        require(len(set(points)) == k, f"block {index} repeats a point")
        key = tuple(points)
        require(key not in seen, f"block {index} duplicates an earlier block")
        seen.add(key)
        rows.append(points)
    rows = np.asarray(rows, dtype=np.int64)
    # rank(c_1 < ... < c_t) = sum_i C(c_i, i) enumerates the t-subsets of {0..v-1} as 0..C(v,t)-1.
    binom = np.array([[math.comb(c, i) for i in range(1, t + 1)] for c in range(v)], dtype=np.int64)
    combos = np.array(list(itertools.combinations(range(k), t)), dtype=np.int64)
    covered = np.zeros(total, dtype=bool)
    for start in range(0, len(rows), 4096):
        chosen = rows[start:start + 4096][:, combos]  # (blocks, C(k,t), t), ascending within each t-subset
        ranks = binom[chosen, np.arange(t)].sum(axis=2)
        covered[ranks.ravel()] = True
    missing = int(total - covered.sum())
    if missing:
        rank = int(np.flatnonzero(~covered)[0])
        require(False, f"{missing} of {total} t-subsets are not covered; first uncovered: {unrank(rank, t)}")
    pair_count = len(rows) * math.comb(k, t)
    return len(rows), {"blocks": len(rows), "t_subsets": total, "covered_t_subsets": total,
                       "excess_incidences": pair_count - total, "schoenheim_bound": schoenheim(v, k, t),
                       "verification": "exhaustive_t_subset_cover"}


def unrank(rank, t):
    """Inverse of the combinatorial number system rank used above."""
    out = []
    for i in range(t, 0, -1):
        c = i - 1
        while math.comb(c + 1, i) <= rank:
            c += 1
        out.append(c)
        rank -= math.comb(c, i)
    return sorted(out)


def schoenheim(v, k, t):
    """Schoenheim's lower bound on C(v, k, t), reported for orientation only."""
    bound = 1
    for i in range(t - 1, -1, -1):
        bound = -(-(v - i) * bound // (k - i))
    return bound


# --- difference triangle sets ------------------------------------------------

def difference_triangle_set(rows, n_rows, k, max_entry=1_000_000):
    """Return the scope of an (n_rows, k) difference triangle set.

    ``rows`` holds exactly ``n_rows`` rows of k + 1 integers each, strictly
    increasing and starting at 0. Every positive within-row difference
    a_j - a_i (i < j) over all rows must be distinct. The scope, the largest
    entry, is minimized; m(n, k) is the least possible scope.
    """
    n_rows, k = integer(n_rows, "n_rows"), integer(k, "k")
    limit = integer(max_entry, "max_entry")
    require(1 <= n_rows <= 64 and 1 <= k <= 32, "require 1 <= n_rows <= 64 and 1 <= k <= 32")
    rows = sequence(rows, "rows")
    require(len(rows) == n_rows, f"return exactly {n_rows} rows")
    marks, differences = [], {}
    for r, row in enumerate(rows):
        row = sequence(row, f"row {r}")
        require(len(row) == k + 1, f"row {r} must contain exactly k + 1 = {k + 1} integers")
        values = [integer(x, "entry") for x in row]
        require(values[0] == 0, f"row {r} must start at 0 (normalized form)")
        require(all(a < b for a, b in zip(values, values[1:])), f"row {r} must be strictly increasing")
        require(values[-1] <= limit, f"entries must not exceed {limit}")
        for i, j in itertools.combinations(range(k + 1), 2):
            d = values[j] - values[i]
            require(d not in differences, f"difference {d} occurs twice (rows {differences.get(d)} and {r})")
            differences[d] = r
        marks.append(values)
    scope = max(row[-1] for row in marks)
    trivial = n_rows * k * (k + 1) // 2
    return scope, {"scope": scope, "sum_of_row_scopes": sum(row[-1] for row in marks),
                   "distinct_differences": len(differences), "trivial_lower_bound": trivial,
                   "verification": "exact_all_differences"}
