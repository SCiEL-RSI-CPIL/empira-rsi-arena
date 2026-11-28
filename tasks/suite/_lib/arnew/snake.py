"""Snake-in-the-box: induced paths in the hypercube Q_n, verified exactly.

Records: the longest snakes on T. Ace's leaderboard (minortriad.com/snake), each replayed through check() by the record validation.
"""
from __future__ import annotations

from .common import CandidateError, exact_int, require


def check(vertices, n):
    try:
        v = [exact_int(x) for x in vertices]  # 3.7 must not be read as vertex 3
    except (TypeError, ValueError):
        raise CandidateError("return a list of integers") from None
    require(len(v) >= 2, "a snake needs at least two vertices")
    require(all(0 <= x < (1 << n) for x in v), f"vertices must be in [0, 2**{n})")
    pos = {x: k for k, x in enumerate(v)}
    require(len(pos) == len(v), "vertices must be distinct")
    for a, b in zip(v, v[1:]):
        require(bin(a ^ b).count("1") == 1, "consecutive vertices must differ in exactly one bit")
    for k, x in enumerate(v):
        for bit in range(n):
            other = pos.get(x ^ (1 << bit))
            require(other is None or abs(other - k) == 1, "non-consecutive vertices must differ in at least two bits")
    return len(v) - 1, {"edges": len(v) - 1, "dimension": n, "verification": "exact_induced_path"}
