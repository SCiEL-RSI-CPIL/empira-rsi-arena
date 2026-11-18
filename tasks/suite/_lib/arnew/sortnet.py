"""Sorting networks, verified exactly with the 0-1 principle on all 2^n binary inputs (bit-parallel, n <= 32).

A network is a list of comparators (i, j) with 0 <= i < j < n; a comparator puts the smaller value on wire i. The
network sorts every input iff it sorts every 0-1 input (Knuth, TAOCP 5.3.4). The 2^n inputs are processed in blocks: the
low BLOCK_BITS wires hold a bitset over all 2^BLOCK_BITS patterns, the higher wires are constant (all zeros or all ones) within a block, and
every assignment of the higher wires is one block (n = 32 takes 256 blocks and about half a minute).
"""
from __future__ import annotations

import numpy as np

from .common import CandidateError, require

MAX_EXACT_N = 32
BLOCK_BITS = 24


def check(comparators, n):
    if n > MAX_EXACT_N:
        raise NotImplementedError(f"exact 0-1 verification is implemented for n <= {MAX_EXACT_N} only (2^n inputs); n = {n} needs a symmetry- or SAT-based verifier")
    try:
        comps = [(int(a), int(b)) for a, b in comparators]
    except (TypeError, ValueError):
        raise CandidateError("return a list of (i, j) integer pairs") from None
    require(len(comps) > 0, "the network has no comparators")
    require(all(0 <= i < j < n for i, j in comps), f"every comparator needs 0 <= i < j < {n}")
    low = min(n, BLOCK_BITS)
    words = max(1, (1 << low) // 64)  # below 6 wires the 2^n patterns fit in one word; a zero-length array would accept every network (fixed 2026-11-18)
    idx = np.arange(words, dtype=np.uint64)
    ones, zeros = ~np.uint64(0), np.uint64(0)
    base = []
    for i in range(low):
        if i < 6:
            const = 0
            for x in range(64):
                if (x >> i) & 1:
                    const |= 1 << x
            base.append(np.full(words, const, dtype=np.uint64))
        else:
            base.append(np.where(((idx >> np.uint64(i - 6)) & np.uint64(1)) == 1, ones, zeros))
    depth_of = [0] * n
    for i, j in comps:
        depth_of[i] = depth_of[j] = max(depth_of[i], depth_of[j]) + 1
    for block in range(1 << (n - low)):
        wires = [w.copy() for w in base] + [np.full(words, ones if (block >> (i - low)) & 1 else zeros, dtype=np.uint64) for i in range(low, n)]
        for i, j in comps:
            a, b = wires[i], wires[j]
            wires[i], wires[j] = a & b, a | b
        bad = np.zeros(words, dtype=np.uint64)
        for k in range(n - 1):
            bad |= wires[k] & ~wires[k + 1]
        require(not bad.any(), "the network does not sort every 0-1 input")
    return len(comps), {"comparators": len(comps), "depth": max(depth_of), "wires": n, "verification": "exact_zero_one_principle"}


def batcher(n):
    """Odd-even merge sort (Knuth 5.2.2 Algorithm M) for any n; a valid but not minimal network."""
    out = []
    t = max(1, (n - 1).bit_length())
    p = 1 << (t - 1)
    while p > 0:
        q, r, d = 1 << (t - 1), 0, p
        while d > 0:
            for i in range(0, n - d):
                if i & p == r:
                    out.append((i, i + d))
            d, q, r = q - p, q >> 1, p
        p >>= 1
    return out
