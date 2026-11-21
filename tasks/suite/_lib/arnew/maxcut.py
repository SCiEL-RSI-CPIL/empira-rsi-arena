"""Max-cut on fixed G-set graphs (Helmberg and Rendl), scored exactly.

Data: the G-set file format, `n m` followed by `i j w` lines (1-based vertices). The candidate returns one side
(0 or 1) per vertex; the score is the total weight of edges whose endpoints are on different sides.
"""
from __future__ import annotations

import numpy as np

from .common import CandidateError, read_data, require


def instance(name):
    tokens = read_data("gset", name).split()
    n, m = int(tokens[0]), int(tokens[1])
    require(len(tokens) >= 2 + 3 * m, f"{name}: expected {m} edges")
    edges = np.array(tokens[2:2 + 3 * m], dtype=np.int64).reshape(m, 3)
    edges[:, :2] -= 1
    return n, edges


def cut_value(sides, n, edges):
    try:
        given = np.asarray(sides).reshape(-1)
        s = given.astype(np.int64)
        exact = bool((given == s).all())  # 0.5 or nan must not be read as 0
    except (TypeError, ValueError):
        raise CandidateError("return a sequence of n integers (0 or 1)") from None
    require(len(s) == n, f"return exactly n = {n} values, one side per vertex")
    require(exact and bool(np.isin(s, (0, 1)).all()), "every side must be 0 or 1")
    differ = s[edges[:, 0]] != s[edges[:, 1]]
    return int(edges[differ, 2].sum()), {"cut": int(edges[differ, 2].sum()), "n": n, "edges": len(edges), "verification": "exact_integer_cut"}
