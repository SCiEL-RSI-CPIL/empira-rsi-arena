"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def solve(a, b):
    # Rank-matching start: the indices of a with the largest row sums are paired with the indices of b with the smallest row sums.
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(len(a), dtype=np.int64)
    p[heavy] = central
    return p.tolist()
# EVOLVE-BLOCK-END
