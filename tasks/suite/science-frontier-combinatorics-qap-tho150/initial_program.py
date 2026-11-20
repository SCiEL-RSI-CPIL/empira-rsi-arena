"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def solve(a, b):
    # Rank-matching start: rows of the first matrix with the largest totals go to the
    # rows of the second matrix with the smallest totals (heavy flows on short distances).
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(len(a), dtype=np.int64)
    p[heavy] = central
    return p.tolist()
# EVOLVE-BLOCK-END
