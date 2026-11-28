"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def solve(n, edges):
    # One-flip local search from an alternating partition. edges: int64 array of rows (i, j, w), 0-based vertices.
    side = np.arange(n) % 2
    adj = [[] for _ in range(n)]
    for i, j, w in edges.tolist():
        adj[i].append((j, w))
        adj[j].append((i, w))
    for _ in range(20):
        improved = False
        for v in range(n):
            gain = sum(w if side[u] == side[v] else -w for u, w in adj[v])
            if gain > 0:
                side[v] ^= 1
                improved = True
        if not improved:
            break
    return side.tolist()
# EVOLVE-BLOCK-END
