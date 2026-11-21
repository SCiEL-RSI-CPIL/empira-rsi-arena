"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Seed for the register-coalescing tasks: affinity-biased greedy colouring.

Nodes are coloured one at a time, heaviest total affinity first; each node takes the legal colour with the
largest affinity weight to already coloured nodes (ties: smallest colour). Every uncoloured node interferes
with fewer than K nodes, so a legal colour always exists.
"""


def coalesce(problem):
    n, k = problem["nodes"], problem["registers"]
    neighbours = [[] for _ in range(n)]
    for a, b in problem["interference"]:
        neighbours[a].append(b)
        neighbours[b].append(a)
    partners = [[] for _ in range(n)]
    weight = [0] * n
    for a, b, w in problem["affinities"]:
        partners[a].append((b, w))
        partners[b].append((a, w))
        weight[a] += w
        weight[b] += w
    colour = [None] * n
    for v in range(k):
        colour[v] = v
    for v in sorted(range(k, n), key=lambda v: (-weight[v], v)):
        taken = {colour[u] for u in neighbours[v] if colour[u] is not None}
        gain = [0] * k
        for u, w in partners[v]:
            if colour[u] is not None:
                gain[colour[u]] += w
        legal = [c for c in range(k) if c not in taken]
        colour[v] = max(legal, key=lambda c: (gain[c], -c))
    return colour
# EVOLVE-BLOCK-END
