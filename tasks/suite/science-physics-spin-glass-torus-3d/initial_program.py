"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random


def ground_state(instance):
    """Randomized greedy start plus single-spin local search, best of a few restarts."""
    n = instance["nodes"]
    neighbours = [[] for _ in range(n)]
    for i, j, w in instance["edges"]:
        neighbours[i].append((j, w))
        neighbours[j].append((i, w))
    rng = random.Random(20260101 + n)
    best, best_cut = None, None
    for _ in range(8):
        spins = [rng.choice((-1, 1)) for _ in range(n)]
        improved = True
        while improved:
            improved = False
            for i in range(n):
                # Flipping i changes the cut by the weight of its satisfied minus unsatisfied edges.
                delta = sum(w * spins[i] * spins[j] for j, w in neighbours[i])
                if delta > 0:
                    spins[i] = -spins[i]
                    improved = True
        cut = sum(w for i, j, w in instance["edges"] if spins[i] != spins[j])
        if best_cut is None or cut > best_cut:
            best, best_cut = list(spins), cut
    return best
# EVOLVE-BLOCK-END
