"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def select_exchanges(pool):
    """Greedy seed: take mutually compatible pairs (2-cycles) in index order."""
    edges = set(map(tuple, pool["edges"].tolist()))
    used, cycles = set(), []
    for a, b in sorted(edges):
        if a < b and (b, a) in edges and a not in used and b not in used:
            used.update((a, b))
            cycles.append([a, b])
    return cycles
# EVOLVE-BLOCK-END
