"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def allocate(values):
    """Round-robin seed: agents take turns picking their most valued remaining good (envy-free up to one good)."""
    n, m = values.shape
    owner, free = [0] * m, set(range(m))
    turn = 0
    while free:
        agent = turn % n
        good = max(free, key=lambda g: (values[agent, g], -g))
        owner[good] = agent
        free.remove(good)
        turn += 1
    return owner
# EVOLVE-BLOCK-END
