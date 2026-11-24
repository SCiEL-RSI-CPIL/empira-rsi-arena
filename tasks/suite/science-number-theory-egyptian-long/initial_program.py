"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Subset dynamic programming over divisors of a locally chosen common denominator.
    # Denominators are tried in decreasing order: a valid but long representation.
    target, reached = 2520, {0: []}
    for d in range(target, 4, -1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
