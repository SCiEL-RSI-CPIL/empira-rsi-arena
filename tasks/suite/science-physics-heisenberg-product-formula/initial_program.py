"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """Strang splitting repeated over stages/2 substeps: a = (1/2, 1/2) per substep, b = (1, 0)."""
    substeps = stages // 2
    a = [0.5 / substeps] * (2 * substeps)
    b = []
    for _ in range(substeps):
        b += [1.0 / substeps, 0.0]
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a, b
# EVOLVE-BLOCK-END
