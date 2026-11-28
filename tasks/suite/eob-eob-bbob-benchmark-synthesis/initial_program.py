"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def objective(X):
    """Shifted Rastrigin-type function on [-5, 5]^10 (minimum 0 at x = 1.5)."""
    z = X - 1.5
    return np.sum(z ** 2, axis=1) + 10.0 * np.sum(1.0 - np.cos(2.0 * np.pi * z), axis=1)
# EVOLVE-BLOCK-END
