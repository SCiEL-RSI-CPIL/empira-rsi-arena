"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def choose_weights(excess_returns, context):
    """1/N seed: equal positions in every asset, the benchmark of DeMiguel, Garlappi and Uppal (2009)."""
    return np.ones(excess_returns.shape[1])
# EVOLVE-BLOCK-END
