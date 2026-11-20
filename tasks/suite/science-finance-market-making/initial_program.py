"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """Seed: quote both sides at the risk-neutral depth 1/k, ignoring inventory."""
    depth = np.full(len(inventory), 1.0 / params["k"])
    return depth, depth.copy()
# EVOLVE-BLOCK-END
