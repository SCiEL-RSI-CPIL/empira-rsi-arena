"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def trade_schedule(problem):
    """TWAP seed: sell the same number of shares in every period."""
    return np.full(problem["periods"], problem["shares"] / problem["periods"])
# EVOLVE-BLOCK-END
