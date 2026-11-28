"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def estimate_covariance(returns):
    """Sample-covariance seed."""
    return np.cov(returns, rowvar=False)
# EVOLVE-BLOCK-END
