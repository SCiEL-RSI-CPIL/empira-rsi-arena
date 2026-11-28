"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def fit_surface(quotes):
    """Flat-smile seed: each slice's total variance equals its quote nearest the money (b = 0)."""
    params, previous = [], 0.0
    for i in range(len(quotes["maturities"])):
        j = int(np.argmin(np.abs(quotes["log_moneyness"][i])))
        level = max(float(quotes["total_variance"][i][j]), previous + 1e-6)  # keep calendar order
        previous = level
        params.append([level, 0.0, 0.0, 0.0, 0.1])
    return np.array(params)
# EVOLVE-BLOCK-END
