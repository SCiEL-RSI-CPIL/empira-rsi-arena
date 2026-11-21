"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def forecast_yields(history, horizons):
    """Random-walk seed: the latest curve is the forecast for every horizon."""
    return np.repeat(history["yields"][-1][None, :], len(horizons), axis=0)
# EVOLVE-BLOCK-END
