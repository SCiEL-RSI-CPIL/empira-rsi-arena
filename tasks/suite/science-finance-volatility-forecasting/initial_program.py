"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class RollingVariance:
    """Seed: mean squared return of the last 63 trading days for every series."""

    def __init__(self, training_returns):
        self.window = list(training_returns[-63:])

    def forecast(self):
        return np.mean(np.square(self.window), axis=0)

    def update(self, returns):
        self.window = self.window[1:] + [returns]


def make_forecaster(training_returns, names):
    return RollingVariance(training_returns)
# EVOLVE-BLOCK-END
