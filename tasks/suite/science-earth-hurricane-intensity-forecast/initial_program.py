"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Persistence:
    """Keep the current maximum sustained wind."""

    def forecast(self, history):
        return np.full(7, float(history["vmax_kt"][-1]))


def fit_intensity_model(training_storms):
    return Persistence()
# EVOLVE-BLOCK-END
