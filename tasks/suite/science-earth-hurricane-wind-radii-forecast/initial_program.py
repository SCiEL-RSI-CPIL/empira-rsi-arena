"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Persistence:
    """Keep the current 34-kt wind radii at every lead."""

    def forecast(self, history, future):
        return np.tile(history["r34"][-1], (len(future["lead_hours"]), 1))


def fit_radii_model(training_storms):
    return Persistence()
# EVOLVE-BLOCK-END
