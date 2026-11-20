"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class FreeDrift:
    """Ice velocity = complex factor times the geostrophic wind plus a mean current (Thorndike and Colony 1982)."""

    def __init__(self, factor, current):
        self.factor, self.current = factor, current

    def forecast(self, history, today):
        w = self.factor * (today["ug"] + 1j * today["vg"]) + self.current
        return np.column_stack([w.real, w.imag])


def fit_drift_model(training):
    wind = training["ug"] + 1j * training["vg"]
    ice = training["u"] + 1j * training["v"]
    (factor, current), *_ = np.linalg.lstsq(np.column_stack([wind, np.ones_like(wind)]), ice, rcond=None)
    return FreeDrift(factor, current)
# EVOLVE-BLOCK-END
