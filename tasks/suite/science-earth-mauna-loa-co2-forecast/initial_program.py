"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

LEADS = 12


def design(t):
    """Quadratic trend and two annual harmonics in decimal time t (years)."""
    w = 2 * np.pi * t
    return np.column_stack([np.ones_like(t), t, t ** 2, np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])


class TrendAndCycle:
    """Refit a quadratic trend plus mean seasonal cycle to the last ten years before every forecast."""

    def forecast(self, history):
        t = history["year"] + (history["month"] - 0.5) / 12.0
        recent = slice(len(t) - 120, len(t))
        origin = t[-1]
        x = design(t[recent] - origin)
        beta, *_ = np.linalg.lstsq(x, history["co2_ppm"][recent], rcond=None)
        future = np.arange(1, LEADS + 1) / 12.0
        return design(future) @ beta


def fit_co2_model(training):
    return TrendAndCycle()
# EVOLVE-BLOCK-END
