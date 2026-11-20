"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

LEADS = (12, 24, 36, 48, 72, 96, 120)


class ConstantMotion:
    """Persistence: extrapolate the last 12-h displacement."""

    def forecast(self, history):
        lat, lon = history["lat"], history["lon"]
        j = max(len(lat) - 3, 0)
        steps = np.array(LEADS) / (6.0 * max(len(lat) - 1 - j, 1))
        new_lat = np.clip(lat[-1] + steps * (lat[-1] - lat[j]), -89.0, 89.0)
        new_lon = (lon[-1] + steps * (lon[-1] - lon[j]) + 180.0) % 360.0 - 180.0
        return np.column_stack([new_lat, new_lon])


def fit_track_model(training_storms):
    return ConstantMotion()
# EVOLVE-BLOCK-END
