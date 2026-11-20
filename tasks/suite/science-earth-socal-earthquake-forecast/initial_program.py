"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class SmoothedSeismicity:
    """Time-independent forecast: Gaussian-smoothed past epicentres scaled to the past target rate."""

    def __init__(self, rates):
        self.rates = rates

    def forecast(self, history, day):
        return self.rates


def fit_quake_model(training, grid):
    lat_e, lon_e = np.asarray(grid["lat_edges"]), np.asarray(grid["lon_edges"])
    lat_c, lon_c = (lat_e[:-1] + lat_e[1:]) / 2, (lon_e[:-1] + lon_e[1:]) / 2
    shallow = training["depth"] <= grid["target_max_depth_km"]
    lat, lon = training["lat"][shallow], training["lon"][shallow]
    w_lat = np.exp(-0.5 * ((lat_c[:, None] - lat[None, :]) / 0.1) ** 2)
    w_lon = np.exp(-0.5 * ((lon_c[:, None] - lon[None, :]) / 0.1) ** 2)
    density = w_lat @ w_lon.T + 1e-3
    inside = ((training["mag"] >= grid["target_min_magnitude"]) & shallow & (training["lat"] >= lat_e[0])
              & (training["lat"] < lat_e[-1]) & (training["lon"] >= lon_e[0]) & (training["lon"] < lon_e[-1]))
    daily_rate = inside.sum() / grid["first_forecast_day"]
    return SmoothedSeismicity(daily_rate * density / density.sum())
# EVOLVE-BLOCK-END
