"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import datetime

import numpy as np


def season_day(day):
    """Days since 1 January of the snow season (negative in October-December); day counts from 1970-01-01."""
    out = []
    for d in day:
        date = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(d))
        out.append((date - datetime.date(date.year + (date.month >= 10), 1, 1)).days)
    return np.array(out, float)


def features(station, index):
    depth = station["depth_cm"][index]
    return np.column_stack([np.ones(len(index)), season_day(station["day"][index]), np.log(depth), np.full(len(index), station["elevation_m"] / 1000.0)])


def estimate_swe(train, test):
    """Bulk density as a linear function of season day, log depth and elevation, fitted on the training stations."""
    x, y = [], []
    for station in train.values():
        index = station["target_index"].astype(int)
        x.append(features(station, index))
        y.append(station["swe_mm"][index] / (10.0 * station["depth_cm"][index]))
    beta, *_ = np.linalg.lstsq(np.vstack(x), np.concatenate(y), rcond=None)
    out = {}
    for name, station in test.items():
        index = station["target_index"].astype(int)
        density = np.clip(features(station, index) @ beta, 0.05, 0.65)
        out[name] = density * 10.0 * station["depth_cm"][index]
    return out
# EVOLVE-BLOCK-END
