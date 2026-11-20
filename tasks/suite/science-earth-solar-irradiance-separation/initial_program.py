"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def clearness(station, index):
    return np.clip(station["ghi"][index] / np.maximum(station["e0_horizontal"][index], 1.0), 0.0, 1.2)


def estimate_diffuse(train, test):
    """Logistic diffuse fraction kd = 1 / (1 + exp(a + b kt)) fitted by least squares in logit space."""
    kt, kd = [], []
    for station in train.values():
        index = station["target_index"].astype(int)
        kt.append(clearness(station, index))
        kd.append(station["dhi"][index] / station["ghi"][index])
    kt, kd = np.concatenate(kt), np.clip(np.concatenate(kd), 0.02, 0.98)
    b, a = np.polyfit(kt, np.log(1.0 / kd - 1.0), 1)
    out = {}
    for name, station in test.items():
        index = station["target_index"].astype(int)
        fraction = 1.0 / (1.0 + np.exp(a + b * clearness(station, index)))
        out[name] = fraction * station["ghi"][index]
    return out
# EVOLVE-BLOCK-END
