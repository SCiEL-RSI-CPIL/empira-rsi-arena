"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def features(t, log_q):
    return np.column_stack([np.ones_like(t), log_q, log_q ** 2, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), t - 2015.0])


def estimate_nitrate(sites):
    """Per-site log-linear regression on discharge, season and trend (LOADEST-like) with smearing bias correction."""
    out = {}
    for name, site in sites.items():
        t, log_q = site["decimal_year"], np.log(site["discharge_m3s"])
        index = site["sample_index"].astype(int)
        y = np.log(np.maximum(site["sample_nitrate_mg_l"], 0.01))
        x = features(t[index], log_q[index])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        smearing = np.mean(np.exp(y - x @ beta))
        out[name] = np.clip(np.exp(features(t, log_q) @ beta) * smearing, 0.0, 100.0)
    return out
# EVOLVE-BLOCK-END
