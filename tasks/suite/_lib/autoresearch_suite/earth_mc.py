"""Gutenberg-Richter b-value and magnitude of completeness from incomplete earthquake catalogues.

Every seismicity rate, forecast and hazard estimate starts from the b-value of the frequency-
magnitude distribution, which must be estimated above the magnitude of completeness Mc; errors
in Mc bias b (Woessner and Wiemer 2005). The evaluator generates 600 synthetic catalogues: true
magnitudes follow the Gutenberg-Richter law with b drawn from 0.6-1.4 and are thinned by a
cumulative-normal detection probability Phi((m - mu) / sigma) (Ogata and Katsura 1993), with mu
from 0.5-3.0, sigma from 0.05-0.5 and 150-5000 detected events; magnitudes are reported in 0.1
bins. The candidate returns b and Mc for every catalogue; the score is the mean absolute error of
b. Mc is defined as the 95 percent detection magnitude mu + 1.645 sigma and reported, not scored.
"""
from __future__ import annotations

import math

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

RAW_SOURCES = ()
DATA = "completeness/synthetic_catalogues.json.gz"
DERIVED = {DATA: "ee3f511891a7bf5a6e1697a0712354a45d58ac3d9a1526ae4b4ec4e6b620bb8f"}
CATALOGUES, SEED, BIN = 600, 1993, 0.1


def catalogue(rng):
    """(b, Mc95, binned detected magnitudes); magnitudes start at mu - 3 sigma, where detection is below 0.14 percent."""
    from scipy.special import ndtr  # preparation-only dependency
    b, mu, sigma = rng.uniform(0.6, 1.4), rng.uniform(0.5, 3.0), rng.uniform(0.05, 0.5)
    wanted = int(round(np.exp(rng.uniform(np.log(150), np.log(5000)))))
    m0, beta = mu - 3 * sigma, b * math.log(10)
    kept = []
    while sum(len(k) for k in kept) < wanted:
        m = m0 + rng.exponential(1.0 / beta, 4 * wanted)
        kept.append(m[rng.random(len(m)) < ndtr((m - mu) / sigma)])
    magnitudes = np.concatenate(kept)[:wanted]
    return b, mu + 1.645 * sigma, np.round(np.round(magnitudes / BIN) * BIN, 1)


def derive(paths):
    rng = np.random.default_rng(SEED)
    rows = [catalogue(rng) for _ in range(CATALOGUES)]
    return {DATA: {"generator": f"earth_mc.catalogue, numpy default_rng({SEED})", "bin": BIN,
                   "b": [round(r[0], 6) for r in rows], "mc95": [round(r[1], 6) for r in rows],
                   "magnitudes": [r[2].tolist() for r in rows]}}


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])
    return [np.array(m, float) for m in raw["magnitudes"]], np.array(raw["b"]), np.array(raw["mc95"])


def aki_utsu(magnitudes, mc):
    """Maximum-likelihood b above the bin-centred threshold mc for 0.1-binned magnitudes (Aki 1965; Utsu 1965)."""
    above = magnitudes[magnitudes >= mc - 1e-9]
    if len(above) < 2 or above.mean() <= mc - BIN / 2:
        return float("nan")
    return float(np.log10(np.e) / (above.mean() - (mc - BIN / 2)))


def max_curvature(magnitudes):
    """Magnitude bin with the most events (maximum-curvature Mc, before any correction)."""
    values, counts = np.unique(np.round(magnitudes, 1), return_counts=True)
    return float(values[np.argmax(counts)])


def score(module, cpu_seconds=120):
    catalogues, b_true, mc_true = load()
    output = earth_sandbox.call(module, "estimate_b_and_mc", [catalogues], cpu_seconds=cpu_seconds)
    estimates = ec.as_array(output, (len(catalogues), 2), "b-value and Mc estimates")
    require(np.all((estimates[:, 0] >= 0.1) & (estimates[:, 0] <= 3.0)), "b-values must lie in [0.1, 3]")
    require(np.all((estimates[:, 1] >= -2.0) & (estimates[:, 1] <= 10.0)), "Mc must lie in [-2, 10]")
    error = np.abs(estimates[:, 0] - b_true)
    maxc = np.array([aki_utsu(m, max_curvature(m) + 0.2) for m in catalogues])
    oracle = np.array([aki_utsu(m, np.ceil(mc / BIN - 1e-9) * BIN) for m, mc in zip(catalogues, mc_true)])
    small = np.array([len(m) < 500 for m in catalogues])
    details = {"catalogues": len(catalogues), "b_bias": round(float(np.mean(estimates[:, 0] - b_true)), 5),
               "b_mae_small_catalogues": round(float(error[small].mean()), 5), "mc95_mae": round(float(np.mean(np.abs(estimates[:, 1] - mc_true))), 4),
               "maxc_plus_0.2_aki_utsu_b_mae": round(float(np.nanmean(np.abs(maxc - b_true))), 5),
               "true_mc_aki_utsu_b_mae": round(float(np.nanmean(np.abs(oracle - b_true))), 5)}
    return float(error.mean()), details
