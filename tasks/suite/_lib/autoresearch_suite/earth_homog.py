"""Homogenisation of monthly temperature networks on a synthetic benchmark with known truth.

Long station records contain non-climatic jumps (relocations, instrument and observing-time
changes), gradual drifts (for example urbanisation), outliers and gaps; homogenisation methods
try to remove them before climate trends are estimated. Following the design of the HOME
benchmark (Venema et al. 2012), the evaluator generates 40 networks of 10 correlated monthly
temperature series of 100 years (a shared regional signal with a trend, station noise and
annual cycles), and inserts break inhomogeneities (about five per century per station, sizes
with standard deviation 0.8 deg C and a seasonal cycle), network-wide clustered breaks,
gradual drifts, outliers and missing data. The candidate returns homogenised series; the score
is the mean over stations of the centred root-mean-square difference (CRMSD) between the
homogenised and true annual means.
"""
from __future__ import annotations

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

RAW_SOURCES = ()
DATA = "homogenisation/synthetic_networks.json.gz"
DERIVED = {DATA: "4049ee8e272a8c4b9dc4a16bce47fd8c8ec2cb81ce1a3c8b42883913046a9e96"}
NETWORKS, STATIONS, YEARS, SEED = 40, 10, 100, 4207
MONTHS = 12 * YEARS
MIN_MONTHS_PER_YEAR = 10


def ar1(rng, n, phi, sd):
    out = np.empty(n)
    out[0] = rng.normal(0, sd)
    noise = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + noise[t]
    return out


def network(rng):
    """(truth, observed) arrays of shape (STATIONS, MONTHS); observed has NaN gaps."""
    month = np.arange(MONTHS) % 12
    common = ar1(rng, MONTHS, 0.6, 1.2) + rng.uniform(0.0, 2.0) * np.arange(MONTHS) / MONTHS
    truth = np.empty((STATIONS, MONTHS))
    for s in range(STATIONS):
        cycle = rng.uniform(6.0, 12.0) * -np.cos(2 * np.pi * (month + 0.5) / 12)
        truth[s] = rng.normal(10.0, 3.0) + cycle + common + ar1(rng, MONTHS, 0.3, rng.uniform(0.3, 0.6))
    offsets = np.zeros_like(truth)
    for s in range(STATIONS):  # breaks: the most recent segment is homogeneous, earlier ones are offset
        for position in rng.integers(12, MONTHS - 12, rng.poisson(5.0 * YEARS / 100)):
            size = rng.normal(0.0, 0.8) * (1 + 0.3 * np.cos(2 * np.pi * (month - rng.uniform(0, 12)) / 12))
            offsets[s, :position] += size[:position]
    if rng.random() < 0.3:  # a network-wide change of instruments or observing practice
        position, size = rng.integers(120, MONTHS - 120), rng.normal(0.3, 0.2)
        for s in rng.choice(STATIONS, rng.integers(3, 7), replace=False):
            offsets[s, :position] += size + rng.normal(0, 0.05)
    for s in np.flatnonzero(rng.random(STATIONS) < 0.1):  # gradual drift over 20-50 years
        start, length = rng.integers(0, MONTHS - 600), rng.integers(240, 600)
        ramp = np.clip((np.arange(MONTHS) - start) / length, 0.0, 1.0)
        offsets[s] += rng.uniform(-1.0, 1.0) * (ramp - 1.0)
    observed = truth + offsets
    outliers = rng.random(observed.shape) < 0.01
    observed[outliers] += rng.choice([-1, 1], outliers.sum()) * rng.uniform(3.0, 5.0, outliers.sum())
    for s in np.flatnonzero(rng.random(STATIONS) < 0.5):  # networks grow: late starts
        observed[s, :12 * rng.integers(1, 50)] = np.nan
    observed[rng.random(observed.shape) < 0.02] = np.nan
    return truth, observed


def derive(paths):
    rng = np.random.default_rng(SEED)
    truth, observed = zip(*(network(rng) for _ in range(NETWORKS)))
    round_ = lambda a: [[None if not np.isfinite(v) else round(float(v), 3) for v in row] for row in a]  # noqa: E731
    return {DATA: {"generator": f"earth_homog.network, numpy default_rng({SEED})", "networks": NETWORKS, "stations": STATIONS,
                   "years": YEARS, "observed": [round_(o) for o in observed], "truth": [round_(t) for t in truth]}}


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])
    to_array = lambda nets: [np.array([[np.nan if v is None else v for v in row] for row in net], float) for net in nets]  # noqa: E731
    return to_array(raw["observed"]), to_array(raw["truth"])


def annual(series, mask):
    """Annual means over the months in ``mask``; years with too few months are NaN."""
    values = np.where(mask, series, 0.0).reshape(-1, YEARS, 12).sum(axis=2)
    counts = mask.reshape(-1, YEARS, 12).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts >= MIN_MONTHS_PER_YEAR, values / counts, np.nan)


def crmsd(a, b):
    d = a - b
    d = d - np.nanmean(d, axis=-1, keepdims=True)
    return np.sqrt(np.nanmean(d ** 2, axis=-1))


def trend_per_century(series):
    x = np.arange(series.shape[-1], dtype=float)
    ok = np.isfinite(series)
    return float(np.polyfit(x[ok], series[ok], 1)[0] * 100.0) if ok.sum() > 10 else 0.0


def score(module, cpu_seconds=120):
    observed, truth = load()
    calendar = {"year": np.repeat(np.arange(1901, 1901 + YEARS), 12).astype(float), "month": np.tile(np.arange(1, 13), YEARS).astype(float)}
    output = earth_sandbox.call(module, "homogenize", [observed, calendar], cpu_seconds=cpu_seconds, memory_mb=3000)
    require(isinstance(output, list) and len(output) == NETWORKS, f"homogenize must return a list of {NETWORKS} arrays")
    station_scores, raw_scores, monthly, trend_err, raw_trend_err = [], [], [], [], []
    for net, (obs, true) in enumerate(zip(observed, truth)):
        mask = np.isfinite(obs)
        try:
            result = np.asarray(output[net], dtype=float)
        except (TypeError, ValueError):
            result = None
        require(result is not None and result.shape == obs.shape, f"network {net}: expected an array of shape {obs.shape}")
        require(np.all(np.isfinite(result[mask])) and np.all(np.abs(result[mask]) < 100), f"network {net}: non-finite or implausible values")
        hom = np.where(mask, result, np.nan)
        station_scores += list(crmsd(annual(hom, mask), annual(true, mask)))
        raw_scores += list(crmsd(annual(obs, mask), annual(true, mask)))
        monthly += list(crmsd(hom, np.where(mask, true, np.nan)))
        anomalies = lambda x: np.nanmean(x - np.nanmean(x, axis=1, keepdims=True), axis=0)  # noqa: E731
        true_trend = trend_per_century(anomalies(annual(true, mask)))
        trend_err.append(abs(trend_per_century(anomalies(annual(hom, mask))) - true_trend))
        raw_trend_err.append(abs(trend_per_century(anomalies(annual(obs, mask))) - true_trend))
    details = {"stations": len(station_scores), "networks": NETWORKS, "unit": "deg C",
               "raw_annual_crmsd": round(float(np.mean(raw_scores)), 4), "monthly_crmsd": round(float(np.mean(monthly)), 4),
               "network_trend_abs_error_per_century": round(float(np.mean(trend_err)), 4),
               "raw_network_trend_abs_error_per_century": round(float(np.mean(raw_trend_err)), 4)}
    return float(np.mean(station_scores)), details
