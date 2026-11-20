"""Probabilistic forecasts of aftershock counts after large shallow earthquakes worldwide (ComCat).

For every shallow (depth <= 70 km) M >= 6 mainshock that is not itself inside the zone of an
equal or larger event of the previous year, two forecasts are issued: at 1 day after the
mainshock for the next 7 days, and at 7 days for the next 30 days. The target is the number
of M >= 5 earthquakes (any depth) within D(M) = max(50, 2 * 10^(0.5 M - 1.8)) km of the
mainshock epicentre. Candidates return probabilities for 17 count bins; the score is the mean
logarithmic score of the observed bins over the 2015-2024 forecasts, which are requested in
time order in the sandboxed online protocol (history = all M >= 5 events before the forecast
start). This follows the setting of operational aftershock forecasting (Reasenberg and Jones
1989; Page et al. 2016).
"""
from __future__ import annotations

import csv
import datetime as dt
import math

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

EPOCH = dt.datetime(1980, 1, 1)
CHUNKS = ((1980, 1990), (1990, 2000), (2000, 2008), (2008, 2016), (2016, 2025))
URL = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={0}-01-01&endtime={1}-01-01"
       "&minmagnitude=5&eventtype=earthquake&orderby=time-asc")
RAW_SOURCES = tuple(f"comcat_global_m5_{a}_{b - 1}.csv" for a, b in CHUNKS)
SOURCE_SPECS = tuple({"name": name, "url": URL.format(a, b), "pinning": "live",
                      "license": "Public domain (U.S. Geological Survey ANSS Comprehensive Catalog)",
                      "license_url": "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
                      "citation": "U.S. Geological Survey, Earthquake Hazards Program (2017). Advanced National Seismic System (ANSS) Comprehensive Catalog of Earthquake Events and Products. doi:10.5066/F7MS3QZH",
                      "retrieved": "2026-10-10"} for name, (a, b) in zip(RAW_SOURCES, CHUNKS))
DATA = "aftershocks/comcat_global_m5_1980_2024.json.gz"
DERIVED = {DATA: "b90319360b22d4638c2018374a0f3d44eb153393251e6ef9dc19c5153de5c880"}
FIELDS = ("t", "lat", "lon", "depth", "mag")
MAIN_MAG, MAIN_DEPTH, TARGET_MAG = 6.0, 70.0, 5.0
FIRST_MAIN, TEST_START, LAST_START = 3653.0, 12784.0, 16397.0  # 1990-01-01, 2015-01-01, 2024-11-22 (days since 1980)
WINDOWS = ((1.0, 7.0), (7.0, 30.0))  # (start after mainshock, length), days
EDGES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 200)  # bin k holds counts in [EDGES[k], EDGES[k+1])
MIN_PROBABILITY = 1e-9


def derive(paths):
    seen, rows = set(), []
    for name in RAW_SOURCES:
        with open(paths[name], newline="") as stream:
            for row in csv.DictReader(stream):
                if row["id"] in seen or not row["mag"]:
                    continue
                seen.add(row["id"])
                when = dt.datetime.strptime(row["time"][:23], "%Y-%m-%dT%H:%M:%S.%f")
                rows.append((round((when - EPOCH).total_seconds() / 86400.0, 6), round(float(row["latitude"]), 4),
                             round(float(row["longitude"]), 4), round(float(row["depth"] or 0.0), 2), round(float(row["mag"]), 2)))
    rows.sort()
    return {DATA: {"source": "USGS ComCat FDSN event service, magnitude >= 5, worldwide, event type earthquake",
                   "time_unit": "days since 1980-01-01T00:00:00Z", **{f: list(c) for f, c in zip(FIELDS, zip(*rows))}}}


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])
    return {f: np.array(raw[f], float) for f in FIELDS}


def radius_km(magnitude):
    return np.maximum(50.0, 2.0 * 10 ** (0.5 * np.asarray(magnitude, float) - 1.8))


def distance_km(lat1, lon1, lat2, lon2):
    return ec.haversine_km(lat1, lon1, lat2, lon2)


def count_bin(count):
    return int(np.searchsorted(EDGES, count, side="right") - 1)


def cases(catalog):
    """Forecast cases (dict of arrays) in time order: mainshock fields, start, window, radius, observed count."""
    t, lat, lon, depth, mag = (catalog[f] for f in FIELDS)
    out = {k: [] for k in ("main_t", "main_lat", "main_lon", "main_depth", "main_mag", "start", "window_days", "radius_km", "count")}
    for i in np.flatnonzero((mag >= MAIN_MAG) & (depth <= MAIN_DEPTH) & (t >= FIRST_MAIN)):
        prior = np.arange(np.searchsorted(t, t[i] - 365.0), i)
        prior = prior[mag[prior] >= mag[i]]
        if len(prior) and np.any(distance_km(lat[prior], lon[prior], lat[i], lon[i]) <= radius_km(mag[prior])):
            continue  # inside the zone of an equal or larger recent event
        r = float(radius_km(mag[i]))
        for offset, length in WINDOWS:
            start = t[i] + offset
            if start > LAST_START:
                continue
            window = np.arange(np.searchsorted(t, start), np.searchsorted(t, start + length))
            window = window[mag[window] >= TARGET_MAG]
            n = int(np.sum(distance_km(lat[window], lon[window], lat[i], lon[i]) <= r))
            for key, value in zip(out, (t[i], lat[i], lon[i], depth[i], mag[i], start, length, r, n)):
                out[key].append(value)
    order = np.argsort(out["start"], kind="mergesort")
    return {k: np.array(v, float)[order] for k, v in out.items()}


def protocol(catalog):
    every = cases(catalog)
    train = every["start"] < TEST_START
    training_cases = {k: v[train] for k, v in every.items()}
    test = {k: v[~train] for k, v in every.items()}
    history = {f: catalog[f][catalog["t"] < TEST_START] for f in FIELDS}
    steps, previous = [], TEST_START
    for j in range(len(test["start"])):
        start = test["start"][j]
        lo, hi = np.searchsorted(catalog["t"], previous), np.searchsorted(catalog["t"], start)
        steps.append({"append": {f: catalog[f][lo:hi] for f in FIELDS},
                      "args": [{k: float(v[j]) for k, v in test.items() if k != "count"}]})
        previous = start
    info = {"bin_edges": np.array(EDGES, float), "target_min_magnitude": TARGET_MAG, "test_start_day": TEST_START}
    return {"args": [history, training_cases, info], "history": history}, steps, test


def validate(index, value):
    p = ec.as_array(value, (len(EDGES),), "count probabilities", low=MIN_PROBABILITY, high=1.0)
    require(abs(p.sum() - 1.0) <= 1e-6, f"count probabilities sum to {p.sum():.8f}, not 1")
    return p / p.sum()


def rj_expected(a, p, mag, offset, length, c=0.05, b=1.0):
    """Reasenberg-Jones expected number of M >= TARGET_MAG events in [offset, offset + length) days."""
    if abs(p - 1.0) < 1e-9:
        integral = np.log((offset + length + c) / (offset + c))
    else:
        integral = ((offset + length + c) ** (1 - p) - (offset + c) ** (1 - p)) / (1 - p)
    return 10 ** (a + b * (mag - TARGET_MAG)) * integral


def nb_log_pmf(n, mean, k):
    """Negative-binomial log probabilities with mean and shape k (variance mean + mean^2 / k)."""
    from scipy.special import gammaln
    mean = np.maximum(mean, 1e-12)
    return gammaln(n + k) - gammaln(k) - gammaln(n + 1.0) + k * np.log(k / (k + mean)) + n * np.log(mean / (k + mean))


def nb_bins(mean, k):
    """Binned negative-binomial probabilities, floored at MIN_PROBABILITY and renormalised."""
    pmf = np.exp(nb_log_pmf(np.arange(EDGES[-1], dtype=float), mean, k))
    probs = np.array([pmf[EDGES[i]:EDGES[i + 1]].sum() for i in range(len(EDGES) - 1)] + [max(0.0, 1.0 - pmf.sum())])
    probs = np.maximum(probs, MIN_PROBABILITY)
    return probs / probs.sum()


def fit_reference(training):
    """Grid maximum likelihood of (a, p, k) for the Reasenberg-Jones negative-binomial model on training cases."""
    offsets = training["start"] - training["main_t"]
    best = None
    for p in np.arange(0.7, 1.41, 0.05):
        for a in np.arange(-3.5, -0.99, 0.05):
            mean = rj_expected(a, p, training["main_mag"], offsets, training["window_days"])
            for k in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
                ll = float(np.sum(nb_log_pmf(training["count"], mean, k)))
                if best is None or ll > best[0]:
                    best = (ll, a, p, k)
    return best[1:]


def score(module, cpu_seconds=240):
    catalog = load()
    setup, steps, test = protocol(catalog)
    require(len(steps) > 100, "too few test forecasts")
    forecasts = earth_sandbox.run_online(module, "fit_aftershock_model", setup, steps, validate, cpu_seconds=cpu_seconds, memory_mb=3000)
    observed = np.array([count_bin(n) for n in test["count"]])
    log_score = np.log(np.array([f[b] for f, b in zip(forecasts, observed)]))
    a, p, k = fit_reference(setup["args"][1])
    reference = np.array([math.log(nb_bins(float(rj_expected(a, p, m, s - t0, w)), k)[b])
                          for m, s, t0, w, b in zip(test["main_mag"], test["start"], test["main_t"], test["window_days"], observed)])
    details = {"forecasts": len(log_score), "mainshocks": int(len(np.unique(test["main_t"]))), "training_cases": int(len(setup["args"][1]["start"])),
               "test_period": "forecast starts 2015-01-01 to 2024-11-22", "zero_count_fraction": round(float(np.mean(test["count"] == 0)), 4),
               "reference_mean_log_score": round(float(reference.mean()), 5), "reference_parameters": {"a": round(a, 3), "p": round(p, 3), "k": k},
               "information_gain_over_reference": round(float(log_score.mean() - reference.mean()), 5),
               "mean_log_score_7_day": round(float(log_score[test["window_days"] == 7.0].mean()), 5),
               "mean_log_score_30_day": round(float(log_score[test["window_days"] == 30.0].mean()), 5)}
    return float(log_score.mean()), details
