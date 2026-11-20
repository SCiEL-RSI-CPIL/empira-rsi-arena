"""One-day earthquake forecasts for southern California scored by Poisson likelihood (CSEP style).

The candidate fits a model on the 1981-2015 ComCat catalogue (M >= 2.5, 31-38 N, 122-113 W)
and is then stepped through every day of 2016-2024 in the sandboxed online protocol: before
each day it has seen all events up to 00:00 UTC of that day and nothing later, and returns
the expected number of target earthquakes (M >= 3.45, depth <= 30 km) in each 0.1-degree
cell of 32-37 N, 121-114 W during the day. The score is the information gain per target
earthquake of the joint Poisson log-likelihood over the reference time-independent
smoothed-seismicity forecast (Helmstetter et al. 2006; Zechar et al. 2010).
"""
from __future__ import annotations

import csv
import datetime as dt
import math

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

EPOCH = dt.datetime(1981, 1, 1)
CHUNKS = ((1981, 1986), (1986, 1991), (1991, 1996), (1996, 2001), (2001, 2006), (2006, 2011), (2011, 2016),
          (2016, 2020), (2020, 2025))
OUTER = {"minlatitude": 31, "maxlatitude": 38, "minlongitude": -122, "maxlongitude": -113}
URL = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={0}-01-01&endtime={1}-01-01"
       "&minlatitude=31&maxlatitude=38&minlongitude=-122&maxlongitude=-113&minmagnitude=2.5&eventtype=earthquake&orderby=time-asc")
RAW_SOURCES = tuple(f"comcat_socal_m2.5_{a}_{b - 1}.csv" for a, b in CHUNKS)
SOURCE_SPECS = tuple({"name": name, "url": URL.format(a, b), "pinning": "live",
                      "license": "Public domain (U.S. Geological Survey ANSS Comprehensive Catalog; southern California data from the Southern California Seismic Network)",
                      "license_url": "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
                      "citation": "U.S. Geological Survey, Earthquake Hazards Program (2017). Advanced National Seismic System (ANSS) Comprehensive Catalog of Earthquake Events and Products. doi:10.5066/F7MS3QZH",
                      "retrieved": "2026-10-10"} for name, (a, b) in zip(RAW_SOURCES, CHUNKS))
DATA = "quakes/comcat_socal_m2.5_1981_2024.json.gz"
DERIVED = {DATA: "397b00a735ef35a7946535ffe40ea858f78caf870efb207324d1a14243ec6ad5"}
LAT_EDGES = np.round(np.arange(32.0, 37.0001, 0.1), 1)
LON_EDGES = np.round(np.arange(-121.0, -113.9999, 0.1), 1)
TARGET_MAGNITUDE, TARGET_DEPTH = 3.45, 30.0
FIRST_DAY = (dt.datetime(2016, 1, 1) - EPOCH).days
LAST_DAY = (dt.datetime(2024, 12, 31) - EPOCH).days
MIN_RATE, MAX_DAILY_TOTAL = 1e-10, 1e4
FIELDS = ("t", "lat", "lon", "depth", "mag")


def derive(paths):
    seen, rows = set(), []
    for name in RAW_SOURCES:
        with open(paths[name], newline="") as stream:
            for row in csv.DictReader(stream):
                if row["id"] in seen or not row["mag"]:
                    continue
                seen.add(row["id"])
                when = dt.datetime.strptime(row["time"][:23], "%Y-%m-%dT%H:%M:%S.%f")
                days = (when - EPOCH).total_seconds() / 86400.0
                rows.append((round(days, 6), round(float(row["latitude"]), 4), round(float(row["longitude"]), 4),
                             round(float(row["depth"] or 0.0), 2), round(float(row["mag"]), 2)))
    rows.sort()
    columns = list(zip(*rows))
    return {DATA: {"source": "USGS ComCat FDSN event service, M >= 2.5, 31-38 N, 122-113 W, event type earthquake",
                   "time_unit": "days since 1981-01-01T00:00:00Z", **{f: list(c) for f, c in zip(FIELDS, columns)}}}


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])
    return {f: np.array(raw[f], float) for f in FIELDS}


def select(catalog, mask):
    return {f: catalog[f][mask] for f in FIELDS}


def target_counts(catalog):
    """(days, n_lat, n_lon) counts of target events per test day and cell."""
    keep = ((catalog["mag"] >= TARGET_MAGNITUDE) & (catalog["depth"] <= TARGET_DEPTH) & (catalog["t"] >= FIRST_DAY)
            & (catalog["t"] < LAST_DAY + 1) & (catalog["lat"] >= LAT_EDGES[0]) & (catalog["lat"] < LAT_EDGES[-1])
            & (catalog["lon"] >= LON_EDGES[0]) & (catalog["lon"] < LON_EDGES[-1]))
    day = np.floor(catalog["t"][keep]).astype(int) - FIRST_DAY
    i = np.clip(np.floor((catalog["lat"][keep] - LAT_EDGES[0]) / 0.1 + 1e-9).astype(int), 0, len(LAT_EDGES) - 2)
    j = np.clip(np.floor((catalog["lon"][keep] - LON_EDGES[0]) / 0.1 + 1e-9).astype(int), 0, len(LON_EDGES) - 2)
    counts = np.zeros((LAST_DAY - FIRST_DAY + 1, len(LAT_EDGES) - 1, len(LON_EDGES) - 1))
    np.add.at(counts, (day, i, j), 1.0)
    return counts


def smoothed_reference(training, bandwidth=0.15):
    """Time-independent forecast: Gaussian-smoothed M >= 2.5 epicentres scaled to the training target rate."""
    lat_c, lon_c = (LAT_EDGES[:-1] + LAT_EDGES[1:]) / 2, (LON_EDGES[:-1] + LON_EDGES[1:]) / 2
    shallow = training["depth"] <= TARGET_DEPTH
    density = np.zeros((len(lat_c), len(lon_c)))
    for lat, lon in zip(training["lat"][shallow], training["lon"][shallow]):
        density += np.outer(np.exp(-0.5 * ((lat_c - lat) / bandwidth) ** 2), np.exp(-0.5 * ((lon_c - lon) / bandwidth) ** 2))
    density = density / density.sum() + 1e-6
    density /= density.sum()
    inside = ((training["mag"] >= TARGET_MAGNITUDE) & shallow & (training["lat"] >= LAT_EDGES[0]) & (training["lat"] < LAT_EDGES[-1])
              & (training["lon"] >= LON_EDGES[0]) & (training["lon"] < LON_EDGES[-1]))
    return density * inside.sum() / FIRST_DAY


def poisson_loglik(rates, counts):
    """Joint Poisson log-likelihood of observed counts (log n! included)."""
    occupied = counts > 0
    log_factorial = sum(math.lgamma(n + 1.0) for n in counts[occupied])
    return float(np.sum(counts[occupied] * np.log(rates[occupied])) - np.sum(rates) - log_factorial)


def grid():
    return {"lat_edges": LAT_EDGES.copy(), "lon_edges": LON_EDGES.copy(), "target_min_magnitude": TARGET_MAGNITUDE,
            "target_max_depth_km": TARGET_DEPTH, "first_forecast_day": float(FIRST_DAY), "last_forecast_day": float(LAST_DAY)}


def protocol(catalog):
    """(setup, steps): training catalogue, then one step per test day appending the previous day's events."""
    training = select(catalog, catalog["t"] < FIRST_DAY)
    steps = []
    for day in range(FIRST_DAY, LAST_DAY + 1):
        append = None if day == FIRST_DAY else select(catalog, (catalog["t"] >= day - 1) & (catalog["t"] < day))
        steps.append({"append": append, "args": [float(day)]})
    return {"args": [training, grid()], "history": training}, steps


def validate(index, value):
    shape = (len(LAT_EDGES) - 1, len(LON_EDGES) - 1)
    rates = ec.as_array(value, shape, "daily rate forecast", low=MIN_RATE)
    require(rates.sum() <= MAX_DAILY_TOTAL, f"daily rate forecast: total {rates.sum():.3g} exceeds {MAX_DAILY_TOTAL:g} events")
    return rates


def score(module, cpu_seconds=300):
    catalog = load()
    setup, steps = protocol(catalog)
    forecasts = earth_sandbox.run_online(module, "fit_quake_model", setup, steps, validate, cpu_seconds=cpu_seconds, memory_mb=3000)
    counts = target_counts(catalog)
    total = int(counts.sum())
    require(total > 0, "no target events in the test period")
    reference = smoothed_reference(setup["history"])
    ll_candidate = sum(poisson_loglik(f, c) for f, c in zip(forecasts, counts))
    ll_reference = sum(poisson_loglik(reference, c) for c in counts)
    expected = float(sum(f.sum() for f in forecasts))
    daily_counts = counts.sum(axis=(1, 2))
    daily_expected = np.array([f.sum() for f in forecasts])
    details = {"target_events": total, "forecast_days": len(forecasts), "test_period": "2016-01-01 to 2024-12-31",
               "log_likelihood": round(ll_candidate, 3), "reference_log_likelihood": round(ll_reference, 3),
               "information_gain_per_earthquake_nats": round((ll_candidate - ll_reference) / total, 5),
               "expected_target_events": round(expected, 2),
               "number_test_ratio_observed_over_expected": round(total / expected, 4),
               "days_with_targets": int((daily_counts > 0).sum()),
               "mean_expected_on_days_with_targets": round(float(daily_expected[daily_counts > 0].mean()), 4)}
    return (ll_candidate - ll_reference) / total, details
