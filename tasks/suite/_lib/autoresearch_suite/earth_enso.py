"""Monthly Nino 3.4 forecasts at 1-12 month leads from ERSSTv5 sea-surface temperatures.

The candidate fits a statistical forecast model on 1950-1999 monthly ERSSTv5 data (Nino
indices and the 2-degree tropical Pacific SST field) and is then stepped through the
held-out origins January 2000 - December 2024 in the sandboxed online protocol: at each
origin it has seen every month up to and including the origin month and nothing later, and
returns the Nino 3.4 area-mean SST (deg C) for the following twelve months. The score is
the mean over the twelve leads of the root-mean-square error. Anomalies relative to any
fixed climatology give the same errors, so no climatology is imposed on candidates.
"""
from __future__ import annotations

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

RAW_SOURCES = ("ersst.v5.sst.mnmean.nc",)
SOURCE_SPECS = ({"name": RAW_SOURCES[0], "url": "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc",
                 "pinning": "live", "license": "No constraints on data access or use (licence attribute of the file); work of the U.S. Government (NOAA NCEI, distributed by NOAA PSL)",
                 "license_url": "https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html",
                 "citation": "Huang, B. et al. (2017). Extended Reconstructed Sea Surface Temperature, Version 5 (ERSSTv5): upgrades, validations, and intercomparisons. Journal of Climate 30, 8179-8205.",
                 "retrieved": "2026-10-10"},)
DATA = "enso/ersstv5_tropical_pacific_1950_2025.json.gz"
DERIVED = {DATA: "99664f0ac1f376df7174a118d4aebb85ee097778cb9bdfaea43f90ba685fbb65"}
FIRST, LAST = (1950, 1), (2025, 12)
TRAIN_END = (1999, 12)
TEST_ORIGINS = ((2000, 1), (2024, 12))
LEADS = 12
FIELD_LAT, FIELD_LON = (-30.0, 30.0), (120.0, 290.0)
BOXES = {"nino12": ((-10.0, 0.0), (270.0, 280.0)), "nino3": ((-5.0, 5.0), (210.0, 270.0)),
         "nino34": ((-5.0, 5.0), (190.0, 240.0)), "nino4": ((-5.0, 5.0), (160.0, 210.0))}
SERIES = ("nino12", "nino3", "nino34", "nino4")


def box_mean(sst, lat, lon, box):
    """Cosine-latitude weighted mean over grid cells whose centres lie in the box (edges included)."""
    (la0, la1), (lo0, lo1) = box
    rows = (lat >= la0) & (lat <= la1)
    cols = (lon >= lo0) & (lon <= lo1)
    block = sst[:, rows][:, :, cols]
    weights = np.broadcast_to(np.cos(np.radians(lat[rows]))[:, None], block.shape[1:])
    valid = np.isfinite(block)
    return np.sum(np.where(valid, block, 0.0) * weights, axis=(1, 2)) / np.sum(valid * weights, axis=(1, 2))


def derive(paths):
    import netCDF4  # preparation-only dependency
    with netCDF4.Dataset(paths[RAW_SOURCES[0]]) as nc:
        dates = netCDF4.num2date(nc.variables["time"][:], nc.variables["time"].units)
        keys = [(d.year, d.month) for d in dates]
        start, stop = keys.index(FIRST), keys.index(LAST) + 1
        lat = np.asarray(nc.variables["lat"][:], float)
        lon = np.asarray(nc.variables["lon"][:], float)
        sst = np.ma.filled(nc.variables["sst"][start:stop].astype(float), np.nan)
    sst[(sst < -5) | (sst > 50)] = np.nan
    indices = {name: np.round(box_mean(sst, lat, lon, box), 3) for name, box in BOXES.items()}
    rows = np.where((lat >= FIELD_LAT[0]) & (lat <= FIELD_LAT[1]))[0][::-1]  # south to north
    cols = np.where((lon >= FIELD_LON[0]) & (lon <= FIELD_LON[1]))[0]
    field = np.round(sst[:, rows][:, :, cols] * 100.0)
    months = keys[start:stop]
    return {DATA: {"source": RAW_SOURCES[0], "units": "deg C; sst_centi in hundredths of deg C, null over land or ice",
                   "year": [y for y, _ in months], "month": [m for _, m in months],
                   "lat": lat[rows].tolist(), "lon": lon[cols].tolist(),
                   **{name: values.tolist() for name, values in indices.items()},
                   "sst_centi": [[[None if not np.isfinite(v) else int(v) for v in row] for row in grid] for grid in field]}}


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])
    data = {"year": np.array(raw["year"], int), "month": np.array(raw["month"], int),
            "lat": np.array(raw["lat"], float), "lon": np.array(raw["lon"], float)}
    for name in SERIES:
        data[name] = np.array(raw[name], float)
    data["sst"] = np.array([[[np.nan if v is None else v / 100.0 for v in row] for row in grid] for grid in raw["sst_centi"]], float)
    return data


def month_index(year, month):
    return (year - FIRST[0]) * 12 + (month - FIRST[1])


def rows(data, start, stop):
    """Time-indexed arrays for months [start, stop)."""
    return {key: data[key][start:stop] for key in ("year", "month", *SERIES, "sst")}


def climatology(data, end):
    """Monthly mean Nino 3.4 over the 30 training years before index ``end``."""
    clim = np.zeros(12)
    for m in range(12):
        idx = np.arange(end - 360, end)
        clim[m] = data["nino34"][idx[(data["month"][idx] - 1) == m]].mean()
    return clim


def references(data, train_end, origins):
    """Climatology, anomaly persistence and damped persistence (lag correlations fitted on training years)."""
    clim = climatology(data, train_end)
    anomaly = data["nino34"] - clim[data["month"] - 1]
    damping = np.zeros((12, LEADS))
    for m in range(12):
        starts = np.array([i for i in range(train_end - 600, train_end - LEADS) if data["month"][i] - 1 == m])
        for lead in range(1, LEADS + 1):
            x, y = anomaly[starts], anomaly[starts + lead]
            damping[m, lead - 1] = float(np.dot(x, y) / np.dot(x, x))
    out = {"climatology": [], "persistence": [], "damped_persistence": []}
    for t in origins:
        target_months = (data["month"][t] - 1 + np.arange(1, LEADS + 1)) % 12
        base = clim[target_months]
        out["climatology"].append(base)
        out["persistence"].append(base + anomaly[t])
        out["damped_persistence"].append(base + anomaly[t] * damping[data["month"][t] - 1])
    return {name: np.array(values) for name, values in out.items()}


def validate(index, value):
    return ec.as_array(value, (LEADS,), "Nino 3.4 forecast", low=10.0, high=40.0)


def protocol(data):
    """(setup, steps, origins): training data, one step per held-out origin month (appending that month only)."""
    train_end = month_index(*TRAIN_END) + 1
    first, last = month_index(*TEST_ORIGINS[0]), month_index(*TEST_ORIGINS[1])
    require(first == train_end and last + LEADS < len(data["year"]), "prepared data do not cover the protocol")
    training = {**rows(data, 0, train_end), "lat": data["lat"], "lon": data["lon"]}
    steps = [{"append": rows(data, t, t + 1), "args": []} for t in range(first, last + 1)]
    return {"args": [training], "history": rows(data, 0, train_end)}, steps, list(range(first, last + 1))


def score(module, cpu_seconds=240):
    data = load()
    setup, steps, origins = protocol(data)
    train_end = origins[0]
    forecasts = earth_sandbox.run_online(module, "fit_enso_model", setup, steps, validate, cpu_seconds=cpu_seconds, memory_mb=3000)
    candidate = np.array(forecasts)
    truth = np.array([data["nino34"][t + 1:t + 1 + LEADS] for t in origins])
    refs = references(data, train_end, origins)
    rmse = np.sqrt(np.mean((candidate - truth) ** 2, axis=0))
    clim_1991 = np.array([data["nino34"][month_index(1991, 1):month_index(2021, 1)][m::12].mean() for m in range(12)])
    target_month = np.array([(data["month"][t] - 1 + np.arange(1, LEADS + 1)) % 12 for t in origins])
    acc = [ec.correlation(truth[:, k] - clim_1991[target_month[:, k]], candidate[:, k] - clim_1991[target_month[:, k]]) for k in range(LEADS)]
    ref_rmse = {name: np.sqrt(np.mean((values - truth) ** 2, axis=0)) for name, values in refs.items()}
    details = {"unit": "deg C", "origins": len(origins), "held_out_origins": "2000-01 to 2024-12", "leads_months": list(range(1, LEADS + 1)),
               "rmse_by_lead": [round(float(x), 4) for x in rmse],
               "anomaly_correlation_by_lead": [round(float(x), 4) for x in acc],
               **{f"{name}_mean_rmse": round(float(np.mean(values)), 4) for name, values in ref_rmse.items()},
               "damped_persistence_rmse_by_lead": [round(float(x), 4) for x in ref_rmse["damped_persistence"]]}
    return float(np.mean(rmse)), details
