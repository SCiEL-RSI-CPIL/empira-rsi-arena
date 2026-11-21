"""Daily Arctic sea-ice drift from geostrophic winds, verified against drifting buoys.

Thorndike and Colony (1982) showed that on daily time scales most of the variance of Arctic
sea-ice velocity is explained by the geostrophic wind. Daily buoy velocities come from the
00 UTC positions of the IABP "C" product (Arctic buoys north of 70 N); the geostrophic wind at
each buoy is computed from NCEP/NCAR reanalysis daily-mean sea-level pressure by a weighted
local plane fit (grid points within 500 km). The candidate fits a model on 2000-2014 and is then
stepped through every day of 2015-2024 in the sandboxed online protocol: at the start of each day
it knows all past buoy positions and velocities and that day's geostrophic wind, and returns the
velocity of every active buoy over the day. The score is the root-mean-square vector error (cm/s).
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

YEARS, TEST_FIRST = range(2000, 2025), 2015
EPOCH = dt.date(2000, 1, 1)
IABP = {"license": "not stated (International Arctic Buoy Programme data products; not redistributed)", "license_url": "https://iabp.apl.uw.edu/data.html",
        "citation": "International Arctic Buoy Programme, Polar Science Center, University of Washington: product C, interpolated buoy positions.",
        "retrieved": "2026-10-11", "pinning": "live"}
NCEP = {"license": "Public domain (U.S. Government work; NCEP/NCAR Reanalysis 1 distributed by NOAA PSL)", "license_url": "https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html",
        "citation": "Kalnay, E. et al. (1996). The NCEP/NCAR 40-year reanalysis project. Bulletin of the American Meteorological Society 77, 437-471.",
        "retrieved": "2026-10-11", "pinning": "static archive file"}
RAW_SOURCES = tuple(f"iabp_C{y}" for y in YEARS) + tuple(f"ncep_slp_daily_{y}.nc" for y in YEARS)
SOURCE_SPECS = tuple({"name": f"iabp_C{y}", "url": f"https://iabp.apl.uw.edu/Data_Products/C/C{y}", **IABP} for y in YEARS) + tuple(
    {"name": f"ncep_slp_daily_{y}.nc", "url": f"https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/slp.{y}.nc", **NCEP} for y in YEARS)
DATA = "icedrift/iabp_buoys_ncep_geostrophic_2000_2024.json.gz"
DERIVED = {DATA: "e316c23e85d09bf9d974637c4ab4416d37e0018bfbca8bbec6f66f1b5924ad24"}
R_EARTH, OMEGA, RHO_AIR = 6.371e6, 7.2921e-5, 1.3
FIELDS = ("day", "buoy", "lat", "lon", "slp_hpa", "ug", "vg", "u", "v")
INPUTS = ("day", "buoy", "lat", "lon", "slp_hpa", "ug", "vg")


def positions(text):
    """{(buoy, day): (lat, lon)} from 00 UTC rows of an IABP C file."""
    out = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) < 7 or int(float(p[3])) != 0:
            continue
        try:
            day = (dt.date(int(p[0]), int(p[1]), int(p[2])) - EPOCH).days
        except ValueError:  # a few 2012-2014 rows carry day 0
            continue
        out[(int(float(p[4])), day)] = (float(p[5]), (float(p[6]) + 180.0) % 360.0 - 180.0)
    return out


def local_xy(lat0, lon0, lat, lon):
    """Local tangent-plane coordinates (m, east and north) of points relative to (lat0, lon0)."""
    p0, p, dl = np.radians(lat0), np.radians(lat), np.radians(lon - lon0)
    x = R_EARTH * np.cos(p) * np.sin(dl)
    y = R_EARTH * (np.sin(p) * np.cos(p0) - np.cos(p) * np.sin(p0) * np.cos(dl))
    return x, y


def geostrophic(field, grid_lat, grid_lon, lat0, lon0):
    """(slp hPa, ug, vg m/s) at a point from a weighted plane fit to grid pressures (Pa) within 500 km."""
    x, y = local_xy(lat0, lon0, grid_lat, grid_lon)
    distance = np.hypot(x, y)
    near = (distance < 5e5) & (np.cos(np.radians(grid_lon - lon0)) * np.cos(np.radians(grid_lat)) > -0.5)
    w = np.exp(-0.5 * (distance[near] / 2.5e5) ** 2)
    a = np.column_stack([np.ones(near.sum()), x[near], y[near]]) * np.sqrt(w)[:, None]
    coef = np.linalg.lstsq(a, field[near] * np.sqrt(w), rcond=None)[0]
    f = 2 * OMEGA * np.sin(np.radians(lat0))
    return coef[0] / 100.0, -coef[2] / (RHO_AIR * f), coef[1] / (RHO_AIR * f)


def derive(paths):
    import netCDF4  # preparation-only dependency
    pos = {}
    for y in YEARS:
        pos.update(positions(paths[f"iabp_C{y}"].read_text(errors="replace")))
    by_day = {}
    for (buoy, day), value in pos.items():
        by_day.setdefault(day, []).append((buoy, value))
    rows = []
    for y in YEARS:
        with netCDF4.Dataset(paths[f"ncep_slp_daily_{y}.nc"]) as nc:
            lat = np.asarray(nc.variables["lat"][:], float)
            lon = np.asarray(nc.variables["lon"][:], float)
            times = netCDF4.num2date(nc.variables["time"][:], nc.variables["time"].units)
            cap = lat >= 55.0
            slp = np.ma.filled(nc.variables["slp"][:, cap, :].astype(float), np.nan)
        glat, glon = np.meshgrid(lat[cap], lon, indexing="ij")
        glat, glon = glat.ravel(), (glon.ravel() + 180.0) % 360.0 - 180.0
        for k, when in enumerate(times):
            day = (dt.date(when.year, when.month, when.day) - EPOCH).days
            field = slp[k].ravel()
            for buoy, (la, lo) in sorted(by_day.get(day, [])):
                nxt = pos.get((buoy, day + 1))
                if nxt is None or not (70.0 <= la <= 90.0):
                    continue
                dx, dy = local_xy(la, lo, nxt[0], nxt[1])
                u, v = dx / 864.0, dy / 864.0  # m per day -> cm/s
                if np.hypot(u, v) > 100.0:
                    continue
                p, ug, vg = geostrophic(field, glat, glon, la, lo)
                rows.append((day, buoy, round(la, 4), round(lo, 4), round(p, 2), round(ug, 3), round(vg, 3), round(u, 3), round(v, 3)))
    rows.sort()
    return {DATA: {"source": "IABP product C (00 UTC positions) and NCEP/NCAR Reanalysis 1 daily mean SLP", "day_unit": "days since 2000-01-01",
                   "velocity_unit": "cm/s (displacement from 00 UTC of the day to 00 UTC of the next day)", "wind_unit": "m/s",
                   **{f: [r[i] for r in rows] for i, f in enumerate(FIELDS)}}}


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])
    out = {f: np.array(raw[f], float) for f in FIELDS}
    out["buoy"] = np.array(raw["buoy"], np.int64)
    return out


def protocol(data):
    first = (dt.date(TEST_FIRST, 1, 1) - EPOCH).days
    train = data["day"] < first
    history = {f: data[f][train] for f in FIELDS}
    days = np.unique(data["day"][~train])
    steps, targets = [], []
    for d in days:
        today = np.flatnonzero(data["day"] == d)
        yesterday = np.flatnonzero(data["day"] == d - 1) if d > first else np.array([], int)
        steps.append({"append": {f: data[f][yesterday] for f in FIELDS}, "args": [{f: data[f][today] for f in INPUTS}]})
        targets.append(today)
    return {"args": [history], "history": history}, steps, targets


def thorndike_colony(history, data, targets):
    """Reference: complex least squares u + iv = A (ug + i vg) + C fitted on the training days."""
    w = history["u"] + 1j * history["v"]
    g = history["ug"] + 1j * history["vg"]
    a, c = np.linalg.lstsq(np.column_stack([g, np.ones_like(g)]), w, rcond=None)[0]
    return [a * (data["ug"][t] + 1j * data["vg"][t]) + c for t in targets]


def score(module, cpu_seconds=240):
    data = load()
    setup, steps, targets = protocol(data)
    sizes = [len(t) for t in targets]
    validate = lambda i, value: ec.as_array(value, (sizes[i], 2), "ice velocity forecast", low=-200.0, high=200.0)  # noqa: E731
    forecasts = earth_sandbox.run_online(module, "fit_drift_model", setup, steps, validate, cpu_seconds=cpu_seconds, memory_mb=3000)
    truth = np.concatenate([np.column_stack([data["u"][t], data["v"][t]]) for t in targets])
    pred = np.concatenate(forecasts)
    ref = np.concatenate(thorndike_colony(setup["history"], data, targets))
    vector_rmse = lambda e: float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))  # noqa: E731
    ref_err = np.column_stack([ref.real, ref.imag]) - truth
    explained = 1.0 - np.sum((pred - truth) ** 2) / np.sum((truth - truth.mean(axis=0)) ** 2)
    details = {"unit": "cm/s", "buoy_days": int(len(truth)), "forecast_days": len(steps), "held_out_years": f"{TEST_FIRST}-{YEARS[-1]}",
               "explained_variance": round(float(explained), 4), "zero_drift_rmse": round(vector_rmse(truth), 4),
               "thorndike_colony_reference_rmse": round(vector_rmse(ref_err), 4)}
    return vector_rmse(pred - truth), details
