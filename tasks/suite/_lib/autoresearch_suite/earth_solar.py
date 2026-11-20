"""Separation of 1-minute global solar irradiance into its diffuse component (SURFRAD).

Most solar-resource and building-energy applications measure or model only global horizontal
irradiance (GHI), yet need its diffuse (DHI) and beam parts; "separation models" estimate the
diffuse fraction from GHI-derived predictors (Erbs et al. 1982; Ridley et al. 2010; Yang 2022).
NOAA SURFRAD stations measure GHI, direct normal and diffuse irradiance every minute. The
candidate learns from four stations (Bondville, Desert Rock, Fort Peck, Goodwin Creek) and
estimates DHI at three other stations (Penn State, Sioux Falls, Table Mountain) from their GHI,
solar geometry and meteorology on 48 sampled days of 2022-2023. The score is the RMSE (W m-2) of
DHI over quality-controlled daytime minutes (solar zenith < 85 deg, GHI >= 10 W m-2, all three
components flagged good and closing within 8 percent or 20 W m-2).
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

STATIONS = {"bon": "Bondville_IL", "dra": "Desert_Rock_NV", "fpk": "Fort_Peck_MT", "gwn": "Goodwin_Creek_MS",
            "psu": "Penn_State_PA", "sxf": "Sioux_Falls_SD", "tbl": "Boulder_CO"}
TRAIN, TEST = ("bon", "dra", "fpk", "gwn"), ("psu", "sxf", "tbl")
YEARS, DAYS = (2022, 2023), tuple(range(8, 366, 15))
URL = "https://gml.noaa.gov/aftp/data/radiation/surfrad/{0}/{1}/{2}{3:02d}{4:03d}.dat"
LICENSE = {"pinning": "static archive file", "license": "Public domain (U.S. Government work, NOAA Global Monitoring Laboratory SURFRAD network)",
           "license_url": "https://gml.noaa.gov/grad/surfrad/", "citation": "Augustine, J. A. et al. (2000). SURFRAD - a national surface radiation budget network for atmospheric research. Bulletin of the American Meteorological Society 81, 2341-2357.",
           "retrieved": "2026-10-11"}
FILES = [(s, y, d) for s in STATIONS for y in YEARS for d in DAYS]
RAW_SOURCES = tuple(f"surfrad_{s}{y % 100:02d}{d:03d}.dat" for s, y, d in FILES)
SOURCE_SPECS = tuple({"name": n, "url": URL.format(STATIONS[s], y, s, y % 100, d), **LICENSE} for n, (s, y, d) in zip(RAW_SOURCES, FILES))
DATA = "solar/surfrad_1min_7_stations_48_days.json.gz"
DERIVED = {DATA: "16cf4d5e716eb4d459f878350812b67a46fc87bc39e0aa2a4196c8aee113918e"}
SOLAR_CONSTANT = 1361.0
COLUMNS = {"zen": 7, "ghi": 8, "dni": 12, "dhi": 14, "temp": 38, "rh": 40, "pres": 46}


def parse(text):
    """(latitude, longitude, elevation_m, rows) of one daily file; rows carry NaN where the QC flag is not 0."""
    lines = text.splitlines()
    lat, lon, elevation = (float(v) for v in lines[1].split()[:3])
    rows = []
    for line in lines[2:]:
        p = line.split()
        if len(p) < 48:
            continue
        row = {"minute": int(p[4]) * 60 + int(p[5]), "doy": int(p[1])}
        for key, column in COLUMNS.items():
            value = float(p[column])
            row[key] = value if key == "zen" or int(p[column + 1]) == 0 else float("nan")
        rows.append(row)
    return lat, lon, elevation, rows


def derive(paths):
    stations = {}
    for name, (s, y, d) in zip(RAW_SOURCES, FILES):
        lat, lon, elevation, rows = parse(paths[name].read_text(errors="replace"))
        st = stations.setdefault(s, {"lat": lat, "lon": lon, "elevation_m": elevation, "rows": []})
        day = (dt.date(y, 1, 1) - dt.date(2022, 1, 1)).days + d - 1
        st["rows"] += [(day, r["minute"], r["zen"], r["ghi"], r["dni"], r["dhi"], r["temp"], r["rh"], r["pres"]) for r in rows if r["zen"] < 90.0]
    out = {}
    for s, st in stations.items():
        cols = list(zip(*st["rows"]))
        out[s] = {"lat": st["lat"], "lon": st["lon"], "elevation_m": st["elevation_m"],
                  **{k: [None if v != v else round(v, 2) for v in c] for k, c in zip(("day", "minute", "zen", "ghi", "dni", "dhi", "temp", "rh", "pres"), cols)}}
    return {DATA: {"source": "NOAA SURFRAD 1-minute data, days 8, 23, ..., 353 of 2022 and 2023", "day_unit": "days since 2022-01-01",
                   "stations": out}}


def extraterrestrial_horizontal(day, zen):
    doy = (np.asarray(day) % 365) + 1
    return SOLAR_CONSTANT * (1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)) * np.maximum(np.cos(np.radians(zen)), 0.0)


def targets(s):
    ghi, dni, dhi, zen = s["ghi"], s["dni"], s["dhi"], s["zen"]
    closure = np.abs(ghi - (dni * np.cos(np.radians(zen)) + dhi))
    with np.errstate(invalid="ignore"):
        return np.flatnonzero((zen < 85.0) & (ghi >= 10.0) & np.isfinite(dni) & np.isfinite(dhi)
                              & (closure <= np.maximum(20.0, 0.08 * ghi)) & (dhi <= ghi + 10.0))


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])["stations"]
    train, test, truth = {}, {}, {}
    for s, r in raw.items():
        arrays = {k: np.array([np.nan if v is None else v for v in r[k]], float) for k in ("day", "minute", "zen", "ghi", "dni", "dhi", "temp", "rh", "pres")}
        station = {"lat": r["lat"], "lon": r["lon"], "elevation_m": r["elevation_m"], **arrays,
                   "e0_horizontal": extraterrestrial_horizontal(arrays["day"], arrays["zen"])}
        index = targets(station)
        station["target_index"] = index.astype(float)
        if s in TRAIN:
            train[s] = station
        else:
            truth[s] = station.pop("dhi")[index]
            station.pop("dni")
            test[s] = station
    return train, test, truth


def clearness(s, index):
    return np.clip(s["ghi"][index] / np.maximum(s["e0_horizontal"][index], 1.0), 0.0, 1.2)


def binned_reference(train, test):
    """Median diffuse fraction in 0.05-wide clearness-index bins of the training stations."""
    kt = np.concatenate([clearness(s, s["target_index"].astype(int)) for s in train.values()])
    kd = np.concatenate([s["dhi"][s["target_index"].astype(int)] / s["ghi"][s["target_index"].astype(int)] for s in train.values()])
    edges = np.arange(0.0, 1.25, 0.05)
    medians = np.array([np.median(kd[(kt >= a) & (kt < b)]) if np.any((kt >= a) & (kt < b)) else 1.0 for a, b in zip(edges[:-1], edges[1:])])
    out = {}
    for name, s in test.items():
        i = s["target_index"].astype(int)
        out[name] = medians[np.clip(np.digitize(clearness(s, i), edges) - 1, 0, len(medians) - 1)] * s["ghi"][i]
    return out


def score(module, cpu_seconds=180):
    train, test, truth = load()
    output = earth_sandbox.call(module, "estimate_diffuse", [train, test], cpu_seconds=cpu_seconds, memory_mb=3500)
    require(isinstance(output, dict), "estimate_diffuse must return a dict keyed by station")
    reference = binned_reference(train, test)
    pred, true, ref, per_station = [], [], [], {}
    for s in TEST:
        i = test[s]["target_index"].astype(int)
        require(s in output, f"missing predictions for station {s}")
        values = ec.as_array(output[s], (len(i),), f"station {s} diffuse irradiance", low=0.0, high=1500.0)
        pred.append(values); true.append(truth[s]); ref.append(reference[s])
        per_station[s] = round(ec.rmse(truth[s], values), 3)
    pred, true, ref = (np.concatenate(x) for x in (pred, true, ref))
    details = {"unit": "W m-2", "test_minutes": int(len(true)), "rmse_by_station": per_station,
               "nrmse_percent": round(100 * ec.rmse(true, pred) / float(np.mean(true)), 3), "mbe": round(float(np.mean(pred - true)), 3),
               "binned_kd_reference_rmse": round(ec.rmse(true, ref), 3)}
    return ec.rmse(true, pred), details
