"""Hindcasting significant wave height on the Great Lakes from buoy meteorology.

Waves on the Great Lakes are generated locally by wind over limited fetch, so wave height is
a function of wind speed, direction (fetch) and duration (Schwab et al. 1984; JONSWAP growth
laws). Seven NDBC buoys (Lakes Superior, Michigan, Huron and Erie) provide hourly wind, gust,
pressure, air, water and dew-point temperature and wave observations. The candidate learns from
2009-2018 (all fields) and receives only the meteorological fields for 2019-2024 at all seven
buoys; it returns the significant wave height at every 2019-2024 hour with a valid wave
observation. The score is the root-mean-square error (m) over those hours.
"""
from __future__ import annotations

import datetime as dt
import gzip

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

BUOYS = {"45001": "Lake Superior (mid)", "45006": "Lake Superior (west)", "45002": "Lake Michigan (north)",
         "45007": "Lake Michigan (south)", "45003": "Lake Huron (north)", "45008": "Lake Huron (south)", "45005": "Lake Erie (west)"}
YEARS, TEST_FIRST = range(2009, 2025), 2019
URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/{0}h{1}.txt.gz"
LICENSE = {"license": "Public domain (U.S. Government work, NOAA National Data Buoy Center)", "license_url": "https://www.ndbc.noaa.gov/disclaimer.shtml",
           "citation": "NOAA National Data Buoy Center (2026). Historical standard meteorological data of Great Lakes buoys.", "retrieved": "2026-10-10"}
RAW_SOURCES = ("ndbc_station_table.txt",) + tuple(f"ndbc_{b}h{y}.txt.gz" for b in BUOYS for y in YEARS)
SOURCE_SPECS = ({"name": RAW_SOURCES[0], "url": "https://www.ndbc.noaa.gov/data/stations/station_table.txt", "pinning": "live", **LICENSE},) + tuple(
    {"name": f"ndbc_{b}h{y}.txt.gz", "url": URL.format(b, y), "pinning": "immutable yearly archive file", **LICENSE} for b in BUOYS for y in YEARS)
DATA = "waves/ndbc_great_lakes_2009_2024.json.gz"
DERIVED = {DATA: "799b734df6e7fe4f161e5796d5c709ce963c4932fd0082afb6b84ec4e6cf33cd"}
EPOCH = dt.datetime(2009, 1, 1)
MET = ("wdir", "wspd", "gst", "pres", "atmp", "wtmp", "dewp")
WAVE = ("wvht", "dpd", "apd", "mwd")
COLUMNS = {"wdir": 5, "wspd": 6, "gst": 7, "wvht": 8, "dpd": 9, "apd": 10, "mwd": 11, "pres": 12, "atmp": 13, "wtmp": 14, "dewp": 15}
MISSING = {"wdir": 999, "wspd": 99, "gst": 99, "wvht": 99, "dpd": 99, "apd": 99, "mwd": 999, "pres": 9999, "atmp": 999, "wtmp": 999, "dewp": 999}


def parse_station_table(text):
    out = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) > 6 and parts[0] in BUOYS:
            lat, ns, lon, ew = parts[6].split()[:4]
            out[parts[0]] = (float(lat) * (1 if ns == "N" else -1), float(lon) * (-1 if ew == "W" else 1))
    return out


def _value(text, missing):
    try:
        value = float(text)
    except ValueError:  # "MM" marks a missing value in some files
        return None
    return None if value >= missing else value


def parse_year(raw):
    """Hourly records keyed by hour since 2009-01-01: for each field, the valid value observed closest to the full hour
    (within 30 minutes). Since 2019 the archive holds 10-minute records with wave data on only one of them."""
    best = {}
    for line in gzip.decompress(raw).decode().splitlines():
        if line.startswith("#"):
            continue
        p = line.split()
        when = dt.datetime(int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]))
        hours = (when - EPOCH).total_seconds() / 3600.0
        hour = int(np.floor(hours + 0.5))
        offset = abs(hours - hour)
        slot = best.setdefault(hour, {})
        for key, column in COLUMNS.items():
            value = _value(p[column], MISSING[key])
            if value is not None and (key not in slot or offset < slot[key][0]):
                slot[key] = (offset, value)
    return {hour: {k: slot[k][1] if k in slot else None for k in COLUMNS} for hour, slot in best.items()}


def derive(paths):
    positions = parse_station_table(paths[RAW_SOURCES[0]].read_text(errors="replace"))
    buoys = {}
    for buoy in BUOYS:
        merged = {}
        for year in YEARS:
            merged.update(parse_year(paths[f"ndbc_{buoy}h{year}.txt.gz"].read_bytes()))
        hours = sorted(merged)
        buoys[buoy] = {"name": BUOYS[buoy], "lat": positions[buoy][0], "lon": positions[buoy][1], "hour": hours,
                       **{k: [merged[h][k] for h in hours] for k in MET + WAVE}}
    return {DATA: {"source": "NOAA NDBC historical standard meteorological data", "hour_unit": "hours since 2009-01-01T00:00Z",
                   "buoys": buoys}}


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])["buoys"]
    test_start = (dt.datetime(TEST_FIRST, 1, 1) - EPOCH).total_seconds() / 3600.0
    train, test, truth = {}, {}, {}
    for buoy, b in raw.items():
        hour = np.array(b["hour"], float)
        fields = {k: np.array([np.nan if v is None else v for v in b[k]], float) for k in MET + WAVE}
        early = hour < test_start
        meta = {"lat": b["lat"], "lon": b["lon"]}
        train[buoy] = {**meta, "hour": hour[early], **{k: v[early] for k, v in fields.items()}}
        late = ~early
        target = np.flatnonzero(np.isfinite(fields["wvht"][late]))
        test[buoy] = {**meta, "hour": hour[late], **{k: fields[k][late] for k in MET}, "target_index": target.astype(float)}
        truth[buoy] = fields["wvht"][late][target]
    return train, test, truth


def sector_regression(train, test, sectors=12, smoothing=3):
    """Reference: per buoy and 30-degree wind sector, least squares Hs = a + b U^2 with U the 3-hour mean wind speed."""
    def smooth(b):
        u = np.nan_to_num(b["wspd"])
        kernel = np.ones(smoothing) / smoothing
        return np.convolve(u, kernel)[:len(u)], np.nan_to_num(b["wdir"]) // (360 / sectors)
    out = {}
    for buoy in train:
        u, s = smooth(train[buoy])
        ok = np.isfinite(train[buoy]["wvht"]) & np.isfinite(train[buoy]["wspd"])
        coef = {}
        for k in range(sectors):
            m = ok & (s == k)
            x = np.column_stack([np.ones(m.sum()), u[m] ** 2])
            coef[k] = np.linalg.lstsq(x, train[buoy]["wvht"][m], rcond=None)[0] if m.sum() > 20 else np.array([np.nanmean(train[buoy]["wvht"]), 0.0])
        ut, st = smooth(test[buoy])
        index = test[buoy]["target_index"].astype(int)
        out[buoy] = np.array([coef[int(st[i]) % sectors][0] + coef[int(st[i]) % sectors][1] * ut[i] ** 2 for i in index])
    return out


def score(module, cpu_seconds=180):
    train, test, truth = load()
    output = earth_sandbox.call(module, "hindcast_waves", [train, test], cpu_seconds=cpu_seconds, memory_mb=3500)
    require(isinstance(output, dict), "hindcast_waves must return a dict keyed by buoy")
    reference = sector_regression(train, test)
    pred, true, ref, per_buoy = [], [], [], {}
    for buoy in test:
        require(buoy in output, f"missing predictions for buoy {buoy}")
        values = ec.as_array(output[buoy], (len(truth[buoy]),), f"buoy {buoy} wave heights", low=0.0, high=20.0)
        pred.append(values); true.append(truth[buoy]); ref.append(reference[buoy])
        per_buoy[buoy] = round(ec.rmse(truth[buoy], values), 4)
    pred, true, ref = (np.concatenate(x) for x in (pred, true, ref))
    details = {"held_out_hours": int(len(true)), "held_out_years": f"{TEST_FIRST}-{YEARS[-1]}", "rmse_by_buoy_m": per_buoy,
               "bias_m": round(float(np.mean(pred - true)), 4), "correlation": round(ec.correlation(true, pred), 4),
               "rmse_above_2m_waves": round(ec.rmse(true[true >= 2.0], pred[true >= 2.0]), 4),
               "sector_regression_reference_rmse_m": round(ec.rmse(true, ref), 4)}
    return ec.rmse(true, pred), details
