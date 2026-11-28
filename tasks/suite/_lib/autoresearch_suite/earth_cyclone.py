"""Atlantic tropical-cyclone track and intensity forecasts verified against HURDAT2.

Training storms are the 1950-2014 seasons of the NHC best track (HURDAT2, Landsea and
Franklin 2013); the 2015-2024 seasons are held out. The candidate fits a model on
the training storms and is then asked, in global chronological order, for 12-120 h
forecasts from every 6-hourly best-track time at which a held-out system is a
tropical or subtropical cyclone. Each call sees only that storm's best track up to
the forecast time. Verification follows the NHC convention: a lead counts only when
the system is still a tropical or subtropical cyclone at the verifying time.

The candidate runs in the earth_sandbox child process, which is sent the training storms
and then one truncated best track per forecast; the verifying records stay in this process.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

RAW_SOURCES = ("hurdat2-1851-2024-040425.txt",)
DATA = "cyclones/hurdat2_atlantic_1950_2024.json.gz"
DERIVED = {DATA: "5b290c62f44998d99c82b394f4847e5eb54be046bf94e474f92d90177b953db0"}
LEADS = (12, 24, 36, 48, 72, 96, 120)
TROPICAL = ("TD", "TS", "HU", "SD", "SS")
TRAIN_YEARS, TEST_YEARS = (1950, 2014), (2015, 2024)
EPOCH = dt.datetime(1950, 1, 1)


def derive(paths):
    """Parse HURDAT2 into storms from 1950 on, keeping the 6-hourly synoptic records."""
    storms, current = [], None
    for line in paths[RAW_SOURCES[0]].read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if line[:2].isalpha():
            current = {"id": parts[0], "name": parts[1], "records": []}
            storms.append(current)
            continue
        date, clock, status, lat, lon, vmax, mslp = parts[0], parts[1], parts[3], parts[4], parts[5], parts[6], parts[7]
        if clock[2:] != "00" or int(clock[:2]) % 6:
            continue  # landfall and other off-synoptic records
        latitude = float(lat[:-1]) * (1 if lat[-1] == "N" else -1)
        longitude = float(lon[:-1]) * (-1 if lon[-1] == "W" else 1)
        current["records"].append([int(date + clock[:2]), status, latitude, longitude, int(vmax), int(mslp)])
    kept = [s for s in storms if s["records"] and int(str(s["records"][0][0])[:4]) >= TRAIN_YEARS[0]]
    return {DATA: {"source": RAW_SOURCES[0], "fields": ["yyyymmddhh", "status", "lat_deg_n", "lon_deg_e", "vmax_kt", "mslp_hpa_or_-999"],
                   "storms": kept}}


def load_storms():
    storms = []
    for s in ec.load_derived(DATA, DERIVED[DATA])["storms"]:
        stamps = [str(r[0]) for r in s["records"]]
        times = [dt.datetime(int(x[:4]), int(x[4:6]), int(x[6:8]), int(x[8:10])) for x in stamps]
        storms.append({"id": s["id"], "year": times[0].year,
                       "time_hours": np.array([int((t - EPOCH).total_seconds() // 3600) for t in times]),
                       "month": np.array([t.month for t in times]), "day": np.array([t.day for t in times]),
                       "hour": np.array([t.hour for t in times]), "day_of_year": np.array([t.timetuple().tm_yday for t in times]),
                       "status": tuple(r[1] for r in s["records"]),
                       "lat": np.array([r[2] for r in s["records"]], float), "lon": np.array([r[3] for r in s["records"]], float),
                       "vmax_kt": np.array([r[4] for r in s["records"]], float),
                       "mslp_hpa": np.array([np.nan if r[5] < 0 else r[5] for r in s["records"]], float)})
    return storms


def view(storm, stop=None):
    """Candidate-facing copy of a storm (records up to and including index stop)."""
    end = len(storm["status"]) if stop is None else stop + 1
    out = {key: ec.frozen(storm[key][:end], int if key in ("time_hours", "month", "day", "hour", "day_of_year") else float)
           for key in ("time_hours", "month", "day", "hour", "day_of_year", "lat", "lon", "vmax_kt", "mslp_hpa")}
    out["status"] = tuple(storm["status"][:end])
    out["year"] = int(storm["year"])
    return out


def cases(storms):
    """(origin hour, storm index, record index, verifying record index per lead or -1), chronological."""
    rows = []
    for k, storm in enumerate(storms):
        position = {int(t): i for i, t in enumerate(storm["time_hours"])}
        for i, status in enumerate(storm["status"]):
            if status not in TROPICAL:
                continue
            verify = []
            for lead in LEADS:
                j = position.get(int(storm["time_hours"][i]) + lead)
                verify.append(j if j is not None and storm["status"][j] in TROPICAL else -1)
            if max(verify) >= 0:
                rows.append((int(storm["time_hours"][i]), k, i, tuple(verify)))
    return sorted(rows)


def split():
    storms = load_storms()
    train = [s for s in storms if TRAIN_YEARS[0] <= s["year"] <= TRAIN_YEARS[1]]
    test = [s for s in storms if TEST_YEARS[0] <= s["year"] <= TEST_YEARS[1]]
    return train, test


def predictors(storm, i):
    """Climatology-and-persistence predictors at record i (used by the reference regressions)."""
    back12, back24 = max(i - 2, 0), max(i - 4, 0)
    lat, lon, v = storm["lat"], storm["lon"], storm["vmax_kt"]
    angle = 2 * np.pi * storm["day_of_year"][i] / 365.25
    return np.array([1.0, lat[i], lon[i], lat[i] - lat[back12], lon[i] - lon[back12], lat[i] - lat[back24],
                     lon[i] - lon[back24], v[i], v[i] - v[back12], np.sin(angle), np.cos(angle)])


def fit_reference(train, kind):
    """CLIPER/SHIFOR-style linear least squares on climatology and persistence predictors."""
    rows, targets = [], []
    for _, k, i, verify in cases(train):
        storm = train[k]
        if min(verify) < 0:
            continue  # keep a homogeneous training sample verifying at every lead
        rows.append(predictors(storm, i))
        if kind == "track":
            targets.append(np.concatenate([storm["lat"][list(verify)] - storm["lat"][i], storm["lon"][list(verify)] - storm["lon"][i]]))
        else:
            targets.append(storm["vmax_kt"][list(verify)] - storm["vmax_kt"][i])
    coefficients, *_ = np.linalg.lstsq(np.array(rows), np.array(targets), rcond=None)
    return coefficients


def reference_forecast(coefficients, storm, i, kind):
    delta = predictors(storm, i) @ coefficients
    if kind == "track":
        n = len(LEADS)
        return np.column_stack([storm["lat"][i] + delta[:n], storm["lon"][i] + delta[n:]])
    return np.clip(storm["vmax_kt"][i] + delta, 10.0, None)


def persistence(storm, i, kind):
    if kind == "intensity":
        return np.full(len(LEADS), storm["vmax_kt"][i])
    j = max(i - 2, 0)  # constant 12-h motion
    steps = np.array(LEADS) / (6.0 * max(i - j, 1))
    return np.column_stack([np.clip(storm["lat"][i] + steps * (storm["lat"][i] - storm["lat"][j]), -89.0, 89.0),
                            storm["lon"][i] + steps * (storm["lon"][i] - storm["lon"][j])])


def errors(kind, storm, verify, forecast):
    out = np.full(len(LEADS), np.nan)
    for n, j in enumerate(verify):
        if j < 0:
            continue
        if kind == "track":
            out[n] = ec.haversine_km(storm["lat"][j], storm["lon"][j], forecast[n, 0], forecast[n, 1])
        else:
            out[n] = abs(forecast[n] - storm["vmax_kt"][j])
    return out


def validate(kind, output):
    if kind == "track":
        array = ec.as_array(output, (len(LEADS), 2), "track forecast")
        require(np.all(np.abs(array[:, 0]) <= 90) and np.all(np.abs(array[:, 1]) <= 360), "track forecast: latitude or longitude out of range")
        return array
    return ec.as_array(output, (len(LEADS),), "intensity forecast", low=0.0, high=300.0)


def score(module, kind, cpu_seconds=120):
    """Mean over leads of the mean forecast error (km for tracks, kt for intensity) on 2015-2024 storms."""
    require(kind in ("track", "intensity"), "unknown forecast kind")
    train, test = split()
    test_cases = cases(test)
    # one call per forecast, each with that storm's records up to the forecast time; nothing later leaves this process
    steps = [{"append": None, "args": [view(test[k], i)]} for _, k, i, _ in test_cases]
    forecasts = earth_sandbox.run_online(module, f"fit_{kind}_model", {"args": [[view(s) for s in train]], "history": None}, steps,
                                         lambda index, value: validate(kind, value), with_history=False, cpu_seconds=cpu_seconds)
    reference = fit_reference(train, kind)
    rows = {"candidate": [], "reference": [], "persistence": []}
    for (_, k, i, verify), forecast in zip(test_cases, forecasts):
        storm = test[k]
        rows["candidate"].append(errors(kind, storm, verify, forecast))
        rows["reference"].append(errors(kind, storm, verify, reference_forecast(reference, storm, i, kind)))
        rows["persistence"].append(errors(kind, storm, verify, persistence(storm, i, kind)))
    table = {name: np.array(values) for name, values in rows.items()}
    per_lead = {name: np.nanmean(values, axis=0) for name, values in table.items()}
    counts = np.sum(np.isfinite(table["candidate"]), axis=0)
    value = float(np.mean(per_lead["candidate"]))
    unit = "km" if kind == "track" else "kt"
    details = {"unit": unit, "leads_hours": list(LEADS), "verifying_cases": counts.astype(int).tolist(),
               "forecast_calls": int(len(table["candidate"])), "held_out_seasons": f"{TEST_YEARS[0]}-{TEST_YEARS[1]}",
               "mean_error_by_lead": [round(float(x), 3) for x in per_lead["candidate"]],
               "reference_regression_mean_error": round(float(np.mean(per_lead["reference"])), 3),
               "reference_regression_by_lead": [round(float(x), 3) for x in per_lead["reference"]],
               "persistence_mean_error": round(float(np.mean(per_lead["persistence"])), 3),
               "skill_vs_reference_percent": [round(float(100 * (1 - c / r)), 2) for c, r in zip(per_lead["candidate"], per_lead["reference"])]}
    return value, details
