"""Forecasts of Atlantic tropical-cyclone wind radii (storm size) verified against HURDAT2.

HURDAT2 holds the 34-, 50- and 64-kt wind radii in four quadrants since 2004. The candidate
fits a model on the 2004-2014 storms and is then asked, in chronological order, for the 34-kt
radii in the NE, SE, SW and NW quadrants 12, 24, 36, 48 and 72 h after every synoptic time of
a 2015-2024 storm at which the radii are analysed. As in the statistical-dynamical radii
schemes that run on a track and intensity forecast (Knaff et al. 2007), the best-track position,
maximum wind and pressure at the verifying times are given (a perfect-prognosis setting), so the
task isolates storm-size prediction. The score is the mean over leads of the mean absolute
radius error (n mi) at leads where the storm is still tropical or subtropical with radii.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from . import earth_common as ec
from . import earth_cyclone
from . import earth_sandbox
from .runtime import require

RAW_SOURCES = earth_cyclone.RAW_SOURCES
DATA = "cyclones/hurdat2_atlantic_radii_2004_2024.json.gz"
DERIVED = {DATA: "ce4c5750cfbdf6267c0ae8fef5e63e363957dc89e865b459ca1a4daa90f9d1f5"}
LEADS = (12, 24, 36, 48, 72)
TRAIN_YEARS, TEST_YEARS = (2004, 2014), (2015, 2024)
WITH_RADII = ("TS", "HU", "SS")
EPOCH = dt.datetime(2004, 1, 1)


def derive(paths):
    storms, current = [], None
    for line in paths[RAW_SOURCES[0]].read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if line[:2].isalpha():
            current = {"id": parts[0], "records": []}
            storms.append(current)
            continue
        if parts[1][2:] != "00" or int(parts[1][:2]) % 6 or int(parts[0][:4]) < TRAIN_YEARS[0]:
            continue
        lat = float(parts[4][:-1]) * (1 if parts[4][-1] == "N" else -1)
        lon = float(parts[5][:-1]) * (-1 if parts[5][-1] == "W" else 1)
        radii = [int(v) for v in parts[8:20]]
        current["records"].append([int(parts[0] + parts[1][:2]), parts[3], lat, lon, int(parts[6]), int(parts[7]), *radii])
    kept = [s for s in storms if s["records"]]
    return {DATA: {"source": RAW_SOURCES[0], "fields": ["yyyymmddhh", "status", "lat", "lon", "vmax_kt", "mslp_hpa_or_-999",
                                                        "r34_ne", "r34_se", "r34_sw", "r34_nw", "r50_ne", "r50_se", "r50_sw", "r50_nw",
                                                        "r64_ne", "r64_se", "r64_sw", "r64_nw"], "storms": kept}}


def load():
    storms = []
    for s in ec.load_derived(DATA, DERIVED[DATA])["storms"]:
        rec = s["records"]
        times = [dt.datetime(int(str(r[0])[:4]), int(str(r[0])[4:6]), int(str(r[0])[6:8]), int(str(r[0])[8:10])) for r in rec]
        radii = np.array([r[6:18] for r in rec], float)
        radii[radii < 0] = np.nan
        storms.append({"id": s["id"], "year": times[0].year, "status": [r[1] for r in rec],
                       "time_hours": np.array([(t - EPOCH).total_seconds() / 3600.0 for t in times]),
                       "lat": np.array([r[2] for r in rec], float), "lon": np.array([r[3] for r in rec], float),
                       "vmax_kt": np.array([r[4] for r in rec], float),
                       "mslp_hpa": np.array([np.nan if r[5] < 0 else r[5] for r in rec], float),
                       "r34": radii[:, 0:4], "r50": radii[:, 4:8], "r64": radii[:, 8:12]})
    return storms


def has_radii(storm, i):
    return storm["status"][i] in WITH_RADII and storm["vmax_kt"][i] >= 34 and np.all(np.isfinite(storm["r34"][i]))


def view(storm, stop=None):
    end = len(storm["status"]) if stop is None else stop + 1
    out = {k: storm[k][:end] for k in ("time_hours", "lat", "lon", "vmax_kt", "mslp_hpa", "r34", "r50", "r64")}
    out["status"] = list(storm["status"][:end])
    out["year"] = storm["year"]
    return out


def cases(storms):
    """(origin hour, storm index, record index, verifying record index per lead or -1), chronological."""
    rows = []
    for k, storm in enumerate(storms):
        position = {int(t): i for i, t in enumerate(storm["time_hours"])}
        for i in range(len(storm["status"])):
            if not has_radii(storm, i):
                continue
            verify = [position.get(int(storm["time_hours"][i]) + lead, -1) for lead in LEADS]
            verify = [j if j >= 0 and has_radii(storm, j) else -1 for j in verify]
            if max(verify) >= 0:
                rows.append((float(storm["time_hours"][i]), k, i, tuple(verify)))
    return sorted(rows)


def future(storm, verify):
    """Best-track position, intensity and pressure at the verifying times (NaN where a lead does not verify)."""
    pick = lambda key: np.array([storm[key][j] if j >= 0 else np.nan for j in verify])  # noqa: E731
    return {"lead_hours": np.array(LEADS, float), "verifies": np.array([j >= 0 for j in verify], float),
            "lat": pick("lat"), "lon": pick("lon"), "vmax_kt": pick("vmax_kt"), "mslp_hpa": pick("mslp_hpa")}


def split():
    storms = load()
    return ([s for s in storms if TRAIN_YEARS[0] <= s["year"] <= TRAIN_YEARS[1]],
            [s for s in storms if TEST_YEARS[0] <= s["year"] <= TEST_YEARS[1]])


def validate(index, value):
    return ec.as_array(value, (len(LEADS), 4), "34-kt radii forecast", low=0.0, high=1000.0)


def regression_reference(train, test_cases, test):
    """Least squares per lead: R34(t+L) on R34(t), the intensity change, the verifying intensity and latitude."""
    def row(storm, i, j):
        return [1.0, storm["vmax_kt"][j] - storm["vmax_kt"][i], storm["vmax_kt"][j], abs(storm["lat"][j]), storm["r34"][i].mean()]
    out = np.full((len(test_cases), len(LEADS), 4), np.nan)
    for n in range(len(LEADS)):
        x, y = [], []
        for _, k, i, verify in cases(train):
            if verify[n] >= 0:
                for q in range(4):
                    x.append(row(train[k], i, verify[n]) + [train[k]["r34"][i][q]]); y.append(train[k]["r34"][verify[n]][q])
        beta = np.linalg.lstsq(np.array(x), np.array(y), rcond=None)[0]
        for c, (_, k, i, verify) in enumerate(test_cases):
            if verify[n] >= 0:
                out[c, n] = [np.dot(row(test[k], i, verify[n]) + [test[k]["r34"][i][q]], beta) for q in range(4)]
    return out


def score(module, cpu_seconds=120):
    train, test = split()
    test_cases = cases(test)
    steps = [{"append": None, "args": [view(test[k], i), future(test[k], verify)]} for _, k, i, verify in test_cases]
    forecasts = earth_sandbox.run_online(module, "fit_radii_model", {"args": [[view(s) for s in train]], "history": None},
                                         steps, validate, with_history=False, cpu_seconds=cpu_seconds)
    reference = regression_reference(train, test_cases, test)
    err = np.full((len(test_cases), len(LEADS)), np.nan)
    ref_err, persistence = err.copy(), err.copy()
    for c, (_, k, i, verify) in enumerate(test_cases):
        for n, j in enumerate(verify):
            if j >= 0:
                truth = test[k]["r34"][j]
                err[c, n] = np.mean(np.abs(forecasts[c][n] - truth))
                ref_err[c, n] = np.mean(np.abs(reference[c, n] - truth))
                persistence[c, n] = np.mean(np.abs(test[k]["r34"][i] - truth))
    by_lead = np.nanmean(err, axis=0)
    require(np.all(np.isfinite(by_lead)), "a lead has no verifying cases")
    details = {"unit": "n mi", "forecast_calls": len(test_cases), "leads_hours": list(LEADS),
               "verifying_cases": np.sum(np.isfinite(err), axis=0).astype(int).tolist(),
               "mae_by_lead": [round(float(x), 3) for x in by_lead],
               "persistence_mean_error": round(float(np.mean(np.nanmean(persistence, axis=0))), 3),
               "regression_reference_mean_error": round(float(np.mean(np.nanmean(ref_err, axis=0))), 3)}
    return float(np.mean(by_lead)), details
