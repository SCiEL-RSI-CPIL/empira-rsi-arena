"""Forecasts of monthly atmospheric CO2 at Mauna Loa 1-12 months ahead.

The rise of atmospheric CO2 varies from year to year with the terrestrial carbon sink, which
responds strongly to ENSO; the Met Office issues annual forecasts of the Mauna Loa CO2 rise
from emissions and ENSO forecasts (Betts et al. 2016, 2018). The candidate fits a model on the
monthly record up to 1999 (NOAA GML/SIO monthly means and a relative Nino 3.4 index from
ERSSTv5) and is stepped through the origins January 2000 - December 2024 in the sandboxed online
protocol, forecasting the monthly mean for the next 12 months from data up to the origin month.
The score is the mean over leads of the root-mean-square error (ppm), verified on months that
were measured rather than interpolated.
"""
from __future__ import annotations

import numpy as np

from . import earth_common as ec
from . import earth_enso
from . import earth_gmst
from . import earth_sandbox
from .runtime import require

RAW_SOURCES = ("co2_mm_mlo.txt", earth_enso.RAW_SOURCES[0])
SOURCE_SPECS = ({"name": RAW_SOURCES[0], "url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt", "pinning": "live",
                 # the locked file (created 2026-10-09) as captured by the Internet Archive; the live file is rewritten monthly
                 "archive_url": "https://web.archive.org/web/20261021085831id_/https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
                 "license": "Freely available to the public (NOAA GML data-use statement in the file; U.S. Government work); March 1958 - April 1974 values from the Scripps Institution of Oceanography with citation requested",
                 "license_url": "https://gml.noaa.gov/ccgg/trends/data.html",
                 "citation": "Lan, X., Tans, P. and Thoning, K. W.: Trends in globally-averaged CO2 determined from NOAA Global Monitoring Laboratory measurements; Keeling, C. D. et al. (1976) Atmospheric carbon dioxide variations at Mauna Loa Observatory, Hawaii. Tellus 28, 538-551.",
                 "retrieved": "2026-10-10"},)
DATA = "co2/mlo_monthly_1958_2025_relative_nino34.json.gz"
DERIVED = {DATA: "b682c064db322bf1ddefa351b38d6ebfdfb799b934f3460fe06067cbf301b7f8"}
LEADS = 12
FIRST, LAST = (1958, 3), (2025, 12)
TEST_ORIGINS = ((2000, 1), (2024, 12))


def derive(paths):
    rows = []
    for line in paths[RAW_SOURCES[0]].read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        year, month = int(p[0]), int(p[1])
        if FIRST <= (year, month) <= LAST:
            rows.append((year, month, float(p[3]), int(p[5])))
    year, month, relative = earth_gmst.relative_nino34(paths[RAW_SOURCES[1]])
    enso = {(int(y), int(m)): float(r) for y, m, r in zip(year, month, relative)}
    # The index is an input only up to the last origin. Later months are left empty: ERSSTv5 revises its most recent
    # years with every monthly update, and storing them would change this file without changing any forecast or score.
    index = [round(enso[(r[0], r[1])], 4) if (r[0], r[1]) <= TEST_ORIGINS[1] else None for r in rows]
    return {DATA: {"source": "NOAA GML Mauna Loa monthly mean CO2 (co2_mm_mlo.txt) and ERSSTv5 relative Nino 3.4",
                   "year": [r[0] for r in rows], "month": [r[1] for r in rows], "co2_ppm": [r[2] for r in rows],
                   "measured_days": [r[3] for r in rows], "relative_nino34": index}}


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])
    return {k: np.array(raw[k], float) for k in ("year", "month", "co2_ppm", "measured_days", "relative_nino34")}


def index_of(data, year, month):
    return int(np.flatnonzero((data["year"] == year) & (data["month"] == month))[0])


def rows(data, start, stop):
    return {k: data[k][start:stop] for k in data}


def protocol(data):
    first, last = index_of(data, *TEST_ORIGINS[0]), index_of(data, *TEST_ORIGINS[1])
    require(last + LEADS < len(data["year"]), "prepared data end before the last verifying month")
    steps = [{"append": rows(data, t, t + 1), "args": []} for t in range(first, last + 1)]
    return {"args": [rows(data, 0, first)], "history": rows(data, 0, first)}, steps, list(range(first, last + 1))


def validate(index, value):
    return ec.as_array(value, (LEADS,), "CO2 forecast", low=250.0, high=600.0)


def reference(data, origins):
    """Seasonal persistence plus the last 12-month increase: x(t+L) = x(t+L-12) + (x(t) - x(t-12))."""
    co2 = data["co2_ppm"]
    out = []
    for t in origins:
        growth = co2[t] - co2[t - 12]
        out.append([co2[t + lead - 12] + growth if lead <= 12 else np.nan for lead in range(1, LEADS + 1)])
    return np.array(out)


def score(module, cpu_seconds=120):
    data = load()
    setup, steps, origins = protocol(data)
    forecasts = np.array(earth_sandbox.run_online(module, "fit_co2_model", setup, steps, validate, cpu_seconds=cpu_seconds))
    truth = np.array([data["co2_ppm"][t + 1:t + 1 + LEADS] for t in origins])
    measured = np.array([data["measured_days"][t + 1:t + 1 + LEADS] > 0 for t in origins])
    ref = reference(data, origins)
    err = np.where(measured, forecasts - truth, np.nan)
    ref_err = np.where(measured, ref - truth, np.nan)
    rmse = np.sqrt(np.nanmean(err ** 2, axis=0))
    details = {"unit": "ppm", "origins": len(origins), "held_out_origins": "2000-01 to 2024-12",
               "verified_months_per_lead": np.sum(measured, axis=0).astype(int).tolist(),
               "rmse_by_lead": [round(float(x), 4) for x in rmse],
               "seasonal_persistence_reference_mean_rmse": round(float(np.mean(np.sqrt(np.nanmean(ref_err ** 2, axis=0)))), 4)}
    return float(np.mean(rmse)), details
