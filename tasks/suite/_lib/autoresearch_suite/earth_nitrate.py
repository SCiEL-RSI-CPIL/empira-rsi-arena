"""River nitrate concentrations and annual loads from sparse samples, verified against continuous sensors.

Eight U.S. rivers have years of daily nitrate (plus nitrite, mg/L as N) from in-situ optical
sensors (USGS parameter 99133) and daily discharge. The evaluator emulates a routine sampling
programme on each record (one sample on a random day of every month plus four high-flow samples
per water year, fixed seed) and gives the candidate those samples and the full daily discharge.
The candidate estimates the daily concentration on every day; the score is the mean absolute
relative error of monthly nitrate loads against loads computed with the sensor concentrations
(calendar months with at least 25 sensor days), the kind of flux estimate that weighted
regressions on time, discharge and season (WRTDS, Hirsch et al. 2010) are used to produce.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

SITES = {  # site number: (name, latitude, longitude, drainage area in square miles)
    "03374100": ("White River at Hazleton, IN", 38.4897748, -87.5500132, 11305.0),
    "05465500": ("Iowa River at Wapello, IA", 41.1780861, -91.1820917, 12500.0),
    "05482300": ("North Raccoon River near Sac City, IA", 42.35455556, -94.9910278, 700.0),
    "05484500": ("Raccoon River at Van Meter, IA", 41.53402778, -93.9497778, 3441.0),
    "05515500": ("Kankakee River at Davis, IN", 41.38963889, -86.7061667, 542.0),
    "05595000": ("Kaskaskia River at New Athens, IL", 38.3197222, -89.8883333, 5189.0),
    "06934500": ("Missouri River at Hermann, MO", 38.70980556, -91.4385, 522500.0),
    "07374000": ("Mississippi River at Baton Rouge, LA", 30.44566667, -91.1915556, 1125810.0),
}
URL = "https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={0}&parameterCd=00060,99133&statCd=00003&startDT=2008-01-01&endDT=2025-09-30"
RAW_SOURCES = tuple(f"nwis_dv_{site}_00060_99133.rdb" for site in SITES)
SOURCE_SPECS = tuple({"name": name, "url": URL.format(site), "pinning": "live",
                      "license": "Public domain (U.S. Geological Survey National Water Information System)",
                      "license_url": "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
                      "citation": "U.S. Geological Survey (2026). National Water Information System data (USGS Water Data for the Nation), daily values of discharge (00060) and nitrate plus nitrite (99133).",
                      "retrieved": "2026-10-10"} for name, site in zip(RAW_SOURCES, SITES))
DATA = "nitrate/nwis_sensor_nitrate_8_sites.json.gz"
DERIVED = {DATA: "e4236451d52fcd4876d9b76e26b410ed056152e54f650a9a90a209d008817294"}
EPOCH = dt.date(1970, 1, 1)
# Last day used at each site. NWIS approves provisional records in blocks, long after the measurements. Days approved
# after the lock was recorded (2026-10-10) would lengthen a record, move its high-flow threshold and redraw its
# samples, so every record ends where its approved discharge ended on that date.
RECORD_END = {site: "2025-09-30" for site in SITES} | {"07374000": "2024-04-14"}
CFS = 0.028316846592  # m3/s per cubic foot per second
KG_PER_DAY = 86.4  # kg/day for 1 mg/L at 1 m3/s
MIN_MONTH_DAYS, MIN_YEAR_DAYS = 25, 300


def parse(path):
    with open(path) as stream:
        lines = [line.rstrip("\n").split("\t") for line in stream if not line.startswith("#")]
    header = lines[0]
    q_col = next(i for i, h in enumerate(header) if h.endswith("_00060_00003"))
    n_col = next(i for i, h in enumerate(header) if h.endswith("_99133_00003"))
    out = []
    for row in lines[2:]:
        day = (dt.date.fromisoformat(row[2]) - EPOCH).days
        q = float(row[q_col]) if row[q_col] and row[q_col + 1] in ("A", "A:e") else None
        n = float(row[n_col]) if len(row) > n_col + 1 and row[n_col] and row[n_col + 1] == "A" else None
        out.append((day, q, n))
    return out


def water_year(day):
    date = EPOCH + dt.timedelta(days=int(day))
    return date.year + (date.month >= 10)


def sample_design(days, discharge, truth, seed):
    """One random sensor day per calendar month plus four random high-flow (top 20%) sensor days per water year."""
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(np.isfinite(truth))
    months = np.array([(EPOCH + dt.timedelta(days=int(d))).year * 12 + (EPOCH + dt.timedelta(days=int(d))).month for d in days])
    chosen = {int(rng.choice(valid[months[valid] == m])) for m in np.unique(months[valid])}
    high = np.quantile(discharge, 0.8)
    years = np.array([water_year(d) for d in days])
    for y in np.unique(years[valid]):
        pool = [i for i in valid[(years[valid] == y) & (discharge[valid] >= high)] if i not in chosen]
        chosen |= {int(i) for i in rng.choice(pool, size=min(4, len(pool)), replace=False)} if pool else set()
    return np.array(sorted(chosen))


def derive(paths):
    sites = {}
    for (site, (name, lat, lon, area)), raw in zip(SITES.items(), RAW_SOURCES):
        end = (dt.date.fromisoformat(RECORD_END[site]) - EPOCH).days
        rows = [r for r in parse(paths[raw]) if r[0] <= end and r[1] is not None and r[1] > 0]
        first = next(i for i, r in enumerate(rows) if r[2] is not None)
        rows = rows[first:]
        days = np.array([r[0] for r in rows])
        discharge = np.array([r[1] * CFS for r in rows])
        truth = np.array([np.nan if r[2] is None else r[2] for r in rows])
        samples = sample_design(days, discharge, truth, seed=int(site))
        sites[site] = {"name": name, "lat": lat, "lon": lon, "drainage_area_km2": round(area * 2.58998811, 1),
                       "day": days.tolist(), "discharge_m3s": np.round(discharge, 4).tolist(),
                       "nitrate_mg_l": [None if not np.isfinite(v) else v for v in truth], "sample_index": samples.tolist()}
    return {DATA: {"source": "USGS NWIS daily values, approved (non-estimated) sensor nitrate; approved discharge",
                   "day_unit": "days since 1970-01-01", "sites": sites}}


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])["sites"]
    public, truth = {}, {}
    for site, s in raw.items():
        day = np.array(s["day"], float)
        nitrate = np.array([np.nan if v is None else v for v in s["nitrate_mg_l"]], float)
        index = np.array(s["sample_index"], int)
        public[site] = {"day": day, "decimal_year": 1970.0 + day / 365.25, "discharge_m3s": np.array(s["discharge_m3s"], float),
                        "sample_index": index.astype(float), "sample_nitrate_mg_l": nitrate[index], "lat": s["lat"], "lon": s["lon"],
                        "drainage_area_km2": s["drainage_area_km2"]}
        truth[site] = nitrate
    return public, truth


def tricube(distance, half_width):
    x = np.clip(np.abs(distance) / half_width, 0.0, 1.0)
    return (1.0 - x ** 3) ** 3


def wrtds(site, minimum=100):
    """Weighted regressions on time, discharge and season (Hirsch et al. 2010) evaluated on every day."""
    t, q = site["decimal_year"], np.log(site["discharge_m3s"])
    index = site["sample_index"].astype(int)
    ts, qs, ys = t[index], q[index], np.log(np.maximum(site["sample_nitrate_mg_l"], 0.01))
    design = np.column_stack([np.ones_like(ts), ts - 2000.0, qs, np.sin(2 * np.pi * ts), np.cos(2 * np.pi * ts)])
    target = np.column_stack([np.ones_like(t), t - 2000.0, q, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    widths = np.tile([7.0, 2.0, 0.5], (len(t), 1))
    for _ in range(30):  # widen the windows by 10 percent where fewer than ``minimum`` samples have weight
        season = np.abs(((t[:, None] - ts[None, :]) + 0.5) % 1.0 - 0.5)
        weights = (tricube(t[:, None] - ts[None, :], widths[:, :1]) * tricube(q[:, None] - qs[None, :], widths[:, 1:2])
                   * tricube(season, widths[:, 2:3]))
        short = np.sum(weights > 0, axis=1) < min(minimum, len(ts))
        if not short.any():
            break
        widths[short] *= 1.1
    xtwx = np.einsum("ds,si,sj->dij", weights, design, design) + 1e-9 * np.eye(5)
    xtwy = np.einsum("ds,si,s->di", weights, design, ys)
    beta = np.linalg.solve(xtwx, xtwy[..., None])[..., 0]
    residual = ys[None, :] - beta @ design.T
    variance = np.sum(weights * residual ** 2, axis=1) / np.maximum(np.sum(weights, axis=1) - 5.0, 1.0)
    return np.exp(np.sum(target * beta, axis=1) + variance / 2.0)


def periods(site, kind):
    if kind == "month":
        return np.array([(EPOCH + dt.timedelta(days=int(d))).year * 12 + (EPOCH + dt.timedelta(days=int(d))).month for d in site["day"]])
    return np.array([water_year(d) for d in site["day"]])


def load_errors(site, truth, prediction, kind="month"):
    """Absolute relative load errors per calendar month (>= MIN_MONTH_DAYS sensor days) or water year (>= MIN_YEAR_DAYS)."""
    labels = periods(site, kind)
    minimum = MIN_MONTH_DAYS if kind == "month" else MIN_YEAR_DAYS
    flux_true = truth * site["discharge_m3s"] * KG_PER_DAY
    flux_pred = prediction * site["discharge_m3s"] * KG_PER_DAY
    out = []
    for label in np.unique(labels):
        ok = (labels == label) & np.isfinite(truth)
        if ok.sum() >= minimum:
            out.append(abs(flux_pred[ok].sum() - flux_true[ok].sum()) / flux_true[ok].sum())
    return np.array(out)


def daily_nrmse(site, truth, prediction):
    held = np.isfinite(truth)
    held[site["sample_index"].astype(int)] = False
    return float(np.sqrt(np.mean((prediction[held] - truth[held]) ** 2)) / np.mean(truth[held]))


def score(module, cpu_seconds=120):
    public, truth = load()
    output = earth_sandbox.call(module, "estimate_nitrate", [public], cpu_seconds=cpu_seconds)
    require(isinstance(output, dict), "estimate_nitrate must return a dict keyed by site number")
    rows = {key: [] for key in ("month", "year", "ref_month", "ref_year", "nrmse", "ref_nrmse")}
    per_site = {}
    for site, s in public.items():
        require(site in output, f"missing predictions for site {site}")
        prediction = ec.as_array(output[site], (len(s["day"]),), f"site {site} nitrate", low=0.0, high=100.0)
        reference = wrtds(s)
        monthly = load_errors(s, truth[site], prediction)
        rows["month"] += list(monthly)
        rows["year"] += list(load_errors(s, truth[site], prediction, "year"))
        rows["ref_month"] += list(load_errors(s, truth[site], reference))
        rows["ref_year"] += list(load_errors(s, truth[site], reference, "year"))
        rows["nrmse"].append(daily_nrmse(s, truth[site], prediction))
        rows["ref_nrmse"].append(daily_nrmse(s, truth[site], reference))
        per_site[site] = round(100 * float(np.mean(monthly)), 2)
    pct = lambda values: round(100 * float(np.mean(values)), 3)  # noqa: E731
    details = {"site_months": len(rows["month"]), "site_years": len(rows["year"]),
               "samples": int(sum(len(s["sample_index"]) for s in public.values())),
               "mean_abs_monthly_load_error_percent_by_site": per_site, "mean_abs_annual_load_error_percent": pct(rows["year"]),
               "daily_concentration_nrmse": round(float(np.mean(rows["nrmse"])), 4),
               "wrtds_mean_abs_monthly_load_error_percent": pct(rows["ref_month"]), "wrtds_mean_abs_annual_load_error_percent": pct(rows["ref_year"]),
               "wrtds_daily_concentration_nrmse": round(float(np.mean(rows["ref_nrmse"])), 4)}
    return 100.0 * float(np.mean(rows["month"])), details
