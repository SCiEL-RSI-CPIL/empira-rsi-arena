"""Probabilistic forecasts of weekly influenza hospital admissions, scored by the weighted interval score.

Data are the FluSight hub's target file of weekly laboratory-confirmed influenza hospital
admissions per US state (CDC NHSN, redistributed by the hub under the MIT licence). The
candidate is asked, week by week through a held-out season, for 23 predictive quantiles at
horizons of one to four weeks for every state; at each reference date it sees only the
history up to that week. The evaluator scores the quantiles with the weighted interval
score of Bracher, Ray, Gneiting and Reich (2021), the metric used by the FluSight hub.
"""
from __future__ import annotations

import csv

import numpy as np

from .bio_physics_common import as_array, load_derived, run_sandboxed
from .runtime import require

RAW_SOURCES = ("flusight-target-hospital-admissions.csv",)
DATA = "flu/flusight_hospital_admissions.json.gz"
DERIVED = {DATA: "b7e8877a4c30a4d35f8b8bf623674acbd258e9f01376fdca6bf2f44ce2af776b"}

QUANTILES = (0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)
HORIZONS = (1, 2, 3, 4)


def derive(paths):
    rows = list(csv.DictReader(paths[RAW_SOURCES[0]].open(newline="")))
    require(rows and {"date", "location", "location_name", "value"} <= set(rows[0]), "unexpected FluSight columns")
    dates = sorted({r["date"] for r in rows})
    names, values = {}, {}
    for row in rows:
        names[row["location"]] = row["location_name"]
        if row["value"] not in ("NA", ""):
            values.setdefault(row["location"], {})[row["date"]] = float(row["value"])
    locations = sorted(names)
    series = {code: [values.get(code, {}).get(date) for date in dates] for code in locations}
    return {DATA: {"source": RAW_SOURCES[0], "dates": dates, "location_names": names, "series": series,
                   "note": "weekly laboratory-confirmed influenza hospital admissions; null where the hub reports NA"}}


def load():
    return load_derived(DATA, DERIVED[DATA])


def season_of(date):
    year, month = int(date[:4]), int(date[5:7])
    return f"{year}-{year + 1}" if month >= 8 else f"{year - 1}-{year}"


SEASON_MONTHS = (11, 12, 1, 2, 3, 4)


def protocol(payload, season, locations=None):
    """Reference-date indices inside the held-out season's active window and the scored location codes."""
    dates = payload["dates"]
    codes = [c for c in sorted(payload["series"]) if c != "US"] if locations is None else list(locations)
    indices = [i for i, date in enumerate(dates)
               if season_of(date) == season and int(date[5:7]) in SEASON_MONTHS and i + max(HORIZONS) < len(dates)]
    require(indices, f"no reference dates for season {season}")
    return indices, codes


def weighted_interval_score(observation, quantiles, levels=QUANTILES):
    """WIS of Bracher et al. (2021) from a sorted vector of predictive quantiles."""
    levels = np.asarray(levels, float)
    median_index = int(np.argmin(np.abs(levels - 0.5)))
    total = 0.5 * abs(observation - quantiles[median_index])
    intervals = median_index  # one central prediction interval per pair of symmetric levels
    for k in range(intervals):
        lower, upper = quantiles[k], quantiles[len(levels) - 1 - k]
        alpha = 2 * levels[k]
        interval = upper - lower
        if observation < lower:
            interval += (2 / alpha) * (lower - observation)
        elif observation > upper:
            interval += (2 / alpha) * (observation - upper)
        total += (alpha / 2) * interval
    return float(total / (intervals + 0.5))


def check_forecast(answer, codes):
    """Validate {location: (4, 23) non-decreasing non-negative quantile matrix}."""
    require(isinstance(answer, dict), "forecast(): expected a dict keyed by location code")
    missing = [c for c in codes if c not in answer]
    require(not missing, f"forecast(): missing locations {missing[:5]}")
    out = {}
    for code in codes:
        matrix = as_array(answer[code], shape=(len(HORIZONS), len(QUANTILES)), name=f"forecast[{code}]",
                          low=0.0, high=1e7)
        require(bool(np.all(np.diff(matrix, axis=1) >= -1e-9)), f"forecast[{code}]: quantiles must be non-decreasing")
        out[code] = np.sort(matrix, axis=1)
    return out


DRIVER = '''

def _bp_flu_driver(jobs, quantiles, horizons):
    _bp_seed()
    return [_bp_plain(forecast(job["history"], job["reference_date"], quantiles, horizons)) for job in jobs]
'''


def score(module, P):
    payload = load()
    dates, series = payload["dates"], payload["series"]
    indices, codes = protocol(payload, P["season"])
    jobs = []
    for i in indices:
        history = {code: [series[code][k] for k in range(i + 1)] for code in sorted(series)}
        jobs.append({"reference_date": dates[i], "history": history})
    answers = run_sandboxed(module, DRIVER, "_bp_flu_driver", [jobs, list(QUANTILES), list(HORIZONS)],
                            required=("forecast",), cpu_seconds=int(P.get("cpu_seconds", 180)), memory_mb=2048)
    require(isinstance(answers, list) and len(answers) == len(jobs), f"expected {len(jobs)} weekly forecasts")
    scores, per_horizon = [], {h: [] for h in HORIZONS}
    scored, skipped = 0, 0
    for i, answer in zip(indices, answers):
        matrices = check_forecast(answer, codes)
        for code in codes:
            for position, horizon in enumerate(HORIZONS):
                observation = series[code][i + horizon]
                if observation is None:
                    skipped += 1
                    continue
                value = weighted_interval_score(observation, matrices[code][position])
                scores.append(value)
                per_horizon[horizon].append(value)
                scored += 1
    require(scores, "no scored forecasts")
    return float(np.mean(scores)), {
        "season": P["season"], "reference_dates": len(indices), "locations": len(codes),
        "scored_forecasts": scored, "skipped_missing_truth": skipped,
        "mean_wis_by_horizon": {str(h): float(np.mean(per_horizon[h])) for h in HORIZONS},
        "median_wis": float(np.median(scores)), "worst_wis": float(np.max(scores))}
