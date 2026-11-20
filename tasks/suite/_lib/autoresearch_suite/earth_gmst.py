"""Out-of-sample emulation of observed global mean surface temperature from radiative forcing and ENSO.

The candidate fits any model of annual HadCRUT5 global mean temperature anomalies on 1850-1979,
given effective radiative forcing (ERF) components for 1750-2024 (Forster et al. 2025 update of
the IPCC AR6 series) and a monthly relative Nino 3.4 index from ERSSTv5 (Nino 3.4 anomaly minus
the 20S-20N mean SST anomaly, which removes the warming trend). It then predicts 1980-2024, the
period of rapid warming, without seeing any temperature after 1979. The score is the root-mean-
square error of the predicted anomalies (deg C).
"""
from __future__ import annotations

import csv

import numpy as np

from . import earth_common as ec
from . import earth_enso
from . import earth_sandbox
from .runtime import require

COMMIT = "852f4f7b0733fa4d573c145cdfaf556da5589231"
RAW_SOURCES = ("hadcrut5_global_annual.csv", "igcc2024_erf_best_aggregates_1750-2024.csv", earth_enso.RAW_SOURCES[0])
SOURCE_SPECS = (
    {"name": RAW_SOURCES[0], "url": "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv",
     "pinning": "live", "license": "Open Government Licence v3.0 (Met Office Hadley Centre / Climatic Research Unit)",
     "license_url": "https://www.metoffice.gov.uk/hadobs/hadcrut5/terms_and_conditions.html",
     "citation": "Morice, C. P. et al. (2021). An updated assessment of near-surface temperature change from 1850: the HadCRUT5 data set. Journal of Geophysical Research: Atmospheres 126, e2019JD032361.",
     "retrieved": "2026-10-10"},
    {"name": RAW_SOURCES[1], "url": f"https://raw.githubusercontent.com/ClimateIndicator/forcing-timeseries/{COMMIT}/output/ERF_best_aggregates_1750-2024.csv",
     "pinning": f"immutable git commit {COMMIT}", "license": "MIT (repository licence of ClimateIndicator/forcing-timeseries)",
     "license_url": f"https://github.com/ClimateIndicator/forcing-timeseries/blob/{COMMIT}/LICENSE",
     "citation": "Forster, P. M. et al. (2025). Indicators of Global Climate Change 2024: annual update of key indicators of the state of the climate system and human influence. Earth System Science Data 17, 2641-2680.",
     "retrieved": "2026-10-10"},
)
DATA = "gmst/hadcrut5_erf_enso_1850_2024.json.gz"
DERIVED = {DATA: "765335b81d2931d5eabf07117646d254f9c185cc4d7e5ad35f158650d0341dd1"}
TRAIN, TEST = (1850, 1979), (1980, 2024)
COMPONENTS = ("CO2", "CH4", "N2O", "halogen", "minor", "O3", "H2O_stratospheric", "contrails", "aerosol-radiation_interactions",
              "aerosol-cloud_interactions", "BC_on_snow", "land_use", "solar", "volcanic", "anthro", "total")


def relative_nino34(path):
    """Monthly Nino 3.4 anomaly minus the 20S-20N mean SST anomaly (both from 1951-1980 monthly climatologies)."""
    import netCDF4  # preparation-only dependency
    with netCDF4.Dataset(path) as nc:
        dates = netCDF4.num2date(nc.variables["time"][:], nc.variables["time"].units)
        lat = np.asarray(nc.variables["lat"][:], float)
        lon = np.asarray(nc.variables["lon"][:], float)
        sst = np.ma.filled(nc.variables["sst"][:].astype(float), np.nan)
    sst[(sst < -5) | (sst > 50)] = np.nan
    year = np.array([d.year for d in dates])
    month = np.array([d.month for d in dates])
    nino = earth_enso.box_mean(sst, lat, lon, earth_enso.BOXES["nino34"])
    tropics = earth_enso.box_mean(sst, lat, lon, ((-20.0, 20.0), (0.0, 360.0)))
    base = (year >= 1951) & (year <= 1980)

    def anomaly(x):
        return x - np.array([x[base & (month == m)].mean() for m in range(1, 13)])[month - 1]
    return year, month, anomaly(nino) - anomaly(tropics)


def derive(paths):
    with open(paths[RAW_SOURCES[0]], newline="") as stream:
        rows = [r for r in csv.reader(stream)][1:]
    temp = {int(r[0]): [float(r[1]), float(r[2]), float(r[3])] for r in rows if TRAIN[0] <= int(r[0]) <= TEST[1]}
    with open(paths[RAW_SOURCES[1]], newline="") as stream:
        reader = csv.DictReader(stream)
        forcing = {int(float(r[""])): {c: float(r[c]) for c in COMPONENTS} for r in reader}
    year, month, relative = relative_nino34(paths[earth_enso.RAW_SOURCES[0]])
    keep = year <= TEST[1]
    years = sorted(temp)
    return {DATA: {"temperature_source": RAW_SOURCES[0], "forcing_source": RAW_SOURCES[1], "units": "deg C anomaly from 1961-1990; ERF W m-2",
                   "year": years, "gmst": [round(temp[y][0], 6) for y in years], "gmst_low": [round(temp[y][1], 6) for y in years],
                   "gmst_high": [round(temp[y][2], 6) for y in years], "forcing_year": sorted(forcing),
                   "forcing": {c: [round(forcing[y][c], 8) for y in sorted(forcing)] for c in COMPONENTS},
                   "enso_year": year[keep].tolist(), "enso_month": month[keep].tolist(),
                   "relative_nino34": np.round(relative[keep], 4).tolist()}}


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])
    out = {key: np.array(raw[key], float) for key in ("year", "gmst", "gmst_low", "gmst_high", "forcing_year", "enso_year", "enso_month", "relative_nino34")}
    out["forcing"] = {c: np.array(v, float) for c, v in raw["forcing"].items()}
    return out


def public_inputs(data):
    train = (data["year"] >= TRAIN[0]) & (data["year"] <= TRAIN[1])
    training = {"year": data["year"][train], "gmst": data["gmst"][train], "gmst_low": data["gmst_low"][train],
                "gmst_high": data["gmst_high"][train]}
    inputs = {"forcing_year": data["forcing_year"], "forcing": data["forcing"], "enso_year": data["enso_year"],
              "enso_month": data["enso_month"], "relative_nino34": data["relative_nino34"],
              "predict_years": np.arange(TEST[0], TEST[1] + 1, dtype=float)}
    return training, inputs


def ebm(forcing, lam=1.3, gamma=0.7, c=8.0, c0=100.0):
    """Two-layer energy-balance model, annual steps (illustrative parameters, not calibrated to observations)."""
    t = t0 = 0.0
    out = []
    for f in forcing:
        t, t0 = t + (f - lam * t - gamma * (t - t0)) / c, t0 + gamma * (t - t0) / c0
        out.append(t)
    return np.array(out)


def references(data, training, inputs):
    years, test = data["forcing_year"], inputs["predict_years"]
    total = data["forcing"]["total"]
    in_train = np.isin(years, training["year"])
    in_test = np.isin(years, test)
    out = {}
    slope, intercept = np.polyfit(total[in_train], training["gmst"], 1)
    out["regression_on_total_forcing"] = intercept + slope * total[in_test]
    model = ebm(total)
    out["two_layer_ebm"] = model[in_test] + np.mean(training["gmst"] - model[in_train])
    enso = np.array([data["relative_nino34"][(data["enso_year"] == y - 1) & (data["enso_month"] >= 9) | (data["enso_year"] == y) & (data["enso_month"] <= 8)].mean()
                     if y >= 1855 else 0.0 for y in years])
    smooth = ebm(data["forcing"]["anthro"])
    design = np.column_stack([np.ones(len(years)), smooth, ebm(data["forcing"]["volcanic"]), ebm(data["forcing"]["solar"]), enso])
    coef, *_ = np.linalg.lstsq(design[in_train], training["gmst"], rcond=None)
    out["lean_rind_style_regression"] = design[in_test] @ coef
    return out


def score(module, cpu_seconds=60):
    data = load()
    training, inputs = public_inputs(data)
    require(len(training["year"]) == TRAIN[1] - TRAIN[0] + 1, "prepared data do not cover the training period")
    prediction = earth_sandbox.call(module, "predict_gmst", [training, inputs], cpu_seconds=cpu_seconds)
    n = TEST[1] - TEST[0] + 1
    prediction = ec.as_array(prediction, (n,), "GMST prediction", low=-5.0, high=5.0)
    truth = data["gmst"][(data["year"] >= TEST[0]) & (data["year"] <= TEST[1])]
    refs = references(data, training, inputs)
    trend = lambda x: float(np.polyfit(np.arange(n), x, 1)[0] * 10)  # noqa: E731
    details = {"unit": "deg C", "held_out_years": f"{TEST[0]}-{TEST[1]}", "bias": round(float(np.mean(prediction - truth)), 4),
               "predicted_trend_per_decade": round(trend(prediction), 4), "observed_trend_per_decade": round(trend(truth), 4),
               "rmse_after_removing_bias": round(float(np.std(prediction - truth)), 4),
               **{f"{name}_rmse": round(ec.rmse(truth, values), 4) for name, values in refs.items()}}
    return ec.rmse(truth, prediction), details
