"""Species distribution models from presence-only records, evaluated on independent presence-absence surveys.

Data: the NCEAS benchmark of Elith et al. (2006, 2020): 226 anonymised species in six regions
(AWT, CAN, NSW, NZ, SA, SWI), each with presence-only training records, 10,000-50,000
background points and environmental covariates, plus independent presence-absence survey
sites. The candidate sees presences, background points and the covariates of the survey sites;
the survey outcomes stay in this process. The score is the mean over species of the area
under the ROC curve (AUC) of the candidate's site scores against the observed presence and
absence, the evaluation measure of Valavi et al. (2022).
"""
from __future__ import annotations

import csv

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

OSF = {  # file name -> OSF download id (project kwc4v, data/Records)
    "AWTtrain_po.csv": "4hdyf", "AWTtrain_bg.csv": "9cg63", "AWTtest_env_bird.csv": "azghc", "AWTtest_pa_bird.csv": "n9bfw",
    "AWTtest_env_plant.csv": "cjrs2", "AWTtest_pa_plant.csv": "g5rsy",
    "CANtrain_po.csv": "5cmkn", "CANtrain_bg.csv": "v237z", "CANtest_env.csv": "zdhpm", "CANtest_pa.csv": "yzn8k",
    "NSWtrain_po.csv": "b2naq", "NSWtrain_bg.csv": "bk92y",
    "NSWtest_env_ba.csv": "bc5k2", "NSWtest_pa_ba.csv": "5mqe6", "NSWtest_env_db.csv": "bnzma", "NSWtest_pa_db.csv": "7ur3t",
    "NSWtest_env_nb.csv": "ms8ew", "NSWtest_pa_nb.csv": "mzuca", "NSWtest_env_ot.csv": "w7pb5", "NSWtest_pa_ot.csv": "mtfhr",
    "NSWtest_env_ou.csv": "npva3", "NSWtest_pa_ou.csv": "pqx9m", "NSWtest_env_rt.csv": "bcukt", "NSWtest_pa_rt.csv": "w38en",
    "NSWtest_env_ru.csv": "uzxy7", "NSWtest_pa_ru.csv": "spx5n", "NSWtest_env_sr.csv": "3uhsc", "NSWtest_pa_sr.csv": "bxh8j",
    "NZtrain_po.csv": "gbz42", "NZtrain_bg.csv": "bwvsa", "NZtest_env.csv": "2zdcv", "NZtest_pa.csv": "27xnk",
    "SAtrain_po.csv": "we9cr", "SAtrain_bg.csv": "3f75h", "SAtest_env.csv": "89vst", "SAtest_pa.csv": "pu5tg",
    "SWItrain_po.csv": "j9c8h", "SWItrain_bg.csv": "hjp8d", "SWItest_env.csv": "hnm7y", "SWItest_pa.csv": "tgk3s",
}
RAW_SOURCES = tuple("nceas_" + name for name in OSF)
SOURCE_SPECS = tuple({"name": "nceas_" + name, "url": f"https://osf.io/download/{key}/", "pinning": "immutable OSF file version",
                      "license": "CC BY 4.0 (OSF project SDM_NCEAS, https://osf.io/kwc4v/)", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                      "citation": "Elith, J. et al. (2020). Presence-only and presence-absence data for comparing species distribution modeling methods. Biodiversity Informatics 15, 69-80.",
                      "retrieved": "2026-10-10"} for name, key in OSF.items())
DATA = "sdm/nceas_226_species.json.gz"
DERIVED = {DATA: "04b296caaa12bf289b948df2acd783606df5b295ed9f99d3111cd7ae9dbea7b8"}
REGIONS = ("AWT", "CAN", "NSW", "NZ", "SA", "SWI")
GROUPS = {"AWT": ("bird", "plant"), "NSW": ("ba", "db", "nb", "ot", "ou", "rt", "ru", "sr")}
CATEGORICAL = {"CAN": ["ontveg"], "NSW": ["vegsys"], "NZ": ["age", "toxicats"], "SWI": ["calc"]}  # disdat documentation
COORDINATES = {"AWT": "UTM zone 55 (GDA94), m", "CAN": "longitude/latitude, deg", "NSW": "longitude/latitude, deg",
               "NZ": "New Zealand Map Grid, m", "SA": "longitude/latitude, deg", "SWI": "Swiss national grid, m"}


def _number(text):
    return None if text in ("", "NA") else float(text)


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def derive(paths):
    regions = {}
    for region in REGIONS:
        po = _read(paths[f"nceas_{region}train_po.csv"])
        bg = _read(paths[f"nceas_{region}train_bg.csv"])
        covariates = [c for c in po[0] if c not in ("siteid", "spid", "x", "y", "occ", "group")]
        presence = {}
        for row in po:
            entry = presence.setdefault(row["spid"], {"x": [], "xy": []})
            entry["x"].append([_number(row[c]) for c in covariates])
            entry["xy"].append([float(row["x"]), float(row["y"])])
        groups = {}
        for group in GROUPS.get(region, ("all",)):
            suffix = "" if group == "all" else "_" + group
            env = _read(paths[f"nceas_{region}test_env{suffix}.csv"])
            pa = _read(paths[f"nceas_{region}test_pa{suffix}.csv"])
            require([r["siteid"] for r in env] == [r["siteid"] for r in pa], f"{region}{suffix}: site order differs")
            species = [c for c in pa[0] if c not in ("group", "siteid", "x", "y")]
            groups[group] = {"x": [[_number(r[c]) for c in covariates] for r in env],
                             "xy": [[float(r["x"]), float(r["y"])] for r in env], "species": species,
                             "pa": [[int(r[s]) for s in species] for r in pa]}
        regions[region] = {"covariates": covariates, "categorical": CATEGORICAL.get(region, []), "coordinates": COORDINATES[region],
                           "presence": presence, "background": {"x": [[_number(r[c]) for c in covariates] for r in bg],
                                                                "xy": [[float(r["x"]), float(r["y"])] for r in bg]},
                           "groups": groups}
    return {DATA: {"source": "NCEAS species distribution benchmark (Elith et al. 2020), OSF kwc4v", "regions": regions}}


def _array(rows):
    return np.array([[np.nan if v is None else v for v in row] for row in rows], float)


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])["regions"]
    public, labels = {}, {}
    for region, r in raw.items():
        groups = {g: {"x": _array(v["x"]), "xy": np.array(v["xy"], float), "species": list(v["species"])} for g, v in r["groups"].items()}
        public[region] = {"covariates": list(r["covariates"]), "categorical": list(r["categorical"]), "coordinates": r["coordinates"],
                          "presence": {s: {"x": _array(v["x"]), "xy": np.array(v["xy"], float)} for s, v in r["presence"].items()},
                          "background": {"x": _array(r["background"]["x"]), "xy": np.array(r["background"]["xy"], float)},
                          "sites": groups}
        labels[region] = {g: np.array(v["pa"], int) for g, v in r["groups"].items()}
    return public, labels


def score(module, cpu_seconds=300):
    public, labels = load()
    output = earth_sandbox.call(module, "predict_distributions", [public], cpu_seconds=cpu_seconds, memory_mb=3500)
    require(isinstance(output, dict), "predict_distributions must return a dict keyed by region")
    aucs, by_region = [], {}
    for region, groups in public.items():
        predicted = output.get(region)
        require(isinstance(predicted, dict), f"missing predictions for region {region}")
        region_aucs = []
        for group, sites in groups["sites"].items():
            for k, species in enumerate(sites["species"]):
                require(species in predicted, f"missing predictions for {region} species {species}")
                values = ec.as_array(predicted[species], (len(sites["x"]),), f"{region} {species} predictions")
                observed = labels[region][group][:, k]
                region_aucs.append(ec.roc_auc(observed, values))
        by_region[region] = round(float(np.mean(region_aucs)), 4)
        aucs += region_aucs
    details = {"species": len(aucs), "mean_auc_by_region": by_region, "median_auc": round(float(np.median(aucs)), 4),
               "species_with_auc_below_0.5": int(np.sum(np.array(aucs) < 0.5))}
    return float(np.mean(aucs)), details
