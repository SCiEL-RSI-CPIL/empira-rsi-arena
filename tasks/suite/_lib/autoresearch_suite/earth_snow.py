"""Snow water equivalent from snow depth at unseen SNOTEL stations.

Snow depth is far easier to measure than snow water equivalent (SWE), so depth-to-SWE
conversion (Sturm et al. 2010; Hill et al. 2019) underpins snow surveys, lidar and depth-sensor
networks. 162 SNOTEL stations (stratified by state) with daily snow depth and snow-pillow SWE
for water years 2006-2025 are split into 114 training and 48 held-out stations. The candidate
sees training-station depth and SWE, held-out-station depth only (all valid days, so depth
history can be used) and station metadata including a 2006-2015 precipitation normal. The score
is the root-mean-square SWE error (mm) on held-out snow days (depth >= 10.16 cm, SWE >= 10.16 mm,
bulk density 0.05-0.65, both values flagged valid).
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

STATIONS = (
    "1003:AK:SNTL", "1063:AK:SNTL", "1064:AK:SNTL", "1176:AK:SNTL", "2080:AK:SNTL", "963:AK:SNTL", "966:AK:SNTL", "511:AZ:SNTL",
    "519:AZ:SNTL", "617:AZ:SNTL", "927:AZ:SNTL", "1049:CA:SNTL", "301:CA:SNTL", "518:CA:SNTL", "540:CA:SNTL", "587:CA:SNTL",
    "697:CA:SNTL", "784:CA:SNTL", "848:CA:SNTL", "1033:CO:SNTL", "1041:CO:SNTL", "1058:CO:SNTL", "1059:CO:SNTL", "1061:CO:SNTL",
    "335:CO:SNTL", "380:CO:SNTL", "408:CO:SNTL", "467:CO:SNTL", "538:CO:SNTL", "547:CO:SNTL", "556:CO:SNTL", "564:CO:SNTL",
    "586:CO:SNTL", "624:CO:SNTL", "629:CO:SNTL", "680:CO:SNTL", "688:CO:SNTL", "773:CO:SNTL", "839:CO:SNTL", "857:CO:SNTL",
    "870:CO:SNTL", "913:CO:SNTL", "940:CO:SNTL", "312:ID:SNTL", "425:ID:SNTL", "450:ID:SNTL", "496:ID:SNTL", "524:ID:SNTL",
    "588:ID:SNTL", "695:ID:SNTL", "735:ID:SNTL", "738:ID:SNTL", "747:ID:SNTL", "761:ID:SNTL", "830:ID:SNTL", "978:ID:SNTL",
    "988:ID:SNTL", "1009:MT:SNTL", "311:MT:SNTL", "349:MT:SNTL", "360:MT:SNTL", "410:MT:SNTL", "413:MT:SNTL", "433:MT:SNTL",
    "448:MT:SNTL", "458:MT:SNTL", "516:MT:SNTL", "530:MT:SNTL", "590:MT:SNTL", "664:MT:SNTL", "667:MT:SNTL", "690:MT:SNTL",
    "693:MT:SNTL", "754:MT:SNTL", "813:MT:SNTL", "835:MT:SNTL", "836:MT:SNTL", "903:MT:SNTL", "916:MT:SNTL", "917:MT:SNTL",
    "1048:NM:SNTL", "532:NM:SNTL", "665:NM:SNTL", "715:NM:SNTL", "755:NM:SNTL", "1006:NV:SNTL", "334:NV:SNTL", "453:NV:SNTL",
    "454:NV:SNTL", "569:NV:SNTL", "698:NV:SNTL", "811:NV:SNTL", "1000:OR:SNTL", "1044:OR:SNTL", "351:OR:SNTL", "398:OR:SNTL",
    "401:OR:SNTL", "422:OR:SNTL", "608:OR:SNTL", "651:OR:SNTL", "666:OR:SNTL", "687:OR:SNTL", "726:OR:SNTL", "729:OR:SNTL",
    "733:OR:SNTL", "810:OR:SNTL", "925:OR:SNTL", "354:SD:SNTL", "920:SD:SNTL", "1013:UT:SNTL", "1039:UT:SNTL", "1054:UT:SNTL",
    "1056:UT:SNTL", "1065:UT:SNTL", "366:UT:SNTL", "374:UT:SNTL", "383:UT:SNTL", "390:UT:SNTL", "400:UT:SNTL", "455:UT:SNTL",
    "475:UT:SNTL", "557:UT:SNTL", "592:UT:SNTL", "593:UT:SNTL", "684:UT:SNTL", "714:UT:SNTL", "723:UT:SNTL", "790:UT:SNTL",
    "795:UT:SNTL", "814:UT:SNTL", "864:UT:SNTL", "865:UT:SNTL", "1043:WA:SNTL", "515:WA:SNTL", "788:WA:SNTL", "841:WA:SNTL",
    "909:WA:SNTL", "910:WA:SNTL", "911:WA:SNTL", "912:WA:SNTL", "941:WA:SNTL", "974:WA:SNTL", "984:WA:SNTL", "985:WA:SNTL",
    "990:WA:SNTL", "326:WY:SNTL", "350:WY:SNTL", "367:WY:SNTL", "499:WY:SNTL", "577:WY:SNTL", "616:WY:SNTL", "625:WY:SNTL",
    "668:WY:SNTL", "676:WY:SNTL", "732:WY:SNTL", "765:WY:SNTL", "772:WY:SNTL", "775:WY:SNTL", "779:WY:SNTL", "786:WY:SNTL",
    "831:WY:SNTL", "852:WY:SNTL",
)  # 162 stations drawn (seed 20261010) by state from the 616 with daily depth from WY2006 to WY2025
API = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/"
BATCHES = [STATIONS[i:i + 9] for i in range(0, len(STATIONS), 9)]
LICENSE = {"pinning": "live", "license": "Public domain (USDA Natural Resources Conservation Service, Air and Water Database)",
           "license_url": "https://www.nrcs.usda.gov/resources/data-and-reports/snow-and-climate-monitoring-predefined-reports-and-maps",
           "citation": "USDA NRCS National Water and Climate Center (2026). SNOTEL daily snow water equivalent, snow depth and precipitation, Air and Water Database REST API.",
           "retrieved": "2026-10-10"}
RAW_SOURCES = ("snotel_station_metadata.json",) + tuple(f"snotel_daily_batch_{k:02d}.json" for k in range(len(BATCHES)))
SOURCE_SPECS = ({"name": RAW_SOURCES[0], "url": API + "stations?stationTriplets=" + ",".join(STATIONS).replace(":", "%3A") + "&returnStationElements=false&activeOnly=false", **LICENSE},) + tuple(
    {"name": name, "url": API + "data?stationTriplets=" + ",".join(batch).replace(":", "%3A")
     + "&elements=WTEQ,SNWD,PREC&duration=DAILY&beginDate=2005-10-01&endDate=2025-09-30&periodRef=END&centralTendencyType=NONE&returnFlags=true", **LICENSE}
    for name, batch in zip(RAW_SOURCES[1:], BATCHES))
DATA = "snow/snotel_depth_swe_162_stations.json.gz"
DERIVED = {DATA: "4012f084a2495dbcf3036cf088ffd46ae9ca2c5288b99b79c11ca237346d5ee4"}
EPOCH = dt.date(1970, 1, 1)
VALID = ("V",)  # SNOTEL quality-control flag "valid"; edited, estimated and suspect values are not used
HELD_OUT_FRACTION, SPLIT_SEED = 0.3, 7


def _series(element):
    out = {}
    for v in element.get("values", []):
        if v.get("value") is not None:
            out[v["date"][:10]] = (float(v["value"]), v.get("qcFlag", ""))
    return out


def derive(paths):
    meta = {s["stationTriplet"]: s for s in json.loads(paths[RAW_SOURCES[0]].read_text())}
    stations = {}
    for name in RAW_SOURCES[1:]:
        for record in json.loads(paths[name].read_text()):
            elements = {e["stationElement"]["elementCode"]: _series(e) for e in record["data"]}
            depth, swe, prec = elements.get("SNWD", {}), elements.get("WTEQ", {}), elements.get("PREC", {})
            dates = sorted(d for d, (v, flag) in depth.items() if flag in ("V", "E") and v >= 0)
            m = meta[record["stationTriplet"]]
            totals = []
            for year in range(2006, 2016):  # water-year precipitation totals (accumulated PREC at 30 September)
                end = prec.get(f"{year}-09-30")
                if end and end[1] in VALID:
                    totals.append(end[0] * 25.4)
            stations[record["stationTriplet"]] = {
                "name": m["name"], "state": m["stateCode"], "lat": m["latitude"], "lon": m["longitude"],
                "elevation_m": round(m["elevation"] * 0.3048, 1), "precip_normal_mm": round(float(np.median(totals)), 1) if totals else None,
                "day": [(dt.date.fromisoformat(d) - EPOCH).days for d in dates],
                "depth_cm": [round(depth[d][0] * 2.54, 2) for d in dates],
                "swe_mm": [round(swe[d][0] * 25.4, 2) if d in swe and swe[d][1] in VALID and depth[d][1] in VALID else None for d in dates]}
    return {DATA: {"source": "USDA NRCS AWDB REST API, SNOTEL daily SNWD, WTEQ and PREC, water years 2006-2025", "day_unit": "days since 1970-01-01",
                   "stations": {k: stations[k] for k in sorted(stations)}}}


def split():
    """Held-out stations: 30 percent of each state's stations (rounded), chosen with a fixed seed."""
    rng = np.random.default_rng(SPLIT_SEED)
    held = []
    for state in sorted({s.split(":")[1] for s in STATIONS}):
        members = sorted(s for s in STATIONS if s.split(":")[1] == state)
        held += list(rng.choice(members, size=max(1, round(HELD_OUT_FRACTION * len(members))), replace=False))
    return set(held)


def targets(station):
    depth, swe = station["depth_cm"], station["swe_mm"]
    with np.errstate(invalid="ignore", divide="ignore"):
        density = swe / (depth * 10.0)
        return np.flatnonzero(np.isfinite(swe) & (depth >= 10.16) & (swe >= 10.16) & (density >= 0.05) & (density <= 0.65))


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])["stations"]
    held = split()
    train, test, truth = {}, {}, {}
    for key, s in raw.items():
        station = {"state": s["state"], "lat": s["lat"], "lon": s["lon"], "elevation_m": s["elevation_m"],
                   "precip_normal_mm": float("nan") if s["precip_normal_mm"] is None else s["precip_normal_mm"],
                   "day": np.array(s["day"], float), "depth_cm": np.array(s["depth_cm"], float),
                   "swe_mm": np.array([np.nan if v is None else v for v in s["swe_mm"]], float)}
        index = targets(station)
        station["target_index"] = index.astype(float)
        if key in held:
            truth[key] = station.pop("swe_mm")[index]
            test[key] = station
        else:
            train[key] = station
    return train, test, truth


def doy(day):
    """Day of year relative to 1 January (negative from October to December)."""
    dates = [EPOCH + dt.timedelta(days=int(d)) for d in day]
    return np.array([(d - dt.date(d.year + (d.month >= 10), 1, 1)).days for d in dates], float)


def sturm_form(train):
    """Density model rho = (rmax - r0)(1 - exp(-k1 h - k2 doy)) + r0 (form of Sturm et al. 2010), fitted on training stations."""
    from scipy.optimize import least_squares
    h, d, swe = [], [], []
    for s in train.values():
        i = s["target_index"].astype(int)
        h.append(s["depth_cm"][i]); d.append(doy(s["day"][i])); swe.append(s["swe_mm"][i])
    h, d, swe = np.concatenate(h), np.concatenate(d), np.concatenate(swe)

    def model(p, h, d):
        return (p[0] - p[1]) * (1 - np.exp(-p[2] * h - p[3] * d)) + p[1]
    fit = least_squares(lambda p: model(p, h, d) * h * 10.0 - swe, x0=[0.55, 0.25, 0.001, 0.004], bounds=([0.2, 0.05, 0, -0.02], [1.0, 0.5, 0.1, 0.05]))
    return lambda h, d: model(fit.x, h, d) * h * 10.0, fit.x


def score(module, cpu_seconds=180):
    train, test, truth = load()
    output = earth_sandbox.call(module, "estimate_swe", [train, test], cpu_seconds=cpu_seconds, memory_mb=3500)
    require(isinstance(output, dict), "estimate_swe must return a dict keyed by station triplet")
    reference, coefficients = sturm_form(train)
    median_density = np.median(np.concatenate([s["swe_mm"][s["target_index"].astype(int)] / (10 * s["depth_cm"][s["target_index"].astype(int)])
                                               for s in train.values()]))
    pred, true, ref, base, density_err = [], [], [], [], []
    for key, s in test.items():
        i = s["target_index"].astype(int)
        require(key in output, f"missing predictions for station {key}")
        values = ec.as_array(output[key], (len(i),), f"station {key} SWE", low=0.0, high=10000.0)
        pred.append(values); true.append(truth[key])
        ref.append(reference(s["depth_cm"][i], doy(s["day"][i])))
        base.append(median_density * 10 * s["depth_cm"][i])
        density_err.append(values / (10 * s["depth_cm"][i]) - truth[key] / (10 * s["depth_cm"][i]))
    pred, true, ref, base, density_err = (np.concatenate(x) for x in (pred, true, ref, base, density_err))
    details = {"held_out_stations": len(test), "training_stations": len(train), "held_out_days": int(len(true)),
               "mae_mm": round(float(np.mean(np.abs(pred - true))), 3), "bias_mm": round(float(np.mean(pred - true)), 3),
               "density_rmse": round(float(np.sqrt(np.mean(density_err ** 2))), 5),
               "sturm_form_fitted_rmse_mm": round(ec.rmse(true, ref), 3), "sturm_form_parameters": [round(float(x), 5) for x in coefficients],
               "median_density_rmse_mm": round(ec.rmse(true, base), 3)}
    return ec.rmse(true, pred), details
