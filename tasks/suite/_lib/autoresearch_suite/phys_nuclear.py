"""Extrapolating nuclear masses: predict the nuclides first measured after the AME2016 evaluation.

Training data are the experimental mass excesses of the Atomic Mass Evaluation 2016
(Wang et al. 2017); the held-out targets are the 71 nuclides (N, Z >= 8) whose masses are
experimental in AME2020 (Wang et al. 2021) but were absent or only estimated in AME2016.
The candidate sees only the AME2016 experimental values and the (N, Z) of the targets;
the evaluator compares its predictions with the AME2020 values. For context the evaluator
also scores the evaluators' own AME2016 extrapolations (values marked '#') of the same
nuclides, which are published estimates derived from the smoothness of the mass surface.
"""
from __future__ import annotations

import numpy as np

from .bio_physics_common import as_array, load_derived, run_sandboxed
from .runtime import require

RAW_SOURCES = ("ame2016-mass16.txt", "ame2020-mass_1.mas20.txt")
DATA = "nuclear/ame2016_to_ame2020.json.gz"
DERIVED = {DATA: "4198fae5f4e742c2e56c8a8a951884866bf4710447d685ec2044a250fcf28bfa"}
MIN_NZ = 8


def parse_table(text, mass_columns, uncertainty_columns):
    """Fixed-width AME mass table -> {(N, Z): (A, element, mass excess field, uncertainty field)}."""
    rows, started = {}, False
    for line in text.splitlines():
        if "MASS EXCESS" in line and "BINDING" in line:
            started = True
            continue
        if not started or len(line) < 60:
            continue
        try:
            n, z, a = int(line[4:9]), int(line[9:14]), int(line[14:19])
        except ValueError:
            continue
        rows[(n, z)] = (a, line[20:23].strip(), line[slice(*mass_columns)].strip(), line[slice(*uncertainty_columns)].strip())
    return rows


def _number(field):
    return float(field.replace("#", "."))


def derive(paths):
    old = parse_table(paths[RAW_SOURCES[0]].read_text(encoding="latin-1"), (28, 41), (41, 52))
    new = parse_table(paths[RAW_SOURCES[1]].read_text(encoding="latin-1"), (28, 42), (42, 54))
    require(len(old) > 3000 and len(new) > 3000, "unexpectedly short AME tables")

    def experimental(rows):
        return {k: v for k, v in rows.items() if "#" not in v[2] and "*" not in v[2] and k[0] >= MIN_NZ and k[1] >= MIN_NZ}

    old_exp, new_exp = experimental(old), experimental(new)
    training = [[n, z, a, el, _number(mass), _number(unc)] for (n, z), (a, el, mass, unc) in sorted(old_exp.items())]
    targets = []
    for (n, z), (a, el, mass, unc) in sorted(new_exp.items()):
        if (n, z) in old_exp:
            continue
        estimate = _number(old[(n, z)][2]) if (n, z) in old and "*" not in old[(n, z)][2] else None
        targets.append([n, z, a, el, _number(mass), _number(unc), estimate])
    return {DATA: {"sources": list(RAW_SOURCES), "min_nz": MIN_NZ,
                   "training_fields": ["N", "Z", "A", "element", "mass_excess_keV", "uncertainty_keV"],
                   "target_fields": ["N", "Z", "A", "element", "ame2020_mass_excess_keV", "ame2020_uncertainty_keV",
                                     "ame2016_estimate_keV_or_null"],
                   "training": training, "targets": targets}}


def load():
    return load_derived(DATA, DERIVED[DATA])


DRIVER = '''

def _bp_nuclear_driver(training, queries):
    _bp_seed()
    return _bp_plain(predict_masses(training, queries))
'''


def rms(values):
    values = np.asarray(values, float)
    return float(np.sqrt(np.mean(values ** 2)))


def score(module, P):
    payload = load()
    training = [dict(zip(("N", "Z", "A", "element", "mass_excess_keV", "uncertainty_keV"), row)) for row in payload["training"]]
    targets = payload["targets"]
    queries = [[row[0], row[1]] for row in targets]
    answer = run_sandboxed(module, DRIVER, "_bp_nuclear_driver", [training, queries], required=("predict_masses",),
                           cpu_seconds=int(P.get("cpu_seconds", 120)), memory_mb=2048)
    predicted = as_array(answer, shape=(len(targets),), name="predicted mass excess", low=-2.0e5, high=3.0e5)
    truth = np.array([row[4] for row in targets])
    errors = predicted - truth
    with_estimate = [i for i, row in enumerate(targets) if row[6] is not None]
    estimate_errors = [targets[i][6] - truth[i] for i in with_estimate]
    worst = int(np.argmax(np.abs(errors)))
    return rms(errors), {
        "targets": len(targets), "training_nuclides": len(training), "rms_keV": rms(errors),
        "mean_error_keV": float(errors.mean()), "max_abs_error_keV": float(np.abs(errors).max()),
        "worst_nuclide": f"{targets[worst][2]}{targets[worst][3]} (N={targets[worst][0]}, Z={targets[worst][1]})",
        "rms_keV_on_ame2016_estimated_subset": rms(errors[with_estimate]),
        "ame2016_estimates_rms_keV": rms(estimate_errors), "ame2016_estimated_subset_size": len(with_estimate)}
