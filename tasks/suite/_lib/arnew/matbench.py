"""Matbench v0.1 tasks missing from autoresearch-tasks (glass, log_kvrh, mp_gap, mp_is_metal, mp_e_form).

Same protocol as the old suite's chem_matbench: the five official folds, read from Matbench's own validation file (it documents 5 splits,
shuffle, random_state 18012019; the classification folds are those of StratifiedKFold, the regression folds those of KFold), each
prediction call in a fresh sandboxed child process that sees the training folds with targets and the held-out fold without.
The score is the mean over folds of the held-out MAE (regression) or of the ROC AUC computed on 0/1 label predictions
(classification, as the old suite's chem_matbench does). The candidate returns 0/1 labels for the classification tasks, so the ROC AUC of the labels equals the
balanced accuracy, which is the metric of the leaderboard's primary classification column ("Balanced accuracy (BA)", checked on the live page 2026-11-18: glass 0.9603).
The leaderboard has a second column, "Probability ROC AUC" (glass 0.9899), which this task does not use. The data are derived from the Materials Project's
Matbench files by tasks/scripts/new_open_problems/prepare_new_data.py --family matbench into the arnew data folder (matbench/<task>.json) (the old suite's format).
"""
from __future__ import annotations

import numpy as np

from .common import candidate_sandbox, read_data, require  # noqa: F401  (importing common puts the old suite library on sys.path)
from autoresearch_suite.chem_common import mae, predictions, regression_metrics  # noqa: E402
from autoresearch_suite.chem_matbench import classification_metrics  # noqa: E402

chem_sandbox = candidate_sandbox()  # hides data/ (the labels) and this library from the candidate's child process

TASKS = {
    "glass": ("composition", "gfa", "1 = glass former, 0 = not", True),
    "log_kvrh": ("structure", "log10(K_VRH)", "log10(GPa)", False),
    "mp_gap": ("structure", "gap pbe", "eV", False),
    "mp_is_metal": ("structure", "is_metal", "1 = metal, 0 = non-metal", True),
    "mp_e_form": ("structure", "e_form", "eV/atom", False),
}


def load(task):
    import json

    return json.loads(read_data("matbench", f"{task}.json"))


def score(module, task, wall_seconds_per_fold=170):
    data = load(task)
    targets = np.asarray(data["targets"], dtype=float)
    n = len(targets)
    folds = []
    for k, test_index in enumerate(data["test_folds"]):
        held_out = set(test_index)
        train_index = [i for i in range(n) if i not in held_out]
        info = {"task": data["task"], "input_type": data["input_type"], "target": data["target"], "unit": data["unit"],
                "fold": k, "n_train": len(train_index), "n_test": len(test_index)}
        raw = chem_sandbox.run_candidate(
            module, "fit_predict",
            [[data["inputs"][i] for i in train_index], targets[train_index].tolist(), [data["inputs"][i] for i in test_index], info],
            wall_seconds=wall_seconds_per_fold)
        predicted = predictions(raw, len(test_index))
        truth = targets[test_index]
        if data["task_type"] == "classification":
            require(bool(np.all((predicted == 0.0) | (predicted == 1.0))), "classification predictions must be 0/1 labels")
            folds.append(classification_metrics(truth, predicted))
            continue
        folds.append({**regression_metrics(truth, predicted), "max_error": float(np.max(np.abs(truth - predicted))),
                      "train_mean_predictor_mae": mae(truth, np.full(len(truth), float(np.mean(targets[train_index]))))})
    if data["task_type"] == "classification":
        auc = [f["rocauc"] for f in folds]
        return float(np.mean(auc)), {"task": data["task"], "target": data["target"], "input_type": data["input_type"], "samples": n,
                                     "mean_rocauc": float(np.mean(auc)), "std_rocauc": float(np.std(auc)),
                                     "mean_balanced_accuracy": float(np.mean([f["balanced_accuracy"] for f in folds]))}
    fold_mae = [f["mae"] for f in folds]
    return float(np.mean(fold_mae)), {"task": data["task"], "target": data["target"], "unit": data["unit"], "input_type": data["input_type"],
                                      "samples": n, "mean_mae": float(np.mean(fold_mae)), "std_mae": float(np.std(fold_mae)),
                                      "mean_rmse": float(np.mean([f["rmse"] for f in folds]))}
