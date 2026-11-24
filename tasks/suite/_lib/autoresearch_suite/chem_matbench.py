"""Matbench v0.1 materials property prediction with the official five-fold nested cross-validation.

For each of the five official folds the candidate's
``fit_predict(train_inputs, train_targets, test_inputs, info)`` runs in a
fresh child process (chem_sandbox) that receives the four training folds with
targets and the held-out fold without targets, so no target of a held-out
fold is ever visible to the call that predicts it (a single process for all
folds would leak them, because every sample is a training sample of the
other folds). The score is Matbench's mean over folds of the test MAE for
regression tasks and of the balanced accuracy for the classification task.
Classification predictions are 0/1 labels, for which the binary ROC AUC equals
the balanced accuracy; the value is comparable with the "mean balanced
accuracy" column of the Matbench leaderboard (computed for every entry, from
hard labels or from probabilities thresholded at 0.5), not with its
"probability ROC AUC" column, which needs unthresholded probabilities.
Inputs are compositions ({"formula", "composition"}) or ordered crystal
structures ({"lattice" (3x3, Angstrom), "species", "frac_coords"}).
"""
from __future__ import annotations

import numpy as np

from . import chem_sandbox
from .chem_adme import reportable
from .chem_common import average_ranks, load_json, mae, predictions, regression_metrics
from .runtime import require

TASKS = ("steels", "expt_gap", "jdft2d", "phonons", "dielectric", "log_gvrh", "perovskites", "expt_is_metal")


def roc_auc(truth, scores):
    """Area under the ROC curve (Mann-Whitney form with average ranks for ties, as scikit-learn)."""
    truth = np.asarray(truth, dtype=bool)
    positives, negatives = int(truth.sum()), int((~truth).sum())
    ranks = average_ranks(scores)
    return float((ranks[truth].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def classification_metrics(truth, scores):
    truth = np.asarray(truth, dtype=bool)
    labels = np.asarray(scores) >= 0.5
    tp, tn = int(np.sum(labels & truth)), int(np.sum(~labels & ~truth))
    fp, fn = int(np.sum(labels & ~truth)), int(np.sum(~labels & truth))
    recall, specificity = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"rocauc": roc_auc(truth, scores), "accuracy": (tp + tn) / len(truth), "balanced_accuracy": (recall + specificity) / 2.0,
            "f1": f1, "n": int(len(truth))}


def load(task):
    if task not in TASKS:
        raise ValueError(f"unknown Matbench task {task}")
    return load_json("matbench", f"{task}.json")


def matbench_score(module, task, wall_seconds_per_fold=170):
    """Mean over the five official folds of the held-out MAE (lower is better) or, for classification, of the
    balanced accuracy of the 0/1 labels (their binary ROC AUC; higher is better)."""
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
                                     "mean_balanced_accuracy": float(np.mean([f["balanced_accuracy"] for f in folds])), "folds": folds}
    fold_mae = [f["mae"] for f in folds]
    return float(np.mean(fold_mae)), reportable({
        "task": data["task"], "target": data["target"], "unit": data["unit"], "input_type": data["input_type"], "samples": n,
        "mean_mae": float(np.mean(fold_mae)), "std_mae": float(np.std(fold_mae)),
        "mean_rmse": float(np.mean([f["rmse"] for f in folds])), "max_max_error": float(max(f["max_error"] for f in folds)),
        "folds": folds})
