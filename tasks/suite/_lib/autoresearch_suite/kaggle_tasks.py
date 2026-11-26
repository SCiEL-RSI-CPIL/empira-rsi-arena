"""Evaluators of the MLE-bench competitions recreated from public, non-Kaggle copies of their data.

Each function ``<task>(module, P) -> (value, details)`` runs the candidate's
``fit_predict(train, test_inputs, time_budget_s)`` in a sandboxed child (kaggle_sandbox) that sees only the
task's public files, validates the returned predictions against the held-out answers' shape and
recomputes the competition metric exactly as MLE-bench's grade.py does (openai/mle-bench at 507f92e1).
The time budget ``P['budget_s']`` is the same for every candidate and is enforced by killing the child.
``module`` only has to carry the candidate's path (``sandboxed.source_program``): the file is executed in
the child, never in the evaluator's process.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import kaggle_data as data
from . import sandboxed
from .kaggle_sandbox import run_fit_predict
from .runtime import CandidateError, require

JIGSAW_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
OPENVACCINE_SCORED = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
NOMAD_TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]


def _run(module, task, job, P, output="array"):
    path = getattr(module, "__file__", None)
    require(path is not None, "candidate module has no source file")
    require(sandboxed.defines(module, "fit_predict"), "candidate must define fit_predict(train, test_inputs, time_budget_s)")
    return run_fit_predict(path, job, budget_s=float(P["budget_s"]), memory_mb=int(P.get("memory_mb", 3500)), output=output,
                           **data.isolation(task))


def _matrix(arrays, rows, cols, name):
    require(len(arrays) == 1, f"{name}: expected one array")
    value = arrays[0]
    if cols == 1 and value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    shape = (rows,) if cols == 1 else (rows, cols)
    require(value.shape == shape, f"{name}: predictions must have shape {shape}, got {value.shape}")
    require(bool(np.isfinite(value).all()), f"{name}: predictions must be finite")
    return value


def _spec(task, name, kind="csv", **extra):
    return {"kind": kind, "path": str(data.public_dir(task) / name), **extra}


# --- Jigsaw toxic comment classification ---------------------------------------------------------------

def jigsaw(module, P):
    """Mean column-wise ROC AUC over the six labels on the scored test rows (MLE-bench grade.py)."""
    from sklearn.metrics import roc_auc_score
    (labels_path,) = data.require_files("jigsaw", public=("train.csv", "test.csv"), private=("test_labels.csv",))[2:]
    answers = pd.read_csv(labels_path, dtype={"id": str})
    text = {"id": "str", "comment_text": "str"}
    job = {"train": _spec("jigsaw", "train.csv", dtype=text), "test": _spec("jigsaw", "test.csv", dtype=text)}
    arrays, run = _run(module, "jigsaw", job, P)
    scores = _matrix(arrays, len(answers), len(JIGSAW_CLASSES), "jigsaw")
    truth = answers[JIGSAW_CLASSES].to_numpy()
    keep = truth.sum(axis=1) >= 0  # every prepared row is scored; kept for parity with grade.py
    per_class = [float(roc_auc_score(truth[keep, j], scores[keep, j])) for j in range(len(JIGSAW_CLASSES))]
    value = float(roc_auc_score(truth[keep], scores[keep], average="macro"))
    return value, {"metric": "mean column-wise ROC AUC", "per_class_auc": dict(zip(JIGSAW_CLASSES, per_class)),
                   "test_rows": int(keep.sum()), **run}


# --- Denoising dirty documents (NoisyOffice analogue) ----------------------------------------------------

def noisyoffice(module, P):
    """Root mean squared error over every pixel of every test image, intensities in [0, 1] (MLE-bench grade.py)."""
    data.require_files("noisyoffice", public=("train_noisy.npz", "train_clean.npz", "test_noisy.npz"), private=("test_clean.npz",))
    with np.load(data.private_dir("noisyoffice") / "test_clean.npz", allow_pickle=False) as stored:
        truth = [stored[f"a{i}"].astype(np.float64) / 255.0 for i in range(len(stored.files))]
    job = {"train": {"kind": "bundle", "parts": {"noisy": _spec("noisyoffice", "train_noisy.npz", "npz_list"),
                                                 "clean": _spec("noisyoffice", "train_clean.npz", "npz_list")}},
           "test": {"kind": "bundle", "parts": {"noisy": _spec("noisyoffice", "test_noisy.npz", "npz_list")}}}
    arrays, run = _run(module, "noisyoffice", job, P, output="array_list")
    require(len(arrays) == len(truth), f"expected {len(truth)} cleaned images, got {len(arrays)}")
    squared, pixels, per_image = 0.0, 0, []
    for k, (pred, true) in enumerate(zip(arrays, truth)):
        require(pred.shape == true.shape, f"image {k}: shape {pred.shape} differs from the input shape {true.shape}")
        require(bool(np.isfinite(pred).all()), f"image {k}: values must be finite")
        error = float(np.sum((pred - true) ** 2))
        squared, pixels = squared + error, pixels + true.size
        per_image.append(math.sqrt(error / true.size))
    value = math.sqrt(squared / pixels)
    return value, {"metric": "pixel RMSE", "test_images": len(truth), "test_pixels": pixels,
                   "per_image_rmse": [round(v, 6) for v in per_image], **run}


# --- Histopathologic cancer detection (PatchCamelyon analogue) -------------------------------------------

def pcam(module, P):
    """ROC AUC of the predicted tumour probabilities; probabilities must lie in [0, 1] (MLE-bench grade.py)."""
    from sklearn.metrics import roc_auc_score
    data.require_files("pcam", public=("train.npz", "test.npz"), private=("test_labels.npz",))
    with np.load(data.private_dir("pcam") / "test_labels.npz", allow_pickle=False) as stored:
        truth = stored["labels"].astype(int)
    job = {"train": _spec("pcam", "train.npz", "npz"), "test": _spec("pcam", "test.npz", "npz")}
    arrays, run = _run(module, "pcam", job, P)
    scores = _matrix(arrays, len(truth), 1, "pcam")
    require(scores.min() >= 0.0 and scores.max() <= 1.0, "predictions must be probabilities in [0, 1]")
    value = float(roc_auc_score(truth, scores))
    return value, {"metric": "ROC AUC", "test_patches": int(len(truth)), "test_positive_fraction": round(float(truth.mean()), 4), **run}


# --- NOMAD 2018 transparent conductors (relaxed-geometry analogue) ----------------------------------------

def nomad(module, P):
    """Mean of the RMSLEs of formation energy and band gap (MLE-bench grade.py, scikit-learn's RMSLE)."""
    from sklearn.metrics import root_mean_squared_log_error
    (labels_path,) = data.require_files("nomad", public=("train.jsonl", "test.jsonl"), private=("test_labels.csv",))[2:]
    answers = pd.read_csv(labels_path)
    job = {"train": _spec("nomad", "train.jsonl", "jsonl"), "test": _spec("nomad", "test.jsonl", "jsonl")}
    arrays, run = _run(module, "nomad", job, P)
    pred = _matrix(arrays, len(answers), 2, "nomad")
    errors = {}
    for j, column in enumerate(NOMAD_TARGETS):
        try:
            errors[column] = float(root_mean_squared_log_error(answers[column].to_numpy(), pred[:, j]))
        except ValueError as error:  # scikit-learn rejects values it cannot take log1p of
            raise CandidateError(f"{column}: {error}") from None
    value = (errors[NOMAD_TARGETS[0]] + errors[NOMAD_TARGETS[1]]) / 2
    return value, {"metric": "mean column-wise RMSLE", "rmsle": errors, "test_materials": int(len(answers)), **run}


# --- OpenVaccine mRNA degradation -------------------------------------------------------------------------

def openvaccine(module, P):
    """MCRMSE over reactivity, deg_Mg_pH10 and deg_Mg_50C on the first seq_scored positions (MLE-bench grade.py)."""
    from sklearn.metrics import root_mean_squared_error
    data.require_files("openvaccine", public=("train.jsonl", "test.jsonl"), private=("test_labels.npz",))
    with np.load(data.private_dir("openvaccine") / "test_labels.npz", allow_pickle=False) as stored:
        truth = stored["targets"].astype(np.float64)  # (n_test, seq_scored, 3)
    job = {"train": _spec("openvaccine", "train.jsonl", "jsonl"), "test": _spec("openvaccine", "test.jsonl", "jsonl")}
    arrays, run = _run(module, "openvaccine", job, P)
    require(len(arrays) == 1, "expected one array")
    pred = arrays[0]
    require(pred.shape == truth.shape, f"predictions must have shape {truth.shape} (sequences, seq_scored, 3), got {pred.shape}")
    require(bool(np.isfinite(pred).all()), "predictions must be finite")
    errors = {name: float(root_mean_squared_error(truth[:, :, j].ravel(), pred[:, :, j].ravel()))
              for j, name in enumerate(OPENVACCINE_SCORED)}
    value = float(np.mean(list(errors.values())))
    return value, {"metric": "MCRMSE", "rmse": errors, "test_sequences": int(truth.shape[0]), "scored_positions": int(truth.shape[1]), **run}


# --- U.S. patent phrase-to-phrase matching ----------------------------------------------------------------

def patent(module, P):
    """Pearson correlation between predicted and annotated similarity scores (MLE-bench grade.py)."""
    from scipy.stats import pearsonr
    (labels_path,) = data.require_files("patent", public=("train.csv", "test.csv"), private=("test_labels.csv",))[2:]
    answers = pd.read_csv(labels_path, dtype={"id": str})
    text = {"id": "str", "anchor": "str", "target": "str", "context": "str"}
    job = {"train": _spec("patent", "train.csv", dtype=text), "test": _spec("patent", "test.csv", dtype=text)}
    arrays, run = _run(module, "patent", job, P)
    pred = _matrix(arrays, len(answers), 1, "patent")
    require(not bool((pred == pred[0]).all()), "constant predictions have no Pearson correlation")  # exact; np.std overflows
    value = float(pearsonr(answers["score"].to_numpy(dtype=float), pred)[0])
    require(math.isfinite(value), "Pearson correlation is undefined for these predictions")
    return value, {"metric": "Pearson r", "test_pairs": int(len(answers)), **run}
