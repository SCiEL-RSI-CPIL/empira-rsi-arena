"""Shared helpers of the chemistry and materials research collection (chem_*.py).

Every prepared data path is resolved through ``data_dir()``; the files are
written by ``scripts/prepare_chemistry.py`` and verified against
``catalog/chemistry-data-lock.json``. Evaluators never use the network.

Prediction tasks hand the candidate only the inputs it may see (training
inputs and labels, test inputs without labels) through ``chem_sandbox`` and
score the returned predictions here against labels the candidate never
receives.
"""
from __future__ import annotations

import json
import math

import numpy as np

from arena_task_paths import data_root
from .runtime import CandidateError


def data_dir():
    return data_root() / "chemistry"


def data_file(*parts):
    """Path of one prepared file; a clear error when the data were not prepared."""
    path = data_dir().joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(f"missing prepared data file {path}; run python3 scripts/prepare_chemistry.py")
    return path


def load_json(*parts):
    return json.loads(data_file(*parts).read_text(encoding="utf-8"))


def predictions(value, n, name="predictions"):
    """Validate a candidate's predictions: n finite real numbers."""
    if isinstance(value, dict) or isinstance(value, (str, bytes)):
        raise CandidateError(f"{name} must be a list of {n} numbers, got {type(value).__name__}")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise CandidateError(f"{name} must be a list of {n} numbers") from None
    if array.shape != (n,):
        raise CandidateError(f"{name} must have shape ({n},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise CandidateError(f"{name} contain NaN or infinite values")
    return array


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def r2(y, p):
    """Coefficient of determination 1 - SS_res / SS_tot (scikit-learn's r2_score)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    total = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / total) if total > 0 else float("nan")


def pearson(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if np.std(y) == 0 or np.std(p) == 0:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def average_ranks(x):
    """Average ranks (ties share their mean rank), as in scipy.stats.rankdata."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sorted_x = x[order]
    start = 0
    while start < len(x):
        stop = start
        while stop + 1 < len(x) and sorted_x[stop + 1] == sorted_x[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def spearman(y, p):
    return pearson(average_ranks(y), average_ranks(p))


def regression_metrics(y, p):
    return {"mae": mae(y, p), "rmse": rmse(y, p), "r2": r2(y, p), "pearson_r": pearson(y, p),
            "spearman_rho": spearman(y, p), "n": int(len(y))}


def finite_or_none(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None
