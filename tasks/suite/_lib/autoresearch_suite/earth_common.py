"""Shared helpers for the earth research tasks: prepared data, validation and scores.

Evaluators never touch the network. ``scripts/prepare_earth.py`` downloads the
pinned public sources, derives small files with the ``derive`` functions of the
task modules and writes them below ``data_dir()``. Every derived file is checked
against the SHA-256 recorded in its task module before it is used.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math

import numpy as np

from arena_task_paths import data_root
from .runtime import CandidateError, require


GZIP_OS_BYTE, RECORDED_OS_CODE = 9, 19


def data_dir():
    return data_root() / "earth"


def canonical_bytes(payload):
    """Deterministic gzip-compressed JSON (sorted keys, no timestamp in the gzip header).

    With mtime=0 zlib writes the header and puts its build's operating-system code in byte 9
    (3 on Linux, 19 on macOS). The recorded digests come from files carrying 19, so that value
    is written on every platform; no other byte depends on the platform.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    data = bytearray(gzip.compress(text.encode(), compresslevel=9, mtime=0))
    data[GZIP_OS_BYTE] = RECORDED_OS_CODE
    return bytes(data)


def load_derived(name, sha256):
    """Read one prepared file, refusing missing or changed bytes."""
    path = data_dir() / name
    if not path.is_file():
        raise RuntimeError(f"missing prepared data {path}; run python3 scripts/prepare_earth.py")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"prepared data {name} has SHA-256 {digest}, expected {sha256}; rerun scripts/prepare_earth.py")
    return json.loads(gzip.decompress(raw))


def frozen(values, dtype=float):
    """A read-only array handed to candidate code."""
    array = np.array(values, dtype=dtype)
    array.setflags(write=False)
    return array


def as_array(value, shape=None, name="output", low=None, high=None):
    """Validate candidate numbers: real, finite, expected shape and optional bounds."""
    try:
        raw = np.asarray(value)
    except Exception as error:  # noqa: BLE001 - any conversion failure is a contract violation
        raise CandidateError(f"{name}: not convertible to an array ({error})") from None
    require(raw.dtype.kind in "fiub" and not np.iscomplexobj(raw), f"{name}: expected real numbers, got dtype {raw.dtype}")
    array = raw.astype(float)
    if shape is not None:
        require(array.shape == tuple(shape), f"{name}: expected shape {tuple(shape)}, got {array.shape}")
    require(np.isfinite(array).all(), f"{name}: contains NaN or infinite values")
    if low is not None:
        require(np.all(array >= low), f"{name}: values below {low}")
    if high is not None:
        require(np.all(array <= high), f"{name}: values above {high}")
    return array


def haversine_km(lat1, lon1, lat2, lon2, radius=6371.0):
    """Great-circle distance in km on a sphere of the given radius."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def rmse(truth, prediction):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def nse(truth, prediction):
    """Nash-Sutcliffe efficiency (1 is perfect, 0 equals the observed mean)."""
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    require(denominator > 0, "reference series has no variance")
    return float(1.0 - np.sum((prediction - truth) ** 2) / denominator)


def kge(truth, prediction):
    """Kling-Gupta efficiency (Gupta et al. 2009)."""
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    require(truth.std() > 0 and truth.mean() != 0, "reference series has no variance or zero mean")
    r = 0.0 if prediction.std() == 0 else float(np.corrcoef(truth, prediction)[0, 1])
    alpha = float(prediction.std() / truth.std())
    beta = float(prediction.mean() / truth.mean())
    return float(1.0 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def correlation(truth, prediction):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.std() == 0 or prediction.std() == 0:
        return 0.0
    return float(np.corrcoef(truth, prediction)[0, 1])


def roc_auc(labels, scores):
    """Area under the ROC curve with tied scores counted as one half (Mann-Whitney)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, float)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    require(positives > 0 and negatives > 0, "AUC needs both presences and absences")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):  # average ranks over ties
        stop = start
        while stop + 1 < len(scores) and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * (start + stop) + 1
        start = stop + 1
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def gaussian_crps(truth, mean, sd):
    """Closed-form CRPS of a normal predictive distribution (Gneiting et al. 2005)."""
    truth, mean, sd = (np.asarray(v, float) for v in (truth, mean, sd))
    z = (truth - mean) / sd
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    return sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def call_candidate(function, *args, name="candidate"):
    """Call candidate code; contract violations stay CandidateErrors, other crashes are reported plainly."""
    require(callable(function), f"{name} is not callable")
    return function(*args)
