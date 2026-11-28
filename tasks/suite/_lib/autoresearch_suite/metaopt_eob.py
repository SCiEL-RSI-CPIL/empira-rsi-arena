"""EoB-style benchmark-function synthesis for BBO (EoB, arXiv:2601.21877, Sections 3.2.2 and 4.2), clean-room.

A candidate defines a 10-dimensional box-constrained minimisation problem
``objective(X)`` on [-5, 5]^10. Two objectives from the paper are computed:

* LSI (Eq. 3): 1 / (1 + min_p ||z_f - z_p||_2 / sigma^2), the similarity of the
  candidate's landscape features z_f to those of the 24 BBOB functions. The paper
  uses NeurELA's 16-dimensional embedding; NeurELA has no licence, so a local set
  of 16 classical exploratory-landscape-analysis features (``metaopt_ela``) is
  used instead, computed on a fixed 500-point Latin-hypercube sample; each BBOB
  function's anchor is the centroid of five random instances (see ``anchors``).
* ADC (Eq. 4): the standard deviation over a pool of 10 optimisers of their mean
  final best value over 10 seeded runs, after min-max normalisation of the whole
  10 x 10 result matrix; the pool is a local stand-in (``metaopt_eob_de``,
  ``metaopt_eob_swarm``) for the paper's ten algorithms.

The paper treats the two objectives as a bi-objective problem (MOEA/D with PBI);
the scalar score here, sqrt(LSI x min(1, ADC / 0.5)), is a suite convention (0.5 is
the largest possible population standard deviation of values in [0, 1]).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import metaopt_ela as ela
from .blackbox import BBOBFunction
from .metaopt_eob_de import ALGORITHMS as DE_ALGORITHMS, Budget
from .metaopt_eob_swarm import ALGORITHMS as SWARM_ALGORITHMS
from .runtime import CandidateError, require

DIM, LB, UB = 10, -5.0, 5.0
POOL = {**DE_ALGORITHMS, **SWARM_ALGORITHMS}
ADC_MAX = 0.5
MAX_ABS = 1e100


def _call(objective, X):
    """Evaluate the candidate on a copy of X and validate the output."""
    X = np.array(X, dtype=float, copy=True)
    with np.errstate(all="ignore"):
        y = objective(X)
    try:
        y = np.asarray(y)
    except (TypeError, ValueError):
        raise CandidateError("objective(X) must return numbers") from None
    # Real numbers only: strings of digits, booleans and complex values would otherwise be converted silently.
    require(y.dtype.kind in "iuf", f"objective(X) must return real numbers, got dtype {y.dtype}")
    y = y.astype(float)
    require(y.shape == (len(X),), f"objective(X) must return shape ({len(X)},), got {y.shape}")
    require(bool(np.all(np.isfinite(y))), "objective(X) returned a non-finite value")
    require(bool(np.all(np.abs(y) <= MAX_ABS)), f"objective(X) returned a value larger than {MAX_ABS:g} in magnitude")
    return y


def check_candidate(objective, sample):
    """Shape, finiteness, determinism, row independence and non-constancy on the feature sample."""
    y = _call(objective, sample)
    again = _call(objective, sample[::-1])[::-1]
    scale = 1e-9 * (1.0 + np.abs(y))
    require(bool(np.all(np.abs(again - y) <= scale)), "objective(X) depends on the batch order or is not deterministic")
    rows = np.concatenate([_call(objective, sample[i:i + 1]) for i in range(32)])
    require(bool(np.all(np.abs(rows - y[:32]) <= scale[:32])), "objective(X) gives different values for single rows and batches")
    require(float(np.ptp(y)) > 1e-12 * (1.0 + float(np.max(np.abs(y)))), "objective(X) is constant on the sample")
    return y


def anchors(n_samples=500, sample_seed=2026, instances=5):
    """Feature anchors of the 24 BBOB functions (suite BBOB implementation, ``instances`` random instances each).

    Features are scaled by their pooled within-function (across-instance) standard
    deviation. Each function's anchor is the centroid of its instances, so no single
    reproducible instance coincides with an anchor. sigma^2 is the median distance
    from an instance to the centroid of the other instances of its function, so
    LSI = 0.5 means "as close to a BBOB function as a typical new instance of it".
    Computed once per process and set of arguments, however the arguments are passed.
    """
    return _anchors(int(n_samples), int(sample_seed), int(instances))


@lru_cache(maxsize=4)
def _anchors(n_samples, sample_seed, instances):
    X = ela.sample(DIM, LB, UB, n_samples, sample_seed)
    F = np.empty((24, instances, len(ela.NAMES)))
    for fid in range(1, 25):
        for k in range(instances):
            f = BBOBFunction(fid, DIM, instance=k + 1)
            F[fid - 1, k] = ela.features(X, np.array([f(x) for x in X]), LB, UB)
    scale = np.maximum(np.sqrt(F.var(axis=1).mean(axis=0)), ela.STD_FLOOR)
    Z = F / scale
    centroids = Z.mean(axis=1)
    held_out = [np.linalg.norm(Z[p, k] - np.delete(Z[p], k, axis=0).mean(axis=0)) for p in range(24) for k in range(instances)]
    return {"X": X, "raw": F, "scale": scale, "Z": centroids, "sigma2": float(np.median(held_out))}


def lsi(features, ref):
    """(LSI, nearest BBOB function id, distance) for a feature vector (Eq. 3 with the unsquared distance as printed)."""
    z = np.asarray(features) / ref["scale"]
    dist = np.linalg.norm(ref["Z"] - z, axis=1)
    nearest = int(np.argmin(dist))
    return float(1.0 / (1.0 + dist[nearest] / ref["sigma2"])), nearest + 1, float(dist[nearest])


def perturbed_sample(X, module, scale=1e-5):
    """The feature sample moved by a tiny candidate-specific offset (seeded by the SHA-256 of the candidate file).

    A locally continuous function has practically the same features on both samples; a function that
    special-cases the public sample points does not, and the smaller LSI of the two counts.
    """
    path = getattr(module, "__file__", None)
    digest = hashlib.sha256(Path(path).read_bytes()).digest() if path and Path(path).is_file() else b"metaopt-eob-default"
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.clip(X + scale * (UB - LB) * rng.standard_normal(X.shape), LB, UB)


def adc(objective, runs=10, evals_per_dim=500, seed=0, algorithms=None):
    """Return (ADC, final-best matrix runs x algorithms, algorithm names)."""
    names = list(algorithms or POOL)
    obj = np.empty((runs, len(names)))
    for j, name in enumerate(names):
        for i in range(runs):
            budget = Budget(lambda X: _call(objective, X), DIM, LB, UB, evals_per_dim * DIM)
            POOL[name](budget, np.random.default_rng([seed, j, i]))
            require(np.isfinite(budget.best), f"{name} found no finite value")
            obj[i, j] = budget.best
    span = float(obj.max() - obj.min())
    if span <= 1e-12 * (1.0 + float(np.max(np.abs(obj)))):
        return 0.0, obj, names
    normalised = (obj - obj.min()) / span
    return float(np.std(normalised.mean(axis=0))), obj, names


def score(module, P):
    objective = getattr(module, "objective", None)
    require(callable(objective), "candidate must define objective(X)")
    ref = anchors(P.get("samples", 500), P.get("sample_seed", 2026))
    y = check_candidate(objective, ref["X"])
    feats = ela.features(ref["X"], y, LB, UB)
    shifted = perturbed_sample(ref["X"], module)
    y2 = _call(objective, shifted)
    require(float(np.ptp(y2)) > 0, "objective(X) is constant on the perturbed sample")
    feats2 = ela.features(shifted, y2, LB, UB)
    require(bool(np.all(np.isfinite(feats)) and np.all(np.isfinite(feats2))), "landscape features of the candidate are not finite")
    fixed, perturbed = lsi(feats, ref), lsi(feats2, ref)
    similarity, nearest, distance = min(fixed, perturbed)
    distinction, obj, names = adc(objective, P.get("runs", 10), P.get("evals_per_dim", 500), P.get("seed", 0))
    value = float(np.sqrt(similarity * min(1.0, distinction / ADC_MAX)))
    means = obj.mean(axis=0)
    details = {"lsi": similarity, "lsi_fixed_sample": fixed[0], "lsi_perturbed_sample": perturbed[0], "adc": distinction, "adc_normalised": min(1.0, distinction / ADC_MAX),
               "nearest_bbob_function": nearest, "feature_distance": distance, "sigma2": ref["sigma2"],
               "features": dict(zip(ela.NAMES, (float(v) for v in feats))),
               "pool_mean_best": dict(zip(names, (float(v) for v in means))),
               "evaluations_per_run": P.get("evals_per_dim", 500) * DIM}
    return value, details
