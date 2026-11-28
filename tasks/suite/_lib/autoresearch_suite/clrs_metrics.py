"""CLRS output metrics on compact predictions (numpy only; clrs/_src/evaluation.py of dm-clrs 2.0.3).

Predictions travel between the training process and the scorer in a compact
integer form, one array per output probe of the original CLRS spec:

``pointer``                node: (B, N) node indices; edge: (B, N, N) indices
``should_be_permutation``  (B, N) predecessor indices, the first node pointing to
                           itself (CLRS ``fuse_perm_and_mask`` of the hard
                           permutation matrix and its first-node mask)
``mask``                   0/1 array of the probe's shape: (B,), (B, N) or (B, N, N)
``mask_one``               (B,) index of the selected node
``categorical``            class indices, the probe's shape without the class axis

``score`` is the CLRS test score: the mean over output probes of pointer
accuracy, mask F1 (entries equal to -1 in the truth are excluded) or one-hot
accuracy (rows containing -1 excluded), each micro-averaged over all samples.
"""
from __future__ import annotations

import numpy as np

from . import clrs_specs as S
from .runtime import CandidateError

COMPACT_DTYPES = {S.POINTER: np.int64, S.SHOULD_BE_PERMUTATION: np.int64, S.MASK: np.int8,
                  S.MASK_ONE: np.int64, S.CATEGORICAL: np.int64}


def outputs(algorithm):
    """[(name, location, type)] of the original CLRS output probes, in spec order."""
    return [(name, loc, typ) for name, (stage, loc, typ) in S.SPECS[algorithm].items() if stage == S.OUTPUT]


def truth_arrays(algorithm, split, indices=None):
    """Stored ground-truth outputs of a prepared split (optionally a subset of samples)."""
    result = {}
    for name, _, _ in outputs(algorithm):
        array = split["output_" + name]
        result[name] = array if indices is None else array[np.asarray(indices)]
    return result


def expected_shape(algorithm, name, truth):
    """Shape of the compact prediction for one output probe, given its stored truth array."""
    typ = S.SPECS[algorithm][name][2]
    if typ in (S.MASK_ONE, S.CATEGORICAL):
        return tuple(truth.shape[:-1])
    return tuple(truth.shape)


def class_count(algorithm, name, truth, nodes):
    """Exclusive upper bound of the values of a compact prediction."""
    typ = S.SPECS[algorithm][name][2]
    if typ in (S.POINTER, S.SHOULD_BE_PERMUTATION):
        return nodes
    if typ == S.MASK:
        return 2
    return int(truth.shape[-1])


def check_predictions(algorithm, predictions, truths, nodes):
    """Validate compact predictions (names, shapes, integer dtype, value range); return int64 arrays."""
    if not isinstance(predictions, dict):
        raise CandidateError("predictions must be a dict of arrays")
    names = [name for name, _, _ in outputs(algorithm)]
    if set(predictions) != set(names):
        raise CandidateError(f"predictions must cover exactly the outputs {names}, got {sorted(map(str, predictions))}")
    checked = {}
    for name in names:
        array = predictions[name]
        truth = truths[name]
        shape = expected_shape(algorithm, name, truth)
        if not isinstance(array, np.ndarray) or array.dtype.kind not in "iu":
            raise CandidateError(f"prediction {name!r} must be an integer array")
        if tuple(array.shape) != shape:
            raise CandidateError(f"prediction {name!r} has shape {tuple(array.shape)}, expected {shape}")
        upper = class_count(algorithm, name, truth, nodes)
        if array.size and (int(array.min()) < 0 or int(array.max()) >= upper):
            raise CandidateError(f"prediction {name!r} has values outside [0, {upper})")
        checked[name] = array.astype(np.int64)
    return checked


def mask_f1(pred, truth):
    """CLRS _mask_fn: F1 of 0/1 predictions against a 0/1 truth with -1 entries excluded."""
    valid = truth != S.MASKED
    positive, actual = (pred > 0.5) & valid, (truth > 0.5) & valid
    tp = float(np.sum(positive & actual))
    fp = float(np.sum(positive & (truth < 0.5) & valid))
    fn = float(np.sum(~(pred > 0.5) & actual))
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def one_hot_accuracy(pred_index, truth):
    """CLRS _eval_one: argmax accuracy over rows of the one-hot truth that contain no -1 entry."""
    valid = np.all(truth != S.MASKED, axis=-1)
    correct = (pred_index == np.argmax(truth, axis=-1)) & valid
    count = float(np.sum(valid))
    return float(np.sum(correct)) / count if count else 0.0


def probe_score(typ, pred, truth):
    if typ in (S.POINTER, S.SHOULD_BE_PERMUTATION):
        return float(np.mean(pred == truth.astype(np.int64)))
    if typ == S.MASK:
        return float(mask_f1(pred, truth))
    if typ in (S.MASK_ONE, S.CATEGORICAL):
        return one_hot_accuracy(pred, truth)
    raise ValueError(f"no CLRS metric for output type {typ}")


def score(algorithm, predictions, truths):
    """(mean CLRS score over output probes, {probe: score}) for validated compact predictions."""
    per_output = {name: probe_score(typ, predictions[name], truths[name]) for name, _, typ in outputs(algorithm)}
    return sum(per_output.values()) / len(per_output), per_output


def fuse_permutation(perm_index, first_index):
    """CLRS fuse_perm_and_mask on indices: the first node points to itself, the others follow the permutation."""
    nodes = perm_index.shape[-1]
    first = np.arange(nodes)[None, :] == np.asarray(first_index)[:, None]
    return np.where(first, np.arange(nodes)[None, :], perm_index)
