"""CLRS-30 data for the EvoPrompting processor-search tasks (numpy only, no TensorFlow).

``scripts/prepare_clrs.py`` converts the official CLRS-30 v1.0.0 archive
(storage.googleapis.com/dm-clrs/CLRS30_v1.0.0.tar.gz) into one compressed
``<algorithm>.npz`` per algorithm in :func:`data_dir` (``data/clrs`` of the task
cache, outside the task library). Keys are
``<split>/<stage>_<probe>`` and ``<split>/lengths`` for the splits ``train``
(1000 trajectories of 16 nodes, all hint frames), ``val`` (16 nodes) and ``test``
(64 nodes; 80 for the string matchers, 65 for optimal BST). Validation and test
keep only the first hint frame, the only one a model reads at evaluation time.
Non-scalar probes are stored as int8 (pointer indices, 0/1 and -1 for masked
entries); scalars as float32, the precision of the archive.

:func:`make_batch` applies the data options of Ibarz et al. (2022) used by
``clrs/examples/run.py``: the constant ``pred_h`` hint becomes the input ``pred``
for the algorithms in ``PRED_AS_INPUT``, sorting outputs become a permutation
matrix plus a first-node mask, and (training only) the ``pos`` input is
re-drawn as sorted uniform values per segment.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arena_task_paths import cache_root, data_root
from . import clrs_specs as S


def data_dir():
    return data_root() / "clrs"


def isolation():
    """Directories the training child must not read: the whole task cache (the prepared files hold the test
    outputs; a raw CLRS-30 archive kept there holds them too), and the default task cache (arena_task_paths
    without ARENA_TASK_CACHE_DIR) when another one is configured. Training reads nothing from them."""
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    if not base.is_absolute():
        base = Path.home() / ".cache"
    roots = {str(cache_root()), str((base / "empira" / "tasks").resolve())}
    return {"deny_read": sorted(roots), "allow_read": []}


def storage_dtype(typ):
    return np.float32 if typ == S.SCALAR else np.int8


def content_digest(arrays):
    """SHA-256 over sorted keys, dtypes, shapes and raw bytes (independent of zip compression)."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(array.dtype.str.encode())
        digest.update(repr(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load(algorithm, expected_digest=None):
    """Return ``{split: {key: array}}`` for one algorithm, optionally checking the content digest."""
    if algorithm not in S.SPECS:
        raise KeyError(f"unknown CLRS algorithm {algorithm!r}")
    path = data_dir() / f"{algorithm}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run scripts/prepare_clrs.py to build the CLRS-30 files")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if expected_digest is not None and content_digest(arrays) != expected_digest:
        raise ValueError(f"{path} does not match its locked content digest; re-run scripts/prepare_clrs.py")
    splits = {}
    for key, array in arrays.items():
        split, name = key.split("/", 1)
        splits.setdefault(split, {})[name] = array
    return splits


@dataclass
class Probe:
    name: str
    stage: str
    location: str
    type: str
    data: np.ndarray  # inputs/outputs: (B, ...); hints: (T, B, ...)


@dataclass
class Batch:
    inputs: list
    hints: list
    outputs: list
    lengths: np.ndarray  # (B,) number of hint frames of each trajectory
    nb_nodes: int


def cyclic_predecessor_and_first(pointers):
    """CLRS ``predecessor_to_cyclic_predecessor_and_first`` for a batch of predecessor arrays (B, N).

    The head of the list points to itself; it is re-pointed to the last node (the node
    no other node points to) and marked in a one-hot mask.
    """
    batch, nodes = pointers.shape
    one_hot = np.eye(nodes, dtype=np.float32)[pointers]  # (B, N, N)
    last = one_hot.sum(-2).argmin(-1)
    first = np.diagonal(one_hot, axis1=-2, axis2=-1).argmax(-1)
    mask = np.eye(nodes, dtype=np.float32)[first]  # (B, N)
    rows = np.arange(batch)
    one_hot[rows, first, first] -= 1.0
    one_hot[rows, first, last] += 1.0
    return one_hot, mask


def random_pos(pos, rng):
    """CLRS ``process_random_pos``: sorted uniform values inside each segment that starts at pos == 0."""
    batch, nodes = pos.shape
    unsorted = rng.uniform(size=(batch, nodes))
    new = np.empty_like(pos, dtype=np.float32)
    for i in range(batch):
        split = np.concatenate([np.flatnonzero(pos[i] == 0), [nodes]])
        new[i] = np.concatenate([np.sort(unsorted[i, split[j]:split[j + 1]]) for j in range(len(split) - 1)])
    return new


def _model_data(typ, array):
    if typ == S.POINTER:
        return array.astype(np.int64)
    return array.astype(np.float32)


def make_batch(split, algorithm, indices, rng=None, with_outputs=True):
    """Build a model batch from one prepared split.

    ``indices`` selects trajectories; hints are trimmed to the longest selected
    trajectory (splits without full hints keep their single first frame). ``rng``
    (a ``numpy.random.RandomState``) enables the randomised ``pos`` input.
    ``with_outputs=False`` skips the output probes (the sandboxed child never
    receives the test outputs).
    """
    indices = np.asarray(indices, dtype=np.int64)
    lengths = split["lengths"][indices].astype(np.int64)
    horizon = int(lengths.max())
    pred_as_input = algorithm in S.PRED_AS_INPUT
    inputs, hints, outputs = [], [], []
    nb_nodes = split["input_pos"].shape[1]
    for name, (stage, loc, typ) in S.SPECS[algorithm].items():
        if stage == S.HINT:
            data = split["hint_" + name][indices]  # (B, T_stored, ...)
            if name == "pred_h" and pred_as_input:
                inputs.append(Probe("pred", S.INPUT, loc, S.POINTER, _model_data(S.POINTER, data[:, 0])))
                continue
            data = np.swapaxes(data[:, :horizon], 0, 1)  # (T, B, ...)
            hints.append(Probe(name, stage, loc, typ, _model_data(typ, data)))
        elif stage == S.INPUT:
            data = split["input_" + name][indices]
            if name == "pos" and rng is not None:
                data = random_pos(data, rng)
            inputs.append(Probe(name, stage, loc, typ, _model_data(typ, data)))
        elif with_outputs:
            data = split["output_" + name][indices]
            if typ == S.SHOULD_BE_PERMUTATION:
                perm, first = cyclic_predecessor_and_first(data.astype(np.int64))
                outputs.append(Probe(name, stage, loc, S.PERMUTATION_POINTER, perm))
                outputs.append(Probe(name + "_mask", stage, loc, S.MASK_ONE, first))
            else:
                outputs.append(Probe(name, stage, loc, typ, _model_data(typ, data)))
    return Batch(inputs, hints, outputs, lengths, nb_nodes)


def spread(count, limit):
    """Deterministic evenly spaced subset of ``range(count)`` with at most ``limit`` items."""
    if limit is None or limit >= count:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, int(limit)).round().astype(np.int64))


def length_batches(lengths, indices, batch_size):
    """Group ``indices`` into batches of similar trajectory length (less padding at evaluation)."""
    order = np.asarray(indices)[np.argsort(lengths[indices], kind="stable")]
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
