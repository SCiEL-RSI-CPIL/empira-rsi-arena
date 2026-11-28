"""ZAPBench data layout for the jax_station ZAPBench task: condition-aware splits and window indexing.

ZAPBench (Lueckmann et al., ICLR 2025; github.com/google-research/zapbench,
Apache-2.0; data CC-BY-4.0) provides whole-brain calcium traces of a larval
zebrafish: 7,879 timesteps x 71,721 neurons recorded under nine stimulus
conditions. The split rules below restate zapbench/constants.py and
zapbench/data_utils.py (get_condition_bounds, adjust_condition_bounds_for_split)
at commit 7347b3b2; scripts/prepare_jax_station.py checks them against the
pinned source. The Station's ZAPBench task (dualverse-ai/station,
example/research_misc/zapbench/misc/process_data.py) builds its train, validation
and test arrays from these bounds with a 4-step context: per training condition
the first 70 % is training data, the next 10 % (plus 4 context steps) validation
and the last 20 % (plus 4 context steps) test; the held-out condition 3 (taxis)
contributes test data only, starting 256 - 4 steps after its padded start.
Windows (4 input steps + 32 target steps, stride 1) never cross conditions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .jax_station_common import data_dir

CONDITION_OFFSETS = (0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879)
CONDITION_PADDING = 1
CONDITIONS_TRAIN = (0, 1, 2, 4, 5, 6, 7, 8)
CONDITIONS_HOLDOUT = (3,)
CONDITION_NAMES = ("gain", "dots", "flash", "taxis", "turning", "position", "open loop", "rotation", "dark")
VAL_FRACTION, TEST_FRACTION = 0.1, 0.2
MAX_CONTEXT_LENGTH = 256
CONTEXT, HORIZON = 4, 32
NUM_TIMESTEPS, NUM_NEURONS = 7879, 71721
ZARR_URL = "https://storage.googleapis.com/zapbench-release/volumes/20240930/traces/"
CHUNK = 512


def zap_dir():
    return data_dir() / "zapbench"


def condition_bounds(condition):
    """zapbench.data_utils.get_condition_bounds: (inclusive_min, exclusive_max) without padding steps."""
    return (CONDITION_OFFSETS[condition] + CONDITION_PADDING, CONDITION_OFFSETS[condition + 1] - CONDITION_PADDING)


def split_bounds(split, inclusive_min, exclusive_max, context=CONTEXT):
    """zapbench.data_utils.adjust_condition_bounds_for_split for train, val, test and test_holdout."""
    total = exclusive_max - inclusive_min
    num_test, num_val = int(total * TEST_FRACTION), int(total * VAL_FRACTION)
    num_train = total - num_test - num_val
    if split == "train":
        exclusive_max = inclusive_min + num_train
    elif split == "val":
        exclusive_max = inclusive_min + num_train + num_val
        inclusive_min += num_train - context
    elif split == "test":
        inclusive_min += num_train + num_val - context
    elif split == "test_holdout":
        inclusive_min += MAX_CONTEXT_LENGTH - context
    else:
        raise ValueError(f"unknown split {split!r}")
    assert 0 <= inclusive_min < exclusive_max
    return inclusive_min, exclusive_max


def segments(split):
    """[(condition, start, stop)] of one split in the order of the Station's process_data.py."""
    if split in ("train", "val"):
        return [(c, *split_bounds(split, *condition_bounds(c))) for c in CONDITIONS_TRAIN]
    if split == "test":
        parts = [(c, *split_bounds("test", *condition_bounds(c))) for c in CONDITIONS_TRAIN]
        return parts + [(c, *split_bounds("test_holdout", *condition_bounds(c))) for c in CONDITIONS_HOLDOUT]
    raise ValueError(f"unknown split {split!r}")


def window_starts(labels, context=CONTEXT, horizon=HORIZON):
    """Start indices of all stride-1 windows of context + horizon steps that stay within one condition."""
    labels = np.asarray(labels)
    total = context + horizon
    if len(labels) < total:
        return np.zeros(0, dtype=np.int64)
    view = np.lib.stride_tricks.sliding_window_view(labels, total)
    return np.flatnonzero(np.all(view == view[:, :1], axis=1)).astype(np.int64)


def expected_counts():
    """Timesteps and windows per split implied by the rules above (used by tests and the prepare script)."""
    out = {}
    for split in ("train", "val", "test"):
        parts = segments(split)
        steps = sum(stop - start for _, start, stop in parts)
        windows = sum(max(0, stop - start - CONTEXT - HORIZON + 1) for _, start, stop in parts)
        out[split] = {"timesteps": steps, "windows": windows}
    return out


FILES = {
    "train": ("trainval/train.npy", "trainval/train_conditions.npy"),
    "val": ("trainval/val.npy", "trainval/val_conditions.npy"),
    "test": ("test/test.npy", "test/test_conditions.npy"),
}


def load_split(split, root=None, mmap=True):
    """(data [T, 71721], condition labels [T]) of a prepared split; data float16 (train/val) or float32 (test)."""
    root = Path(root) if root else zap_dir()
    data_name, label_name = FILES[split]
    path = root / data_name
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run `python scripts/prepare_jax_station.py --zapbench`")
    data = np.load(path, mmap_mode="r" if mmap else None)
    labels = np.load(root / label_name)
    if data.ndim != 2 or data.shape[1] != NUM_NEURONS or len(labels) != data.shape[0]:
        raise ValueError(f"unexpected prepared array shapes in {root}")
    return data, labels


def manifest(root=None):
    path = (Path(root) if root else zap_dir()) / "manifest.json"
    return json.loads(path.read_text()) if path.is_file() else None
