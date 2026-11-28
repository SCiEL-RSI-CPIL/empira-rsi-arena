"""EvoPrompting CLRS-30 processor search: score a candidate processor out of distribution.

``score(module, P)`` is the body of the ``clrs_processor`` family. ``module`` is
the candidate as ``sandboxed.run`` hands it over: not executed, carrying only the
path of its file, so no candidate code runs in this process. It checks the
candidate source statically, loads the locked CLRS-30 split of
``P['algorithm']`` (clrs_data), and runs the training in a restricted child
process (clrs_sandbox / clrs_worker) that receives the train split, the
validation subset and only the inputs and first hint frames of the test
subset. The child trains the CLRS network around ``build_processor`` for the
fixed budget, keeps the best validation checkpoint and returns hard test
predictions; this process validates them and computes the CLRS test score
(clrs_metrics) against the held-out test outputs. The child cannot read the
task cache where the kernel permits hiding it (``os_sandbox`` in the details:
``linux-namespaces``, ``sandbox-exec`` or ``audit-hook-only``, see clrs_sandbox).
With several ``train_seeds`` the score is the mean over independent training runs.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import clrs_data as D
from . import clrs_metrics as M
from . import clrs_sandbox
from . import sandboxed
from .runtime import CandidateError, require

BUDGET_KEYS = ("hidden", "steps", "batch_size", "learning_rate", "grad_clip", "eval_every", "eval_batch_size",
               "max_processor_parameters", "checkpoint_steps", "train_threads")


def visible_splits(data, val_indices, test_indices):
    """What the training process may see: train, a validation subset and the test inputs/first hints only."""
    val = {key: value[val_indices] for key, value in data["val"].items()}
    test = {key: value[test_indices] for key, value in data["test"].items() if not key.startswith("output_")}
    return {"train": data["train"], "val": val, "test": test}


def score(module, P):
    algorithm = P["algorithm"]
    started = time.monotonic()
    path = getattr(module, "__file__", None)
    require(bool(path) and Path(path).is_file(), "candidate module has no source file")
    source = Path(path).read_text(encoding="utf-8")
    clrs_sandbox.check_source(source)
    require(sandboxed.defines(module, "build_processor"), "the program must define build_processor(hidden_dim)")
    data = D.load(algorithm, expected_digest=P["data_sha256"])
    val_indices = D.spread(len(data["val"]["lengths"]), P.get("val_samples"))
    test_indices = D.spread(len(data["test"]["lengths"]), P.get("test_samples"))
    splits = visible_splits(data, val_indices, test_indices)
    truths = M.truth_arrays(algorithm, data["test"], test_indices)
    nodes = int(data["test"]["input_pos"].shape[1])
    budget = {key: P[key] for key in BUDGET_KEYS if key in P}
    seeds = list(P.get("train_seeds", [0]))
    wall = float(P.get("wall_seconds", 1700))  # safety net only: the score depends on the step budget, not on time
    runs, scores = [], []
    for position, seed in enumerate(seeds):
        remaining = wall - (time.monotonic() - started)
        if remaining <= 5:
            raise CandidateError(f"training runs exceeded the {wall:.0f} s wall-clock budget before seed {seed}")
        predictions, details, peak = clrs_sandbox.run_training(
            source, algorithm, splits, dict(budget, seed=int(seed)), wall_seconds=remaining / (len(seeds) - position),
            memory_mb=int(P.get("memory_mb", 4096)), threads=int(P.get("threads", 4)), filename=Path(path).name)
        checked = M.check_predictions(algorithm, predictions, truths, nodes)
        value, per_output = M.score(algorithm, checked, truths)
        scores.append(value)
        runs.append({"seed": int(seed), "test_score": round(value, 6), "test_scores_by_output": {k: round(v, 6) for k, v in per_output.items()},
                     "peak_rss_mb": round(peak, 1), **{k: v for k, v in details.items() if k in RUN_DETAILS}})
    value = float(np.mean(scores))
    first = runs[0]
    return value, {
        "algorithm": algorithm, "test_score": value, "test_score_by_seed": [run["test_score"] for run in runs],
        "processor_parameters": first.get("processor_parameters"), "model_parameters": first.get("model_parameters"),
        "edge_embeddings": first.get("edge_embeddings"), "train_nodes": int(data["train"]["input_pos"].shape[1]),
        "test_nodes": nodes, "val_samples": int(len(val_indices)), "test_samples": int(len(test_indices)),
        "steps": P.get("steps"), "os_sandbox": first.get("os_sandbox"), "runs": runs,
    }


RUN_DETAILS = ("best_val_score", "best_val_step", "val_curve", "final_train_loss", "processor_parameters",
               "model_parameters", "edge_embeddings", "train_seconds", "val_seconds", "test_seconds", "os_sandbox")
