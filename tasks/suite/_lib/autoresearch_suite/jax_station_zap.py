"""The Station's ZAPBench neural-activity forecasting task as a reduced-budget CPU proxy.

The Station (dualverse-ai/station@782088e5, example/research_misc/zapbench,
Apache-2.0) asks for a JAX/Flax forecaster of whole-brain zebrafish activity
(ZAPBench, 71,721 neurons): 4 context steps in, 32 predicted steps out. The
candidate supplies the Station's hooks (create_network, compute_loss,
create_optimizer, _define_hyperparameters, BATCH_SIZE; omitted hooks fall back
to the Station's defaults). Evaluation runs in two child processes:

1. jax_station_zap_train.py trains the candidate with the Station's training
   loop on the ZAPBench training split, with early stopping on the validation
   split, under this task's reduced budget (one seed, at most ``max_epochs``
   epochs and ``time_cap_seconds`` of training on CPU). It can read only the
   training and validation arrays and writes the best parameters and the
   exported single-window predictor to a private directory.
2. jax_station_zap_score.py runs the exported predictor (no candidate Python
   code), one test window per call, and computes the ZAPBench test MAE (mean
   over the 9 test conditions of the MAE over all windows, the 32 steps and all
   neurons).

Data are prepared by ``python scripts/prepare_jax_station.py --zapbench`` (run
from the task library root). jax.export serialisation needs the package
flatbuffers; both children check for it and report its absence as a defect of
the environment, not of the candidate.
"""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

from . import jax_station_zapdata as zd
from .jax_station_common import ChildFailure, run_child
from .runtime import CandidateError

TRAIN_WORKER = Path(__file__).resolve().with_name("jax_station_zap_train.py")
SCORE_WORKER = Path(__file__).resolve().with_name("jax_station_zap_score.py")
DEFAULTS = {"seed": 42, "max_epochs": 3, "patience": 20, "time_cap_seconds": 600, "max_batch_size": 64, "chunk": 8,
            "train_wall_seconds": 720, "score_wall_seconds": 120, "memory_mb": 6000, "threads": 4}


def _child(script, job, P, wall, what):
    try:
        return run_child(script, job, wall_seconds=wall, memory_mb=P["memory_mb"], threads=P["threads"])
    except ChildFailure as failure:
        tail = " | ".join(line for line in failure.stderr_tail.splitlines()[-4:] if line.strip())
        raise CandidateError(f"{what} failed ({failure.kind}): {failure}" + (f"; stderr: {tail[-500:]}" if tail else "")) from None


def _checked(result, what):
    """A child's failure is the candidate's unless the child reports a defect of the environment."""
    if result.get("ok"):
        return
    if result.get("environment"):
        raise RuntimeError(f"{what} cannot run: {result.get('error')}")
    raise CandidateError(f"{what} failed: {result.get('error')}")


def score(module, P):
    """Family entry point: returns (test MAE, details)."""
    P = {**DEFAULTS, **(P or {})}
    path = getattr(module, "__file__", None)
    if not path or not Path(path).is_file():
        raise CandidateError("candidate module has no source file")
    for name in ("create_network", "compute_loss", "create_optimizer", "_define_hyperparameters"):
        if hasattr(module, name) and not callable(getattr(module, name)):
            raise CandidateError(f"{name} must be a function")
    root = zd.zap_dir()
    trainval, test = root / "trainval", root / "test"
    for name in ("trainval/train.npy", "trainval/val.npy", "test/test.npy"):
        if not (root / name).is_file():
            raise RuntimeError(f"missing {root / name}; run `python scripts/prepare_jax_station.py --zapbench`")
    out_dir = Path(tempfile.mkdtemp(prefix="artask-zap-out-"))
    try:
        job = {"source": Path(path).read_text(encoding="utf-8"), "trainval_dir": str(trainval), "out_dir": str(out_dir),
               "seed": P["seed"], "max_epochs": P["max_epochs"], "patience": P["patience"],
               "time_cap_seconds": P["time_cap_seconds"], "max_batch_size": P["max_batch_size"], "chunk": P["chunk"],
               "cpu_seconds": int(P["train_wall_seconds"] * P["threads"] + 60)}
        trained, train_stats = _child(TRAIN_WORKER, job, P, P["train_wall_seconds"], "training")
        _checked(trained, "training")
        job = {"out_dir": str(out_dir), "test_dir": str(test), "cpu_seconds": int(P["score_wall_seconds"] * P["threads"] + 60)}
        scored, score_stats = _child(SCORE_WORKER, job, P, P["score_wall_seconds"], "test scoring")
        _checked(scored, "test scoring")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    value = scored["test_mae"]
    if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid test MAE {value!r}")
    keys = ("best_val_mae", "best_val_step_maes", "best_epoch", "epochs", "stopped_by", "history", "batch_size",
            "hyperparameters", "parameters", "training_seconds", "train_windows", "val_windows", "versions")
    details = {"test_mae": value, **{f"test_{k}": scored[k] for k in ("condition_mae", "step_mae", "pooled_mae",
                                                                       "train_conditions_mae", "holdout_mae", "test_windows")},
               "test_step_mae_all": scored["step_mae_all"], **{k: trained.get(k) for k in keys},
               "budget": {k: P[k] for k in ("seed", "max_epochs", "patience", "time_cap_seconds", "threads")},
               "train_child": train_stats, "score_child": score_stats}
    return value, details
