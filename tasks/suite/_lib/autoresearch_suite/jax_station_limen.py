"""LIMEN interface discovery on XLand-MiniGrid: the pinned LIMEN evaluator on CPU.

LIMEN (Jaswal, Baghel and Chopra, 2026; github.com/Lossfunk/LIMEN, MIT) searches
for the observation and reward functions an RL agent learns from. A candidate
is LIMEN's interface module::

    get_observation(state) -> 1-D float array (length 1..512)
    compute_reward(state, action, next_state) -> scalar float

over XLand-MiniGrid states (dunnolab/xland-minigrid, Apache-2.0). The score is
LIMEN's own fitness, computed by LIMEN's ``CascadeEvaluator`` from the pinned
commit (unchanged, in a child process, see jax_station_limen_worker.py) with
LIMEN's YAML configuration for the task: crash filter, a short single-seed PPO
run whose success rate must reach the cascade threshold (otherwise that
success rate is the fitness), then PPO training with three seeds and the mean
final success rate over 50 evaluation episodes per seed.

The LIMEN files are fetched by ``scripts/prepare_jax_station.py --limen`` into
``data_dir()/limen-cf0b44da`` and checked against catalog/jax_station-data-lock.json.

The evaluator reads the candidate's source only (``candidate_source``); with
``sandboxed.run`` no statement of the candidate runs in the evaluator process.
"""
from __future__ import annotations

import math
from pathlib import Path

from . import jax_station_policy as policy
from .jax_station_common import ChildFailure, data_dir, run_child
from .runtime import CandidateError
from .sandboxed import defines

LIMEN_COMMIT = "cf0b44da37f7e7ee038920c860a8cb00c610c9af"
WORKER = Path(__file__).resolve().with_name("jax_station_limen_worker.py")
CONFIGS = {"easy_pickup": "configs/easy_pickup.yaml", "medium_place_near": "configs/medium_place_near.yaml",
           "hard_rule_chain": "configs/hard_rule_chain.yaml"}
STAGES = ("crashed", "short_train", "short_train_rejected", "full_train")


def limen_dir():
    return data_dir() / f"limen-{LIMEN_COMMIT[:8]}"


def candidate_source(module, names=("get_observation", "compute_reward")):
    for name in names:
        if not defines(module, name):
            raise CandidateError(f"candidate must define {name}()")
    path = getattr(module, "__file__", None)
    if not path or not Path(path).is_file():
        raise CandidateError("candidate module has no source file")
    source = Path(path).read_text(encoding="utf-8")
    problems = policy.check_interface_source(source)
    if problems:
        raise CandidateError("candidate violates the interface policy: " + "; ".join(problems[:6]))
    return source


def run_limen(source, P):
    """Run LIMEN's CascadeEvaluator on ``source`` in a child process; return its result and process stats."""
    root = limen_dir()
    if not (root / "limen" / "evaluator.py").is_file():
        raise RuntimeError(f"LIMEN checkout missing at {root}; run `python scripts/prepare_jax_station.py --limen`")
    threads = int(P.get("threads", 4))
    job = {"limen_dir": str(root), "config": CONFIGS[P["task"]], "source": source,
           "cpu_seconds": int(P["wall_seconds"] * threads + 120), "overrides": P.get("overrides") or {}}
    try:
        return run_child(WORKER, job, wall_seconds=P["wall_seconds"], memory_mb=P.get("memory_mb", 6000), threads=threads)
    except ChildFailure as failure:
        tail = " | ".join(line for line in failure.stderr_tail.splitlines()[-4:] if line.strip())
        raise CandidateError(f"LIMEN evaluation failed ({failure.kind}): {failure}" + (f"; stderr: {tail[-500:]}" if tail else "")) from None


def score(module, P):
    """Family entry point: P['task'] selects LIMEN's configuration; returns (fitness, details)."""
    source = candidate_source(module)
    result, stats = run_limen(source, P)
    if not result.get("ok"):
        raise RuntimeError(f"LIMEN harness error: {result.get('error')}")
    stage = result.get("stage")
    if stage == "policy_rejected":
        raise CandidateError("candidate violates the interface policy: " + str(result.get("error")))
    if stage not in STAGES:
        raise RuntimeError(f"unexpected LIMEN stage {stage!r}")
    crash = result.get("crash") or {}
    if stage == "crashed":
        raise CandidateError(f"LIMEN crash filter rejected the candidate at stage {crash.get('stage_failed')}: {crash.get('error')}")
    metrics = result.get("metrics")
    if not result.get("passed") or not metrics:
        raise CandidateError("PPO training failed or produced non-finite metrics; LIMEN records no fitness for this candidate")
    value = metrics.get("success_rate")
    if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(f"LIMEN returned an invalid success rate {value!r}")
    runs = result.get("runs") or []
    full = [r for r in runs if r.get("total_timesteps") == result.get("total_timesteps_full")]
    rates = [r.get("success_rate") for r in full]
    details = {
        "stage": stage, "fitness_key": result.get("fitness_key"), "cascade": result.get("cascade"),
        "cascade_threshold": result.get("cascade_threshold"), "obs_dim": result.get("obs_dim"),
        "seed_success_rates": rates, "seed_success_std": _std(rates),
        "full_training_failed": stage == "short_train",
        "runs": runs, "final_return": metrics.get("final_return"), "final_length": metrics.get("final_length"),
        "limen_eval_seconds": result.get("eval_seconds"), "child": stats, "task": P["task"],
        "limen_commit": LIMEN_COMMIT, "versions": result.get("versions"),
    }
    return float(value), details


def _std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
