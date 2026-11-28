"""Child process of the LIMEN evaluators: runs the pinned LIMEN cascade evaluator on one candidate.

Started by jax_station_limen.py (through jax_station_common.run_child) as
``python -s -B jax_station_limen_worker.py <result_fd>`` with a clean environment
(CPU-only JAX, at most four XLA threads through NPROC/PJRT_NPROC) and a private,
empty working and temporary directory. The job arrives as JSON on stdin::

    {"limen_dir": str, "config": "configs/<task>.yaml", "source": str,
     "cpu_seconds": int, "overrides": {"training": {...}, "evaluator": {...}}}

Steps: (1) the candidate source is checked against the static policy of
jax_station_policy (LIMEN's rule that interfaces only use jax and jax.numpy,
plus no attribute assignment, private, I/O or introspection attributes,
exec-like builtins, dunder names or module aliases); (2) LIMEN and its dependencies are imported, then an audit hook
confines the process (no subprocesses, sockets or ctypes; writes only in the
private temporary directory); (3) LIMEN's own ``CascadeEvaluator``
(limen/evaluator.py at the pinned commit, unchanged) evaluates the source with
LIMEN's YAML configuration. ``overrides`` is empty for the faithful tasks. The
only addition is bookkeeping: ``adapter.run_training`` is wrapped to record
per-run metrics; the wrapper calls LIMEN's method unchanged.

The JSON result is written to ``result_fd``. Candidate output goes to stderr.

The interface functions are traced inside LIMEN's training and evaluation code,
so this process holds both the candidate and the measurement of its success.
The static policy and the audit hook keep honest candidates from side effects;
they are not a boundary against code written to defeat them.
"""
import dataclasses
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_policy():
    import importlib.util
    spec = importlib.util.spec_from_file_location("jax_station_policy", os.path.join(HERE, "jax_station_policy.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metrics(metrics):
    if metrics is None:
        return None
    keep = {}
    for key in ("success_rate", "final_return", "final_length", "training_time"):
        if key in metrics:
            value = float(metrics[key])
            keep[key] = value if math.isfinite(value) else None
    keep["success_curve"] = [round(float(v), 6) if _finite(v) else None for v in metrics.get("success_curve", [])]
    return keep


def evaluate(job, policy):
    problems = policy.check_interface_source(job["source"])
    if problems:
        return {"ok": True, "stage": "policy_rejected", "error": "; ".join(problems[:10])}
    limen_dir = os.path.realpath(job["limen_dir"])
    sys.path.insert(0, limen_dir)
    import jax  # noqa: F401 - imported before the audit hook is installed
    import limen  # noqa: F401 - imports LIMEN's package (controller, LLM client) as upstream does
    from limen.adapters import get_adapter
    from limen.config import Config
    from limen.evaluator import CascadeEvaluator

    config = Config.from_yaml(os.path.join(limen_dir, job["config"]))
    environment = dataclasses.replace(config.environment, ruleset_file=os.path.join(limen_dir, config.environment.ruleset_file))
    overrides = job.get("overrides") or {}
    config = dataclasses.replace(
        config, environment=environment,
        training=dataclasses.replace(config.training, **overrides.get("training", {})),
        evaluator=dataclasses.replace(config.evaluator, **overrides.get("evaluator", {})))
    adapter = get_adapter(config.environment.adapter_type, config)
    runs = []
    original = adapter.run_training

    def recorded(run_config, interface, obs_dim, total_timesteps=None):
        started = time.time()
        metrics = original(run_config, interface, obs_dim, total_timesteps)
        runs.append({"seed": run_config.training.seed, "total_timesteps": total_timesteps,
                     "wall_seconds": round(time.time() - started, 2), **(_metrics(metrics) or {})})
        return metrics

    adapter.run_training = recorded
    evaluator = CascadeEvaluator(config, adapter)
    tmp = os.environ.get("TMPDIR") or os.getcwd()
    policy.install_audit_hook(read_roots=[limen_dir, tmp, HERE], write_roots=[tmp])
    started = time.time()
    result = evaluator.evaluate(job["source"])
    crash = result.crash_result
    return {
        "ok": True, "stage": result.stage.value, "passed": bool(result.passed), "obs_dim": result.obs_dim,
        "crash": {"passed": bool(crash.passed), "stage_failed": crash.stage_failed,
                  "error": (crash.error_message or "")[:1500] or None},
        "metrics": _metrics(result.metrics), "runs": runs, "eval_seconds": round(time.time() - started, 2),
        "fitness_key": evaluator.fitness_key, "cascade": bool(config.evaluator.cascade_evaluation),
        "cascade_threshold": (config.evaluator.cascade_thresholds or [0.0])[0],
        "total_timesteps": config.training.total_timesteps, "total_timesteps_full": config.training.total_timesteps_full,
        "num_seeds": config.training.num_seeds, "seed": config.training.seed,
        "versions": {"jax": jax.__version__, "python": sys.version.split()[0]},
    }


def main():
    result_fd = int(sys.argv[1])
    job = json.loads(sys.stdin.read())
    policy = _load_policy()
    policy.apply_limits(job.get("cpu_seconds"), file_mb=64)
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # candidate and library prints must not reach the result channel
    try:
        payload = evaluate(job, policy)
    except BaseException as error:  # noqa: BLE001 - every failure is reported to the parent
        payload = {"ok": False, "error": f"{type(error).__name__}: {error}"[:2000],
                   "traceback": traceback.format_exc()[-3000:]}
    sys.stdout = real_stdout
    with os.fdopen(result_fd, "w", encoding="utf-8") as channel:
        channel.write(json.dumps(payload, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
