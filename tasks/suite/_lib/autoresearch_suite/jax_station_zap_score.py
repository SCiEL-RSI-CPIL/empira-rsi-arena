"""Scoring child of the ZAPBench evaluator: test-split MAE of an exported predictor.

Started by jax_station_zap.py (through jax_station_common.run_child) as
``python -s -B jax_station_zap_score.py <result_fd>``; the job arrives as JSON on stdin::

    {"out_dir": str, "test_dir": str, "cpu_seconds": int}

No candidate Python code runs here: the predictor is the StableHLO module that
the training child exported with jax.export (parameters as explicit inputs).
Before running it, the module text is checked for host callbacks, foreign
custom calls and host communication (only LAPACK and sharding custom calls are
allowed), and the parameters are checked against the module's input shapes.

The training child is controlled by the candidate, so nothing it writes is
trusted to treat windows separately: the module must take exactly one window
([1, 4, N] -> [1, 32, N]) and is called once per test window. Consecutive test
windows overlap (stride 1: the last context step of window i + k is target
step k of window i), so a module that received several windows per call could
copy held-out targets from its neighbours.

Metric (the ZAPBench / Station test metric with a 4-step context): for every
test condition (the last 20 % of each of the 8 training conditions and the
held-out condition 3 from step 256 - 4 on), the MAE over all stride-1 windows,
the 32 predicted steps and the 71,721 neurons; the score is the mean of the 9
condition MAEs (equivalently, the condition-averaged MAE averaged over steps 1
to 32). Non-finite predictions make the candidate invalid.
"""
import json
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT, HORIZON = 4, 32
STEPS = (1, 4, 8, 16, 32)
ALLOWED_CUSTOM_CALLS = re.compile(r"^(lapack_\w+|Sharding|SPMDFullToShardShape|SPMDShardToFullShape|annotate_device_placement)$")
REQUIRED_MODULES = ("jax", "numpy", "flatbuffers")  # jax.export (de)serialisation imports flatbuffers
FORBIDDEN_OPS = ("stablehlo.infeed", "stablehlo.outfeed", "stablehlo.send", "stablehlo.recv", "mhlo.infeed",
                 "mhlo.outfeed", "mhlo.send", "mhlo.recv")


class ContractError(Exception):
    """The exported predictor or its outputs violate the task contract."""


def _load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_module(text):
    """Custom-call targets and host-communication ops of a StableHLO text (empty list = accepted)."""
    problems = []
    for target in sorted(set(re.findall(r'custom_call\s*@"?([\w.$-]+)"?', text)) |
                         set(re.findall(r'call_target_name\s*=\s*"([^"]+)"', text))):
        if not ALLOWED_CUSTOM_CALLS.match(target):
            problems.append(f"custom call {target} is not allowed")
    for op in FORBIDDEN_OPS:
        if op in text:
            problems.append(f"operation {op} is not allowed")
    return problems


def check_shapes(in_shape, out_shape):
    """Neuron count of a predictor that maps one window [1, 4, N] to [1, 32, N]; anything else is rejected."""
    in_shape, out_shape = tuple(in_shape), tuple(out_shape)
    if len(in_shape) != 3 or in_shape[:2] != (1, CONTEXT) or out_shape != (1, HORIZON, in_shape[2]):
        raise ContractError(f"the exported predictor must map one window [1, {CONTEXT}, N] to [1, {HORIZON}, N], "
                            f"got {list(in_shape)} -> {list(out_shape)}")
    return int(in_shape[2])


def window_starts(labels):
    """Start indices of all stride-1 windows of CONTEXT + HORIZON steps that stay within one condition."""
    import numpy as np
    view = np.lib.stride_tricks.sliding_window_view(np.asarray(labels), CONTEXT + HORIZON)
    return np.flatnonzero(np.all(view == view[:, :1], axis=1))


def evaluate(call, data, labels):
    """Per-condition, per-step absolute errors of ``call(x[1, 4, N]) -> [1, 32, N]`` over all test windows."""
    import numpy as np
    neurons = data.shape[1]
    starts = window_starts(labels)
    conditions = [int(c) for c in dict.fromkeys(np.asarray(labels).tolist())]
    sums = {c: np.zeros(HORIZON, dtype=np.float64) for c in conditions}
    counts = {c: 0 for c in conditions}
    for start in starts:
        window = np.asarray(data[start:start + CONTEXT + HORIZON], dtype=np.float32)
        pred = np.asarray(call(window[None, :CONTEXT]))
        if pred.shape != (1, HORIZON, neurons):
            raise ContractError("the predictor returned an array of the wrong shape")
        if not np.all(np.isfinite(pred)):
            raise ContractError("the predictor produced non-finite test predictions")
        condition = int(labels[start])
        sums[condition] += np.abs(pred[0] - window[CONTEXT:]).sum(axis=1, dtype=np.float64)
        counts[condition] += 1
    per_condition = {c: sums[c] / (counts[c] * neurons) for c in conditions if counts[c]}
    conditions = list(per_condition)
    condition_mae = {str(c): float(v.mean()) for c, v in per_condition.items()}
    step_curve = np.mean([per_condition[c] for c in conditions], axis=0)
    pooled = float(sum(s.sum() for s in sums.values()) / (sum(counts.values()) * HORIZON * neurons))
    train_conditions = [c for c in conditions if c != 3]
    return {
        "test_mae": float(np.mean([per_condition[c].mean() for c in conditions])),
        "condition_mae": condition_mae, "condition_windows": {str(c): counts[c] for c in conditions},
        "step_mae": {str(s): float(step_curve[s - 1]) for s in STEPS},
        "step_mae_all": [round(float(v), 7) for v in step_curve],
        "pooled_mae": pooled, "train_conditions_mae": float(np.mean([per_condition[c].mean() for c in train_conditions])),
        "holdout_mae": float(per_condition[3].mean()) if 3 in per_condition else None,
        "test_windows": int(len(starts)),
    }


def score(job, policy):
    import jax
    import numpy as np
    from jax import export

    with open(os.path.join(job["out_dir"], "predictor.bin"), "rb") as handle:
        exported = export.deserialize(bytearray(handle.read()))
    problems = check_module(exported.mlir_module())
    if problems:
        raise ContractError("; ".join(problems[:5]))
    with np.load(os.path.join(job["out_dir"], "params.npz"), allow_pickle=False) as archive:
        leaves = [archive[f"arr_{i}"] for i in range(len(archive.files))]
    avals = exported.in_avals
    if len(avals) != len(leaves) + 1:
        raise ContractError("parameter count does not match the exported predictor")
    for leaf, aval in zip(leaves, avals):
        if tuple(leaf.shape) != tuple(aval.shape) or leaf.dtype != aval.dtype:
            raise ContractError("parameter shapes do not match the exported predictor")
    neurons = check_shapes(avals[-1].shape, exported.out_avals[0].shape)

    data = np.load(os.path.join(job["test_dir"], "test.npy"), mmap_mode="r")
    labels = np.load(os.path.join(job["test_dir"], "test_conditions.npy"))
    tmp = os.environ.get("TMPDIR") or os.getcwd()
    policy.install_audit_hook(read_roots=[job["out_dir"], job["test_dir"], tmp], write_roots=[tmp])
    if data.shape[1] != neurons:
        raise ContractError("predictor neuron count does not match the test data")
    call = jax.jit(exported.call)
    return {"ok": True, **evaluate(lambda x: call(leaves, x), data, labels), "versions": {"jax": jax.__version__}}


def main():
    result_fd = int(sys.argv[1])
    job = json.loads(sys.stdin.read())
    policy = _load("jax_station_policy")
    policy.apply_limits(job.get("cpu_seconds"), file_mb=16)
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        missing = policy.missing_modules(REQUIRED_MODULES)
        payload = {"ok": False, "environment": True, "error": missing} if missing else score(job, policy)
    except ContractError as error:
        payload = {"ok": False, "contract": True, "error": str(error)[:2000]}
    except BaseException as error:  # noqa: BLE001
        payload = {"ok": False, "contract": False, "error": f"{type(error).__name__}: {error}"[:2000],
                   "traceback": traceback.format_exc()[-3000:]}
    sys.stdout = real_stdout
    with os.fdopen(result_fd, "w", encoding="utf-8") as channel:
        channel.write(json.dumps(payload, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
