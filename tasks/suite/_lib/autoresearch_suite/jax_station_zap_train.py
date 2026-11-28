"""Training child of the ZAPBench evaluator: trains one candidate on the ZAPBench training split.

Started by jax_station_zap.py (through jax_station_common.run_child) as
``python -s -B jax_station_zap_train.py <result_fd>``; the job arrives as JSON on stdin::

    {"source": str, "trainval_dir": str, "out_dir": str, "seed": int, "max_epochs": int,
     "patience": int, "time_cap_seconds": float, "max_batch_size": int, "chunk": int, "cpu_seconds": int}

This process can read only the training and validation arrays (trainval_dir),
the Python installation and its private directories; the test split is not
readable (audit hook of jax_station_policy) and is never sent to it. The
training loop ports the Station's train_single.py (dualverse-ai/station@782088e5,
example/research_misc/zapbench, Apache-2.0): condition-aware 4 -> 32 windows,
epoch-wise shuffling with jax.random.permutation, the Station's train step
(mutable collections, dropout RNG, optimiser on params['params']), validation
MAE after every epoch with early stopping (patience 20) and the best-validation
parameters kept. Differences: the seed is fixed, the budget is ``max_epochs``
epochs and ``time_cap_seconds`` of training, and validation predictions are
made one window at a time (vmap over single-window calls of ``apply``) so a
model cannot pool information across windows.

After training, the best parameters (params.npz) and the predictor of one
window ([1, 4, N] -> [1, 32, N]) exported with jax.export (predictor.bin,
parameters as explicit inputs) are written to ``out_dir``; a separate child
(jax_station_zap_score.py) that never runs candidate code calls it once per
test window. Candidate code has the whole of this process: it can read the
training and validation arrays itself and replace anything here, including
what is exported. The scoring child therefore accepts single-window modules
only and trusts nothing else this process reports (the validation figures in
the details are the candidate process's own).
"""
import json
import math
import os
import sys
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT, HORIZON, NUM_NEURONS = 4, 32, 71721
STEPS = (1, 4, 8, 16, 32)
REQUIRED_MODULES = ("jax", "flax", "optax", "numpy", "flatbuffers")  # jax.export serialisation imports flatbuffers


class ContractError(Exception):
    """The candidate violated the task contract (reported as an invalid candidate)."""


def _load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def window_starts(labels):
    import numpy as np
    view = np.lib.stride_tricks.sliding_window_view(np.asarray(labels), CONTEXT + HORIZON)
    return np.flatnonzero(np.all(view == view[:, :1], axis=1)).astype(np.int64)


def gather(data, starts):
    """(len(starts), CONTEXT + HORIZON, neurons) float32 windows from a float16/float32 array."""
    import numpy as np
    index = np.asarray(starts)[:, None] + np.arange(CONTEXT + HORIZON)[None, :]
    return np.asarray(data[index.reshape(-1)], dtype=np.float32).reshape(len(starts), CONTEXT + HORIZON, data.shape[1])


def make_train_step(network, optimizer, compute_loss_fn):
    """The Station's make_train_step (train_single.py), unchanged in behaviour."""
    import jax
    import optax
    mutable_collections = getattr(network, "mutable", None)
    needs_rng = getattr(network, "needs_rng", False)

    @jax.jit
    def train_step(params, opt_state, x, y, rng=None):
        def loss_with_params(params):
            if mutable_collections is not None:
                mutable = mutable_collections
            elif isinstance(params, dict) and "params" in params:
                mutable = [k for k in params.keys() if k in ["batch_stats", "intermediates"]]
            else:
                mutable = []
            if mutable:
                if needs_rng and rng is not None:
                    predictions, updates = network.apply(params, x, training=True, mutable=mutable, rngs={"dropout": rng})
                else:
                    predictions, updates = network.apply(params, x, training=True, mutable=mutable)
                return compute_loss_fn(predictions, y, params, x), updates
            if needs_rng and rng is not None:
                predictions = network.apply(params, x, training=True, rngs={"dropout": rng})
            else:
                predictions = network.apply(params, x, training=True)
            return compute_loss_fn(predictions, y, params, x), None

        (loss, updates), grads = jax.value_and_grad(loss_with_params, has_aux=True)(params)
        if updates is not None and isinstance(updates, dict):
            for key, value in updates.items():
                if isinstance(params, dict):
                    params[key] = value
        if isinstance(params, dict) and "params" in params:
            optimizer_updates, opt_state = optimizer.update(grads["params"], opt_state, params["params"])
            params["params"] = optax.apply_updates(params["params"], optimizer_updates)
        else:
            optimizer_updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, optimizer_updates)
        return params, opt_state, loss

    return train_step


def make_predictor(network, treedef):
    """Single-window predictor over a batch: f(leaves, x[batch, 4, N]) -> [batch, 32, N] float32 (vmap of one-window calls)."""
    import jax
    import jax.numpy as jnp

    def predictor(leaves, x):
        params = jax.tree_util.tree_unflatten(treedef, leaves)

        def one(window):
            out = network.apply(params, window[None], training=False)
            return jnp.asarray(out, dtype=jnp.float32)[0]

        return jax.vmap(one)(x)

    return predictor


def validate(predict, leaves, data, starts, chunk):
    """Pooled validation MAE (the Station's metric) and MAE at steps 1, 4, 8, 16, 32."""
    import numpy as np
    step_sum, count = np.zeros(HORIZON), 0
    for begin in range(0, len(starts), chunk):
        part = starts[begin:begin + chunk]
        windows = gather(data, part)
        x = np.zeros((chunk, CONTEXT, data.shape[1]), dtype=np.float32)
        x[:len(part)] = windows[:, :CONTEXT]
        pred = np.asarray(predict(leaves, x))[:len(part)]
        if not np.all(np.isfinite(pred)):
            return float("nan"), {}
        err = np.abs(pred - windows[:, CONTEXT:])
        step_sum += err.sum(axis=(0, 2), dtype=np.float64)
        count += len(part)
    per_step = step_sum / (count * data.shape[1])
    return float(per_step.mean()), {str(s): float(per_step[s - 1]) for s in STEPS}


def resolve(namespace, defaults):
    funcs = {name: namespace.get(name, fallback) for name, fallback in defaults.DEFAULTS.items()}
    for name, fn in funcs.items():
        if not callable(fn):
            raise ContractError(f"{name} must be callable")
    hparams = funcs["_define_hyperparameters"]()
    if not isinstance(hparams, dict) or not all(isinstance(k, str) for k in hparams):
        raise ContractError("_define_hyperparameters() must return a dict with string keys")
    try:
        json.dumps(hparams, allow_nan=False)
    except (TypeError, ValueError):
        raise ContractError("hyperparameters must be plain JSON values (no search spaces)") from None
    batch = namespace.get("BATCH_SIZE", 8)
    if isinstance(batch, bool) or not isinstance(batch, int):
        raise ContractError("BATCH_SIZE must be an int")
    return funcs, hparams, batch


def train(job, policy, defaults):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax  # noqa: F401 - imported before the audit hook for candidate use
    import flax  # noqa: F401

    def load(name):
        return np.load(os.path.join(job["trainval_dir"], name), mmap_mode="r")

    train_data, val_data = load("train.npy"), load("val.npy")
    train_starts = window_starts(np.load(os.path.join(job["trainval_dir"], "train_conditions.npy")))
    val_starts = window_starts(np.load(os.path.join(job["trainval_dir"], "val_conditions.npy")))
    tmp = os.environ.get("TMPDIR") or os.getcwd()
    policy.install_audit_hook(read_roots=[job["trainval_dir"], job["out_dir"], tmp], write_roots=[job["out_dir"], tmp])

    module = types.ModuleType("candidate")
    module.__dict__["__builtins__"] = __builtins__
    exec(compile(job["source"], "candidate.py", "exec"), module.__dict__)  # noqa: S102 - candidate under the audit hook
    sys.modules["candidate"] = module
    funcs, hparams, batch_size = resolve(module.__dict__, defaults)
    if not 1 <= batch_size <= job["max_batch_size"]:
        raise ContractError(f"BATCH_SIZE must be in 1..{job['max_batch_size']}")

    rng = jax.random.PRNGKey(job["seed"])
    rng, init_rng = jax.random.split(rng)
    network = funcs["create_network"](hparams)
    if not callable(getattr(network, "init", None)) or not callable(getattr(network, "apply", None)):
        raise ContractError("create_network(hparams) must return an object with init() and apply()")
    params = network.init(init_rng, jnp.zeros((1, CONTEXT, NUM_NEURONS)))
    optimizer = funcs["create_optimizer"](hparams.get("learning_rate", 0.001))
    if not callable(getattr(optimizer, "init", None)) or not callable(getattr(optimizer, "update", None)):
        raise ContractError("create_optimizer must return an optax GradientTransformation")
    opt_state = optimizer.init(params["params"] if isinstance(params, dict) and "params" in params else params)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    n_params = int(sum(int(np.size(leaf)) for leaf in leaves))
    train_step = make_train_step(network, optimizer, funcs["compute_loss"])
    needs_rng = getattr(network, "needs_rng", False)
    chunk = int(job["chunk"])
    predict = jax.jit(make_predictor(network, treedef))

    best_val, best_params, best_steps, best_epoch, patience = float("inf"), params, None, 0, 0
    history, started, last_epoch = [], time.time(), 0.0
    stopped_by = "max_epochs"
    for epoch in range(job["max_epochs"]):
        elapsed = time.time() - started
        if epoch and elapsed + 1.1 * last_epoch > job["time_cap_seconds"]:
            stopped_by = "time_cap"
            break
        epoch_start = time.time()
        rng, shuffle_rng = jax.random.split(rng)
        order = np.asarray(jax.random.permutation(shuffle_rng, train_starts))
        losses = []
        for begin in range(0, len(order), batch_size):
            windows = gather(train_data, order[begin:begin + batch_size])
            x, y = jnp.asarray(windows[:, :CONTEXT]), jnp.asarray(windows[:, CONTEXT:])
            if needs_rng:
                rng, dropout_rng = jax.random.split(rng)
                params, opt_state, loss = train_step(params, opt_state, x, y, dropout_rng)
            else:
                params, opt_state, loss = train_step(params, opt_state, x, y)
            losses.append(float(loss))
        val_mae, step_maes = validate(predict, jax.tree_util.tree_leaves(params), val_data, val_starts, chunk)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": mean_loss if math.isfinite(mean_loss) else None,
                        "val_mae": val_mae if math.isfinite(val_mae) else None,
                        "seconds": round(time.time() - epoch_start, 2)})
        print(f"epoch {epoch + 1} train {history[-1]['train_loss']} val {val_mae} ({history[-1]['seconds']} s)", file=sys.stderr, flush=True)
        if val_mae < best_val:
            best_val, best_params, best_steps, best_epoch, patience = val_mae, params, step_maes, epoch + 1, 0
        else:
            patience += 1
        last_epoch = time.time() - epoch_start
        if patience >= job["patience"]:
            stopped_by = "early_stopping"
            break
    training_seconds = time.time() - started
    if not math.isfinite(best_val):
        raise ContractError("no epoch produced a finite validation MAE")

    leaves = [np.asarray(jnp.asarray(leaf)) for leaf in jax.tree_util.tree_leaves(best_params)]  # JAX dtypes (no 64-bit)
    for leaf in leaves:
        if leaf.dtype.kind not in "fiub" or leaf.dtype.itemsize > 8 or leaf.dtype.name == "bfloat16":
            raise ContractError(f"parameter dtype {leaf.dtype} is not supported (use float32/float16/int arrays)")
    specs = [jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) for leaf in leaves]
    x_spec = jax.ShapeDtypeStruct((1, CONTEXT, NUM_NEURONS), jnp.float32)  # one window per call of the scoring child
    try:
        exported = jax.export.export(jax.jit(make_predictor(network, treedef)), platforms=("cpu",))(specs, x_spec)
    except Exception as error:  # noqa: BLE001 - e.g. host callbacks cannot be exported
        raise ContractError(f"the predictor cannot be exported for scoring: {type(error).__name__}: {error}") from None
    if tuple(exported.out_avals[0].shape) != (1, HORIZON, NUM_NEURONS):
        raise ContractError(f"apply() must return (batch, {HORIZON}, {NUM_NEURONS}) predictions, got {exported.out_avals[0].shape}")
    with open(os.path.join(job["out_dir"], "predictor.bin"), "wb") as handle:
        handle.write(exported.serialize())
    np.savez(os.path.join(job["out_dir"], "params.npz"), *leaves)
    return {"ok": True, "best_val_mae": best_val, "best_val_step_maes": best_steps, "best_epoch": best_epoch,
            "epochs": len(history), "stopped_by": stopped_by, "history": history, "batch_size": batch_size,
            "hyperparameters": hparams, "parameters": n_params, "training_seconds": round(training_seconds, 1),
            "train_windows": int(len(train_starts)), "val_windows": int(len(val_starts)),
            "versions": {"jax": jax.__version__, "flax": flax.__version__, "optax": optax.__version__, "numpy": np.__version__}}


def main():
    result_fd = int(sys.argv[1])
    job = json.loads(sys.stdin.read())
    policy = _load("jax_station_policy")
    policy.apply_limits(job.get("cpu_seconds"), file_mb=job.get("file_mb", 4096))
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        missing = policy.missing_modules(REQUIRED_MODULES)  # before any training time is spent
        if missing:
            payload = {"ok": False, "environment": True, "error": missing}
        else:
            payload = train(job, policy, _load("jax_station_zap_defaults"))
    except ContractError as error:
        payload = {"ok": False, "contract": True, "error": str(error)[:2000]}
    except BaseException as error:  # noqa: BLE001 - candidate exceptions are reported to the parent
        payload = {"ok": False, "contract": False, "error": f"{type(error).__name__}: {error}"[:2000],
                   "traceback": traceback.format_exc()[-3000:]}
    sys.stdout = real_stdout
    with os.fdopen(result_fd, "w", encoding="utf-8") as channel:
        channel.write(json.dumps(payload, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
