"""BIG-bench scaling-law discovery (LaSR, Section 4.6 / Appendix A.6), recreated.

The data are the BIG-bench evaluation records (google/BIG-bench, Apache-2.0) whose
preferred score is the multiple-choice grade, one row per (task or subtask, model,
number of shots), prepared by ``scripts/prepare_metaopt.py``. A candidate is a
closed-form law ``scaling_law(X, params)`` with at most five free constants; the
evaluator fits the constants on a fixed-seed 80 % training split by nonlinear
least squares (Levenberg-Marquardt, several fixed starts) and scores the mean
squared error on the remaining 20 % (lower is better). The fit runs inside the
sandbox, which cannot load SciPy (it needs ctypes, which the sandbox blocks), so
the Levenberg-Marquardt routine is a small NumPy implementation; the tests check
it against ``scipy.optimize.least_squares``.

The fit and the held-out predictions run in a sandboxed child process that
receives the training rows and the held-out *features* only; the held-out grades
stay in this process. A second child recomputes the held-out predictions from the
reported constants on the held-out rows in a fresh random order, to check that
they come from the formula and not from the position of a row: the data and the
split are public, so a candidate can carry a table of the held-out grades in a
docstring, in identifiers or in one long integer literal, which the source
limits do not count. With the rows reordered such a table is of no use, and what
remains is a table of per-input-combination means, whose error is reported as
``memorisation_floor_mse``.
"""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import io
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from arena_task_paths import data_root
from . import software_sandbox
from .runtime import CandidateError, require

DATA_FILE = "bigbench_mcg.csv.gz"
DATA_ROWS = 53812
# SHA-256 of the decompressed CSV written by scripts/prepare_metaopt.py (catalog/metaopt-data-lock.json).
DATA_SHA256 = "dc5dd63e36e5d6d6120299d4652112cef6b2217b2fb940e15475bb4676a3cf3e"
FEATURES = ("training_steps", "batch_size", "total_params", "non_embedding_params", "flop_matched_params", "shots", "temperature")
COLUMNS = ("task", "subtask", "model_family", "model_name") + FEATURES + ("grade",)
MAX_PARAMS = 5
MAX_LITERALS = 10


def data_dir():
    return data_root() / "metaopt"


@lru_cache(maxsize=1)
def load_rows(path=None):
    """Return (dict of feature arrays, grades, (model, shots, temperature) group labels, task names) from the prepared CSV."""
    path = Path(path) if path is not None else data_dir() / DATA_FILE
    require(path.is_file(), f"missing {path.name}; run python scripts/prepare_metaopt.py")
    payload = gzip.decompress(path.read_bytes())
    if DATA_SHA256 is not None:
        require(hashlib.sha256(payload).hexdigest() == DATA_SHA256, "prepared BIG-bench data do not match the locked SHA-256")
    reader = csv.reader(io.StringIO(payload.decode("utf-8")))
    header = next(reader)
    require(tuple(header) == COLUMNS, "unexpected columns in the prepared BIG-bench data")
    rows = list(reader)
    require(len(rows) == DATA_ROWS, f"expected {DATA_ROWS} rows, found {len(rows)}")
    table = {name: np.array([float(r[i]) for r in rows]) for i, name in enumerate(COLUMNS) if name in FEATURES or name == "grade"}
    groups = ["|".join(r[2:4]) + f"|{r[9]}|{r[10]}" for r in rows]  # model family, model, shots, temperature
    tasks = [r[0] for r in rows]
    return {name: table[name] for name in FEATURES}, table["grade"], groups, tasks


def split_indices(n, seed=0, train_fraction=0.8):
    order = np.random.default_rng(seed).permutation(n)
    cut = int(math.floor(train_fraction * n))
    return np.sort(order[:cut]), np.sort(order[cut:])


def source_profile(source):
    """Counts that bound how much data a candidate can embed in its source (a guard against lookup tables).

    non_small_literals: numeric literals other than integers in [-10, 10]; numeric_literals: all numeric
    literals; string_chars: characters of string literals outside docstrings; length: source characters.
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))
    profile = {"non_small_literals": 0, "numeric_literals": 0, "string_chars": 0, "length": len(source)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
            continue
        value = node.value
        if isinstance(value, (int, float, complex)):
            profile["numeric_literals"] += 1
            if not (isinstance(value, int) and -10 <= value <= 10):
                profile["non_small_literals"] += 1
        elif isinstance(value, (str, bytes)) and id(node) not in docstrings:
            profile["string_chars"] += len(value)
    return profile


def literal_count(source):
    """Numeric literals other than integers in [-10, 10] (a guard against hidden fitted constants)."""
    return source_profile(source)["non_small_literals"]


LIMITS = {"non_small_literals": MAX_LITERALS, "numeric_literals": 64, "string_chars": 1000, "length": 20000}


HARNESS = r'''

def _artask_declared():
    value = globals().get("N_PARAMS")
    return {"type": type(value).__name__, "value": value if type(value) is int else None}


def _artask_real(value):
    import numpy as _np
    value = _np.asarray(value)
    if value.dtype.kind not in "iuf":
        raise ValueError(f"scaling_law must return real numbers, got dtype {value.dtype}")
    return value.astype(float)


def _artask_columns(cols):
    import numpy as _np
    return {k: _np.asarray(v, dtype=float) for k, v in cols.items()}


def _artask_predict(cols, params):
    import numpy as _np
    X = _artask_columns(cols)
    n = len(next(iter(X.values())))
    pred = _artask_real(scaling_law({k: v.copy() for k, v in X.items()}, _np.asarray(params, dtype=float).copy()))
    if pred.ndim == 0:
        pred = _np.full(n, float(pred))
    if pred.shape != (n,):
        raise ValueError(f"scaling_law returned shape {pred.shape}, expected ({n},)")
    return [float(v) for v in pred]


def _artask_lm(residual, x, max_nfev):
    """Levenberg-Marquardt with forward-difference Jacobian and Marquardt scaling (NumPy only)."""
    import numpy as _np
    r = residual(x)
    cost, nfev, lam = float(r @ r), 1, 1e-3
    while nfev < max_nfev:
        J = _np.empty((len(r), len(x)))
        for j in range(len(x)):
            h = 1.4901161193847656e-08 * max(1.0, abs(float(x[j])))
            xp = x.copy()
            xp[j] += h
            J[:, j] = (residual(xp) - r) / h
        nfev += len(x)
        g, A = J.T @ r, J.T @ J
        if not _np.all(_np.isfinite(A)) or float(_np.max(_np.abs(g))) <= 1e-15 * max(cost, 1e-300):
            break
        accepted = False
        while nfev < max_nfev and lam < 1e12:
            step = _np.linalg.lstsq(A + lam * _np.diag(_np.diag(A) + 1e-12), -g, rcond=None)[0]
            trial = x + step
            rt = residual(trial)
            nfev += 1
            ct = float(rt @ rt)
            if _np.isfinite(ct) and ct < cost:
                done = cost - ct <= 1e-12 * cost or float(_np.max(_np.abs(step))) <= 1e-12 * (1.0 + float(_np.max(_np.abs(x))))
                x, r, cost, lam, accepted = trial, rt, ct, max(lam / 3.0, 1e-12), True
                break
            lam *= 4.0
        if not accepted or done:
            break
    return x, cost


def _artask_fit(train_cols, train_y, test_cols, n_params, starts, seed, max_nfev):
    import numpy as _np
    X = _artask_columns(train_cols)
    y = _np.asarray(train_y, dtype=float)
    k = int(n_params)
    init = globals().get("initial_params")
    p0 = _np.ones(k) if not callable(init) else _np.asarray(init(), dtype=float).ravel()
    if p0.shape != (k,) or not _np.all(_np.isfinite(p0)):
        raise ValueError(f"initial_params() must return {k} finite numbers")

    def residual(p):
        with _np.errstate(all="ignore"):
            pred = _artask_real(scaling_law({c: v.copy() for c, v in X.items()}, p.copy()))
        if pred.ndim == 0:
            pred = _np.full(y.shape, float(pred))
        if pred.shape != y.shape:
            raise ValueError(f"scaling_law returned shape {pred.shape}, expected {y.shape}")
        r = pred - y
        return _np.where(_np.isfinite(r), _np.clip(r, -1e3, 1e3), 1e3)

    rng = _np.random.default_rng(seed)
    points = [p0] + [p0 + rng.normal(size=k) * (1.0 + _np.abs(p0)) for _ in range(int(starts) - 1)]
    best, tried = None, []
    for start in points:
        try:
            x, cost = _artask_lm(residual, start.astype(float), int(max_nfev))
        except (FloatingPointError, _np.linalg.LinAlgError):
            tried.append(None)
            continue
        mse = cost / len(y)
        tried.append(mse)
        if _np.isfinite(mse) and _np.all(_np.isfinite(x)) and (best is None or mse < best[0]):
            best = (mse, x)
    if best is None:
        raise ValueError("no start produced a finite least-squares fit")
    return {"params": [float(v) for v in best[1]], "train_mse": float(best[0]), "start_mse": tried,
            "test_pred": _artask_predict(test_cols, best[1])}
'''


def _columns(X, index):
    return {name: X[name][index].tolist() for name in FEATURES}


def group_mean_mse(groups, y, train, test):
    """Held-out MSE of per-(model, shots, temperature) training means: no function of the features can do better in expectation."""
    sums, counts = {}, {}
    for i in train:
        sums[groups[i]] = sums.get(groups[i], 0.0) + y[i]
        counts[groups[i]] = counts.get(groups[i], 0) + 1
    fallback = float(np.mean(y[train]))
    pred = np.array([sums[groups[i]] / counts[groups[i]] if groups[i] in counts else fallback for i in test])
    return float(np.mean((pred - y[test]) ** 2))


def memorisation_floor(X, y, test):
    """Held-out MSE of the held-out means per combination of the seven inputs.

    A prediction that is a function of a row's inputs cannot score below it on these rows. It is the error of a
    table of held-out means, so a score between it and ``group_mean_mse`` does not show a better law.
    """
    keys = list(zip(*(X[name][test].tolist() for name in FEATURES)))
    sums, counts = {}, {}
    for key, value in zip(keys, y[test]):
        sums[key] = sums.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1
    pred = np.array([sums[key] / counts[key] for key in keys])
    return float(np.mean((pred - y[test]) ** 2))


def declared_n_params(module, combined, cpu):
    """N_PARAMS as the candidate binds it; a candidate that was not executed here (sandboxed.run) is asked in a child."""
    if not getattr(module, "__candidate_source__", False):
        return getattr(module, "N_PARAMS", None)
    declared = _child(combined, "_artask_declared", [], cpu)
    return declared.get("value") if isinstance(declared, dict) and declared.get("type") == "int" else None


def score(module, P):
    """Fit the candidate on the training split and return (held-out MSE, details)."""
    path = software_sandbox.require_entry(module, "scaling_law")
    source = Path(path).read_text(encoding="utf-8")
    profile = source_profile(source)
    limits = {**LIMITS, "non_small_literals": P.get("max_literals", MAX_LITERALS)}
    for key, limit in limits.items():
        require(profile[key] <= limit, f"candidate source exceeds the {key} limit ({profile[key]} > {limit})")
    literals = profile["non_small_literals"]
    require("_artask_" not in source, "names starting with _artask_ are reserved for the evaluator")
    combined = source + HARNESS
    cpu = P.get("cpu_seconds", 90)
    n_params = declared_n_params(module, combined, cpu)
    require(isinstance(n_params, int) and not isinstance(n_params, bool) and 1 <= n_params <= P.get("max_params", MAX_PARAMS),
            f"N_PARAMS must be an integer between 1 and {P.get('max_params', MAX_PARAMS)}")
    X, y, groups, _ = load_rows()
    train, test = split_indices(len(y), seed=P.get("split_seed", 0))
    result = _child(combined, "_artask_fit", [_columns(X, train), y[train].tolist(), _columns(X, test), n_params,
                                               P.get("starts", 8), P.get("start_seed", 0), P.get("max_nfev", 400)], cpu)
    require(isinstance(result, dict), "fit returned no result")
    params, pred = _numbers(result.get("params"), "fitted constants"), _numbers(result.get("test_pred"), "held-out predictions")
    train_mse = _numbers(result.get("train_mse"), "training MSE")
    require(train_mse.shape == () and bool(np.isfinite(train_mse)), "training MSE must be one finite number")
    require(params.shape == (n_params,) and np.all(np.isfinite(params)), "fitted constants must be N_PARAMS finite numbers")
    require(pred.shape == test.shape and np.all(np.isfinite(pred)), "held-out predictions must be finite, one per held-out row")
    # The recomputation sees the held-out rows in an order drawn now from the operating system's entropy, which no
    # candidate can know or reproduce: predictions that depend on where a row stands do not come back the same.
    order = np.random.default_rng().permutation(len(test))
    reordered = _numbers(_child(combined, "_artask_predict", [_columns(X, test[order]), params.tolist()], cpu), "recomputed predictions")
    require(reordered.shape == pred.shape, "recomputed held-out predictions must be one per held-out row")
    again = np.empty(len(test))
    again[order] = reordered
    require(np.allclose(again, pred, rtol=1e-9, atol=1e-12),
            "scaling_law is not deterministic or depends on the order of the rows: held-out predictions recomputed "
            "from the reported constants on reordered rows differ")
    mse = float(np.mean((pred - y[test]) ** 2))
    require(math.isfinite(mse), "held-out MSE is not finite")
    floor = memorisation_floor(X, y, test)
    require(mse >= floor - 1e-7, "held-out MSE is below what any function of the inputs can reach on these rows")
    details = {"train_mse": float(train_mse), "n_params": n_params, "params": params.tolist(),
               "n_train": int(len(train)), "n_test": int(len(test)), "numeric_literals": literals,
               "reference_constant_mse": float(np.mean((np.mean(y[train]) - y[test]) ** 2)),
               "reference_group_mean_mse": group_mean_mse(groups, y, train, test), "memorisation_floor_mse": floor}
    return mse, details


def _numbers(value, what):
    """A child's reply as a float array; a reply that is not made of real numbers (strings, booleans, None, integers
    beyond the float range) is the candidate's failure."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        array = None
    if array is None or array.dtype.kind not in "iuf":
        raise CandidateError(f"{what} returned by the sandbox are not numbers")
    return array.astype(float)


def _child(source, entry, args, cpu):
    try:
        return software_sandbox.execute(source, entry, args, cpu_seconds=cpu, memory_mb=2048, filename="candidate.py")
    except software_sandbox.SandboxFailure as failure:
        last = failure.traceback_text.strip().splitlines()[-1] if failure.traceback_text.strip() else ""
        raise CandidateError(f"scaling_law failed in the sandbox ({failure.kind}): {failure} {last}"[:1500]) from None
