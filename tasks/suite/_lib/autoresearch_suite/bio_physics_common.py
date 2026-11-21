"""Shared helpers of the biology and physics research tasks (package bio_physics).

Evaluators never use the network. ``scripts/prepare_bio_physics.py`` downloads the
pinned public sources, derives small files with the ``derive`` functions of the task
modules and writes them below ``data_dir()``; every derived file is checked against
the SHA-256 recorded in its task module before use.

Candidate code is executed through ``run_sandboxed``: the candidate source plus a
small trusted driver run in a resource-limited child process (software_sandbox)
that cannot read the prepared data, so held-out answers stay in this process.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from arena_task_paths import data_root
from . import software_sandbox
from .runtime import CandidateError, require


GZIP_OS_BYTE, RECORDED_OS_CODE = 9, 19


def data_dir():
    return data_root() / "bio_physics"


def canonical_bytes(payload):
    """Deterministic gzip-compressed JSON (sorted keys, no timestamp in the gzip header).

    With mtime=0 zlib writes the header and puts its build's operating-system code in byte 9
    (3 on Linux, 19 on macOS). The recorded digests come from files carrying 19, so that value
    is written on every platform; no other byte depends on the platform.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    data = bytearray(gzip.compress(text.encode(), compresslevel=9, mtime=0))
    data[GZIP_OS_BYTE] = RECORDED_OS_CODE
    return bytes(data)


def load_derived(name, sha256):
    """Read one prepared file, refusing missing or changed bytes."""
    path = data_dir() / name
    if not path.is_file():
        raise RuntimeError(f"missing prepared data {path}; run python3 scripts/prepare_bio_physics.py")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise RuntimeError(f"prepared data {name} has SHA-256 {digest}, expected {sha256}; rerun scripts/prepare_bio_physics.py")
    return json.loads(gzip.decompress(raw))


# Appended to every candidate source that runs in the sandbox: converts numpy results to
# JSON-serialisable values and seeds the global numpy generator for reproducibility.
PLAIN = '''

def _bp_plain(value):
    try:
        import numpy as _np
    except ImportError:  # pragma: no cover - numpy is always present in the suite environment
        _np = None
    if _np is not None:
        if isinstance(value, _np.ndarray):
            return value.tolist()
        if isinstance(value, _np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(k): _bp_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bp_plain(v) for v in value]
    return value


def _bp_seed():
    try:
        import numpy as _np
        _np.random.seed(0)
    except ImportError:  # pragma: no cover
        pass
'''


def run_sandboxed(module, driver, entry, args=(), *, required=(), oracle=None, cpu_seconds=60, memory_mb=2048,
                  blocked_modules=()):
    """Run ``entry(*args[, oracle])`` defined in ``driver`` appended to the candidate's source.

    ``required`` names candidate functions that must exist. Any failure of the child
    (exception, time or memory limit, non-JSON result) is the candidate's: CandidateError.
    """
    path = software_sandbox.require_entry(module, *required)
    source = Path(path).read_text(encoding="utf-8") + "\n" + PLAIN + "\n" + driver
    try:
        return software_sandbox.execute(source, entry, list(args), oracle=oracle, cpu_seconds=cpu_seconds,
                                        memory_mb=memory_mb, blocked_modules=blocked_modules, filename=Path(path).name)
    except software_sandbox.SandboxFailure as failure:
        last = failure.traceback_text.strip().splitlines()[-1] if failure.traceback_text.strip() else ""
        detail = f"; {last}" if failure.kind == "exception" and last and last not in str(failure) else ""
        raise CandidateError(f"candidate failed in the sandbox ({failure.kind}): {failure}{detail}"[:2000]) from None


def as_array(value, shape=None, name="output", low=None, high=None, dtype=float):
    """Validate candidate numbers: real, finite, expected shape and optional bounds."""
    try:
        raw = np.asarray(value)
    except Exception as error:  # noqa: BLE001 - any conversion failure is a contract violation
        raise CandidateError(f"{name}: not convertible to an array ({error})") from None
    require(raw.dtype.kind in "fiub" and not np.iscomplexobj(raw), f"{name}: expected real numbers, got dtype {raw.dtype}")
    array = raw.astype(dtype)
    if shape is not None:
        require(array.shape == tuple(shape), f"{name}: expected shape {tuple(shape)}, got {array.shape}")
    require(np.isfinite(array).all(), f"{name}: contains NaN or infinite values")
    if low is not None:
        require(np.all(array >= low), f"{name}: values below {low}")
    if high is not None:
        require(np.all(array <= high), f"{name}: values above {high}")
    return array


def as_int_array(value, shape=None, name="output", low=None, high=None):
    """Validate integer-valued candidate output (integral floats are accepted)."""
    array = as_array(value, shape=shape, name=name, low=low, high=high)
    require(np.all(array == np.round(array)), f"{name}: expected integers")
    return array.astype(np.int64)


def spearman(x, y):
    """Spearman rank correlation with average ranks for ties (0 when either side is constant)."""
    rx, ry = _ranks(np.asarray(x, float)), _ranks(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * (start + stop)
        start = stop + 1
    return ranks
