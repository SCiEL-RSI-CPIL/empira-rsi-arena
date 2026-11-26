"""Run the CLRS training of a candidate processor in a separate, restricted child process.

``run_training(source, algorithm, splits, budget, ...)`` writes the splits the
candidate may see (train and validation with outputs, test inputs and first
hint frames only) to a fresh temporary directory, starts ``clrs_worker.py`` and
returns the child's compact test predictions and diagnostics. The child loads
the trusted trainer and the data, applies OS limits (CPU time, no file writes)
and the audit-hook policy of software_worker.py (no file reads outside the
Python installation, no processes, sockets or ctypes; the evaluator package is
unloaded), and only then executes the candidate source. Its wall-clock time and
resident memory are watched from here (the resident memory of the child and its
descendants is polled twice a second, with psutil when installed, else from /proc
or ``ps``). The test outputs are never sent to the child, and the child cannot read
them from disk either: with the kernel mechanism of the Kaggle packages
(``kaggle_sandbox.os_sandbox``) the task cache that holds the prepared files (and
any raw CLRS-30 archive kept there) is hidden from it. On Linux, where the kernel
lets an unprivileged process create user namespaces, the worker first calls
``kaggle_worker.isolate``: new user, mount, PID, network and IPC namespaces, an
empty tmpfs over every denied directory, every mount read-only except the working
directory and /dev/shm, no other process visible; on macOS the child runs under
``sandbox-exec`` with the Kaggle profile. Both also hold against native code
(torch.from_file, ctypes). The details name what was in force (``os_sandbox``:
``linux-namespaces``, ``sandbox-exec`` or ``audit-hook-only``). Residual risk:
with ``audit-hook-only`` (containers that forbid user namespaces, kernels before
5.12, a task cache that contains the Python installation or the library) the
prepared file stays readable by native readers the audit hook does not see, and a
candidate that also reaches the trainer through frame references can score the
true test outputs.

``check_source`` is a static guard on the candidate source (imports from an
allow-list, no dunder attribute access, no eval/exec/open/getattr...). The
guard and the policy are defence in depth for honest program search, not a
security boundary against deliberately hostile code.
"""
from __future__ import annotations

import ast
import base64
import json
import os
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import clrs_data, kaggle_sandbox
from .runtime import CandidateError

HEADER = struct.Struct(">Q")
WORKER = Path(__file__).resolve().with_name("clrs_worker.py")
LIB = Path(__file__).resolve().parents[1]
MAX_FRAME = 256 << 20
ADDRESS_SPACE_MARGIN_MB = 2048
ALLOWED_IMPORTS = frozenset({"torch", "math", "numpy", "functools", "itertools", "typing", "dataclasses",
                             "collections", "operator", "__future__"})
FORBIDDEN_NAMES = frozenset({"eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr",
                             "globals", "locals", "vars", "breakpoint", "input", "exit", "quit", "memoryview",
                             "__builtins__", "__loader__", "__spec__", "__file__"})
ARRAY_DTYPES = {"int8", "int16", "int32", "int64", "uint8"}
MAX_DIMENSIONS = 4


def check_source(source):
    """Static guard (not a sandbox): allowed imports only, no dunder attributes, no dynamic code or file access."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise CandidateError(f"cannot parse candidate source: {error}") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]] if node.level == 0 else ["<relative import>"]
        else:
            roots = []
        bad = [root for root in roots if root not in ALLOWED_IMPORTS]
        if bad:
            raise CandidateError(f"import of {bad[0]!r} is not allowed (only {sorted(ALLOWED_IMPORTS - {'__future__'})})")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise CandidateError(f"use of {node.id!r} is not allowed in processor code")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__") and node.attr != "__init__":
            raise CandidateError(f"access to dunder attribute {node.attr!r} is not allowed in processor code")


def encode_arrays(arrays):
    return {name: {"dtype": array.dtype.name, "shape": list(array.shape),
                   "data": base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")}
            for name, array in arrays.items()}


def decode_arrays(payload):
    """Inverse of encode_arrays for untrusted child output (integer dtypes only, sizes checked)."""
    if not isinstance(payload, dict):
        raise CandidateError("child predictions are not a mapping")
    arrays = {}
    for name, item in payload.items():
        if not isinstance(item, dict) or not isinstance(item.get("dtype"), str) or item["dtype"] not in ARRAY_DTYPES:
            raise CandidateError(f"prediction {name!r} is malformed")
        shape, data = item.get("shape"), item.get("data")
        if not isinstance(shape, list) or not isinstance(data, str) or len(shape) > MAX_DIMENSIONS:
            raise CandidateError(f"prediction {name!r} is malformed")
        if not all(type(d) is int and 0 <= d <= MAX_FRAME for d in shape):  # noqa: E721 - bool is not a dimension
            raise CandidateError(f"prediction {name!r} has a malformed shape")
        try:
            raw = base64.b64decode(data, validate=True)
        except (TypeError, ValueError):
            raise CandidateError(f"prediction {name!r} is malformed") from None
        dtype, count = np.dtype(item["dtype"]), 1
        for d in shape:
            count *= d  # Python integers: no overflow
        if len(raw) != count * dtype.itemsize:
            raise CandidateError(f"prediction {name!r} has inconsistent size")
        arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return arrays


def _write_frame(fd, message):
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")
    data = memoryview(HEADER.pack(len(payload)) + payload)
    while data:
        data = data[os.write(fd, data):]


def _drain(stream, sink, limit=6000):
    for chunk in iter(lambda: stream.read(4096), b""):
        sink.append(chunk)
        while sum(map(len, sink)) > limit and len(sink) > 1:
            sink.pop(0)


def _rss_mb(pid):
    """Resident memory in MB of the process and its descendants: psutil when installed, else /proc or ps."""
    try:
        import psutil
    except ImportError:
        return _rss_mb_without_psutil(pid)
    try:
        process = psutil.Process(pid)
        return sum(p.memory_info().rss for p in [process, *process.children(recursive=True)]) / (1 << 20)
    except Exception:  # noqa: BLE001 - process gone
        return 0.0


def _proc_resident_pages(pid, depth=0):
    """Linux: resident pages of a process and of its descendants (listed in /proc/<pid>/task/<tid>/children)."""
    pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
    if depth < 8:
        for listing in Path(f"/proc/{pid}/task").glob("*/children"):
            for child in listing.read_text().split():
                try:
                    pages += _proc_resident_pages(int(child), depth + 1)
                except (OSError, ValueError, IndexError):
                    pass
    return pages


def _rss_mb_without_psutil(pid):
    """Resident memory without psutil: /proc on Linux, else the child's process group in ``ps``; 0.0 if unreadable."""
    try:
        if Path("/proc/self/statm").is_file():
            return _proc_resident_pages(pid) * os.sysconf("SC_PAGE_SIZE") / (1 << 20)
        listing = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True, timeout=5).stdout
        rows = (line.split() for line in listing.splitlines())  # the child leads its own process group
        return sum(int(row[1]) for row in rows if len(row) == 2 and row[0] == str(pid)) / 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0.0


def _read_result(out_r, process, wall_seconds, memory_mb, stderr_tail):
    deadline = time.monotonic() + wall_seconds
    buffer, peak = bytearray(), 0.0
    while True:
        if len(buffer) >= HEADER.size:
            (size,) = HEADER.unpack(bytes(buffer[:HEADER.size]))
            if size > MAX_FRAME:
                raise CandidateError(f"child message of {size} bytes exceeds the {MAX_FRAME}-byte limit")
            if len(buffer) >= HEADER.size + size:
                try:
                    return json.loads(bytes(buffer[HEADER.size:HEADER.size + size]).decode("utf-8")), peak
                except ValueError as error:
                    raise CandidateError(f"malformed child message: {error}") from None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CandidateError(f"training and prediction exceeded the {wall_seconds:.0f} s wall-clock limit")
        ready, _, _ = select.select([out_r], [], [], min(remaining, 0.5))
        rss = _rss_mb(process.pid)
        peak = max(peak, rss)
        if memory_mb and rss > memory_mb:
            raise CandidateError(f"the training process exceeded the {memory_mb} MB memory limit")
        if ready:
            chunk = os.read(out_r, 1 << 20)
            if not chunk:
                process.wait(timeout=10)
                code = process.returncode
                if code > 128:  # isolate's waiting process exits with 128 + the signal that ended the worker
                    code = 128 - code
                reason = {-24: "CPU-time limit", -9: "kill signal", -25: "file-size limit (attempted file write)"}.get(code, f"exit status {code}")
                tail = b"".join(stderr_tail).decode("utf-8", "replace").strip().splitlines()
                raise CandidateError(f"training process ended without a result ({reason})" + (f": {tail[-1]}" if tail else ""))
            buffer += chunk


def os_sandbox(workdir, deny_read):
    """The kernel mechanism that hides ``deny_read`` from the training child, or None (audit hook only)."""
    return kaggle_sandbox.os_sandbox(deny_read, [], workdir, needed=(WORKER, LIB))


def run_training(source, algorithm, splits, budget, *, wall_seconds, memory_mb=4096, threads=4, filename="candidate.py",
                 deny_read=None):
    """Train and predict in the child; return (predictions {name: int array}, details, peak RSS in MB).

    ``deny_read`` lists the directories the child must not read (default: clrs_data.isolation(), the task cache).
    """
    deny_read = [os.path.realpath(p) for p in (clrs_data.isolation()["deny_read"] if deny_read is None else deny_read)]
    workdir = os.path.realpath(tempfile.mkdtemp(prefix="artask-clrs-"))
    job_r = job_w = out_r = out_w = None
    process = drain = None
    stderr_tail = []
    try:
        data_path = Path(workdir) / "job.npz"
        np.savez(data_path, **{f"{split}/{key}": value for split, arrays in splits.items() for key, value in arrays.items()})
        job_r, job_w = os.pipe()
        out_r, out_w = os.pipe()
        env = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PATH": "/usr/bin:/bin",
               "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
               "VECLIB_MAXIMUM_THREADS": str(threads), "HOME": workdir, "TMPDIR": workdir}
        sandbox = os_sandbox(workdir, deny_read)
        argv = [sys.executable, "-s", str(WORKER), str(job_r), str(out_w)]
        if sandbox == "sandbox-exec":
            argv = [kaggle_sandbox.SANDBOX_EXEC, "-p", kaggle_sandbox._profile(deny_read, [], workdir)] + argv
        process = subprocess.Popen(argv, pass_fds=(job_r, out_w),
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   cwd=workdir, env=env, start_new_session=True, close_fds=True)
        os.close(job_r)
        os.close(out_w)
        job_r = out_w = None
        drain = threading.Thread(target=_drain, args=(process.stderr, stderr_tail), daemon=True)
        drain.start()
        cpu_seconds = int(wall_seconds * (threads + 1)) + 60
        # The stated limit is the resident memory polled in _read_result. The child's own limit (address space,
        # applied on Linux only) is a backstop against a single huge allocation between two polls; it gets a
        # margin for what is mapped but not resident (libraries, thread stacks, allocator arenas).
        address_space_mb = memory_mb + ADDRESS_SPACE_MARGIN_MB if memory_mb else 0
        _write_frame(job_w, {"lib": str(LIB), "data_path": str(data_path), "source": source, "filename": filename,
                             "algorithm": algorithm, "budget": dict(budget, threads=threads),
                             "limits": {"cpu_seconds": cpu_seconds, "memory_mb": address_space_mb},
                             "namespaces": sandbox == "linux-namespaces", "deny_read": deny_read, "allow_read": []})
        message, peak = _read_result(out_r, process, wall_seconds, memory_mb, stderr_tail)
        if isinstance(message, dict) and message.get("type") == "setup":
            raise RuntimeError(f"the training process could not enter its namespaces: {message.get('error')}"[:1500])
        if not isinstance(message, dict) or message.get("type") != "result":
            raise CandidateError("training process sent an unknown message")
        if not message.get("ok"):
            where = [line.strip() for line in str(message.get("traceback", "")).splitlines() if line.strip().startswith("File ")]
            raise CandidateError((f"{message.get('error')}" + (f" (at {where[-1]})" if where else ""))[:1500])
        details = message.get("details")
        if not isinstance(details, dict):
            raise CandidateError("training process sent no diagnostics")
        return decode_arrays(message.get("predictions")), dict(details, os_sandbox=sandbox or "audit-hook-only"), peak
    finally:
        for fd in (job_r, job_w, out_r, out_w):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, 9)
                except OSError:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            if drain is not None:
                drain.join(timeout=5)
            if process.stderr is not None:
                process.stderr.close()
        shutil.rmtree(workdir, ignore_errors=True)
