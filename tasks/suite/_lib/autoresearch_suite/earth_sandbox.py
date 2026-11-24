"""Run earth-task candidates in a child process that never receives held-out data.

``call(module, entry, args)`` executes ``entry(*args)`` from the candidate's source in
``earth_worker.py`` and returns the decoded result. ``run_online(module, ...)`` drives an
online forecasting protocol: trusted harness code in the child asks this process (through an
oracle) for the training data and then, step by step, for the information available at each
forecast time; the next step is released only after the forecast for the current step has
been submitted and validated here, so a candidate cannot see data from after a forecast time.

The child gets a clean environment (one BLAS/OpenMP thread, ``PYTHONHASHSEED=0``), an empty
temporary working directory, CPU, wall-clock and memory limits and the audit-hook policy of
earth_worker.py. On macOS it additionally runs under ``sandbox-exec`` with a profile that denies
network access, reading of the prepared data directory and of the raw download cache, and
writing outside the working directory.

The evaluators load the candidate with ``sandboxed.run``, so no statement of the candidate
is executed in the evaluator process. Residual risk (stated in every task's limitations): without
``sandbox-exec`` deliberately hostile native code can bypass the audit hook.
"""
from __future__ import annotations

import base64
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from arena_task_paths import cache_path, data_root
from .earth_common import data_dir
from .runtime import CandidateError
from .sandboxed import defines
from .software_sandbox import HEADER, SandboxFailure, _drain, _rss_mb, _write_frame

WORKER = Path(__file__).resolve().with_name("earth_worker.py")
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
MAX_FRAME = 512 << 20


def deny_read():
    """Directories the child must not read: the prepared data (wherever data_dir() points) and the raw cache."""
    return (data_dir(), data_root(), cache_path("earth-raw"))


def encode(value):
    """Python/numpy objects -> JSON transport form (arrays as exact base64 buffers)."""
    if isinstance(value, np.ndarray):
        kind = value.dtype.kind
        if kind not in "biuf":
            raise TypeError(f"cannot transport arrays of dtype {value.dtype}")
        array = np.ascontiguousarray(value, dtype=np.float64 if kind == "f" else np.int64)
        return {"__nd__": base64.b64encode(array.tobytes()).decode("ascii"), "dtype": array.dtype.str, "shape": list(array.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def decode(value):
    if isinstance(value, dict):
        if "__nd__" in value:
            try:
                raw = base64.b64decode(value["__nd__"], validate=True)
                dtype = np.dtype(value["dtype"])
                if dtype.kind not in "biuf":
                    raise ValueError(f"dtype {dtype}")
                return np.frombuffer(raw, dtype=dtype).reshape([int(n) for n in value["shape"]]).copy()
            except (TypeError, ValueError, KeyError) as error:
                raise CandidateError(f"malformed array in the candidate's output ({error})") from None
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


class _Reader:
    def __init__(self, fd):
        self.fd, self.buffer = fd, bytearray()

    def feed(self):
        chunk = os.read(self.fd, 1 << 20)
        self.buffer += chunk
        return bool(chunk)

    def frame(self):
        if len(self.buffer) < HEADER.size:
            return None
        (size,) = HEADER.unpack(bytes(self.buffer[:HEADER.size]))
        if size > MAX_FRAME:
            raise SandboxFailure("protocol", f"child message of {size} bytes exceeds the {MAX_FRAME}-byte limit")
        if len(self.buffer) < HEADER.size + size:
            return None
        payload = bytes(self.buffer[HEADER.size:HEADER.size + size])
        del self.buffer[:HEADER.size + size]
        try:
            return json.loads(payload.decode("utf-8"))
        except ValueError as error:
            raise SandboxFailure("protocol", f"malformed child message: {error}") from None


def _profile(workdir):
    def quoted(path):
        return json.dumps(os.path.realpath(path))
    rules = ["(version 1)", "(allow default)", "(deny network*)"]
    rules += [f"(deny file-read* (subpath {quoted(p)}))" for p in deny_read() if Path(p).exists()]
    rules += ["(deny file-write*)", f"(allow file-write* (subpath {quoted(workdir)}) (literal \"/dev/null\") (literal \"/dev/tty\"))"]
    return "".join(rules)


def kernel_sandbox_available():
    return sys.platform == "darwin" and os.path.exists(SANDBOX_EXEC)


def execute(source, entry, args=(), *, harness=None, oracle=None, cpu_seconds=120, wall_seconds=None, memory_mb=3000,
            filename="<candidate>", seed=0):
    """Run ``entry(*args[, oracle])`` in a sandboxed child; return its decoded value or raise SandboxFailure."""
    wall_seconds = wall_seconds or cpu_seconds * 1.5 + 15
    workdir = os.path.realpath(tempfile.mkdtemp(prefix="artask-earth-"))
    env = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PATH": "/usr/bin:/bin",
           "HOME": workdir, "TMPDIR": workdir, "MPLCONFIGDIR": workdir, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_MAX_THREADS": "1", "NUMBA_NUM_THREADS": "1",
           "KMP_DUPLICATE_LIB_OK": "TRUE"}
    job_r, job_w = os.pipe()
    out_r, out_w = os.pipe()
    process = drain = None
    stderr_tail = []
    try:
        argv = [sys.executable, "-s", str(WORKER), str(job_r), str(out_w)]
        if kernel_sandbox_available():
            argv = [SANDBOX_EXEC, "-p", _profile(workdir)] + argv
        process = subprocess.Popen(argv, pass_fds=(job_r, out_w), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, cwd=workdir, env=env, start_new_session=True, close_fds=True)
        os.close(job_r)
        os.close(out_w)
        job_r = out_w = None
        drain = threading.Thread(target=_drain, args=(process.stderr, stderr_tail), daemon=True)
        drain.start()
        _write_frame(job_w, {"source": source, "harness": harness, "entry": entry, "args": encode(list(args)),
                             "oracle": oracle is not None, "seed": seed, "filename": filename,
                             "limits": {"cpu_seconds": int(cpu_seconds), "memory_mb": int(memory_mb)}})
        reader = _Reader(out_r)
        deadline = time.monotonic() + wall_seconds
        while True:
            message = reader.frame()
            if message is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SandboxFailure("timeout", f"exceeded the {wall_seconds:.0f} s wall-clock limit")
                ready, _, _ = select.select([out_r], [], [], min(remaining, 0.1))
                if not ready and memory_mb and _rss_mb(process.pid) > memory_mb:
                    raise SandboxFailure("memory", f"exceeded the {memory_mb} MB memory limit")
                if ready and not reader.feed():
                    process.wait(timeout=5)
                    code = process.returncode
                    reason = {-24: "CPU-time limit", -9: "kill signal"}.get(code, f"exit status {code}")
                    tail = b"".join(stderr_tail).decode("utf-8", "replace")
                    raise SandboxFailure("timeout" if code in (-24, -9) else "crash", f"child process ended without a result ({reason})", tail)
                continue
            if not isinstance(message, dict):
                raise SandboxFailure("protocol", "child message is not an object")
            if message.get("type") == "oracle":
                if oracle is None:
                    raise SandboxFailure("protocol", "unexpected oracle request")
                try:
                    reply = {"response": encode(oracle(decode(message.get("request"))))}
                except CandidateError as error:
                    reply = {"error": str(error)}
                _write_frame(job_w, reply)
                continue
            if message.get("type") != "result":
                raise SandboxFailure("protocol", "unknown child message type")
            if message.get("ok"):
                return decode(message.get("value"))
            raise SandboxFailure("exception", str(message.get("error"))[:1500], str(message.get("traceback", ""))[-3000:])
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


def candidate_source(module, *names):
    """Source text of the candidate after a cheap check that the named entry points exist."""
    for name in names:
        if not defines(module, name):
            raise CandidateError(f"candidate must define {name}()")
    path = getattr(module, "__file__", None)
    if not path or not Path(path).is_file():
        raise CandidateError("candidate module has no source file")
    return Path(path).read_text(encoding="utf-8"), Path(path).name


def _candidate_failure(failure, entry):
    last = failure.traceback_text.strip().splitlines()[-1] if failure.traceback_text.strip() else ""
    detail = f"; {last}" if failure.kind == "exception" and last and last not in str(failure) else ""
    return CandidateError(f"candidate {entry}() failed in the sandbox ({failure.kind}): {failure}{detail}"[:2000])


def call(module, entry, args=(), **limits):
    """Run one candidate function on public inputs; any failure is the candidate's."""
    source, name = candidate_source(module, entry)
    try:
        return execute(source, entry, args, filename=name, **limits)
    except SandboxFailure as failure:
        raise _candidate_failure(failure, entry) from None


ONLINE_HARNESS = '''
def __earth_online__(fit_name, oracle):
    setup = oracle({"op": "setup"})
    model = globals()[fit_name](*setup["args"])
    forecast = getattr(model, "forecast", None)
    if not callable(forecast):
        raise TypeError(fit_name + "() must return an object with a forecast() method")
    store = __EarthHistory__()
    store.append(setup.get("history"))
    while True:
        step = oracle({"op": "next"})
        if step is None:
            return "done"
        store.append(step.get("append"))
        args = step.get("args", [])
        oracle({"op": "submit", "value": forecast(store.view(), *args) if setup.get("with_history", True) else forecast(*args)})


class __EarthHistory__:
    """Growing history with amortised appends; forecast() receives views of the filled part."""

    def __init__(self):
        self.buffers, self.size = None, 0

    def append(self, rows):
        import numpy as np
        if not rows:
            return
        rows = {key: np.asarray(value) for key, value in rows.items()}
        count = len(next(iter(rows.values())))
        if self.buffers is None:
            self.buffers = {k: np.empty((max(16, 2 * count),) + v.shape[1:], v.dtype) for k, v in rows.items()}
        capacity = len(next(iter(self.buffers.values())))
        if self.size + count > capacity:
            capacity = max(2 * capacity, self.size + count)
            for key, old in self.buffers.items():
                grown = np.empty((capacity,) + old.shape[1:], old.dtype)
                grown[:self.size] = old[:self.size]
                self.buffers[key] = grown
        for key, value in rows.items():
            self.buffers[key][self.size:self.size + count] = value
        self.size += count

    def view(self):
        return None if self.buffers is None else {key: buffer[:self.size] for key, buffer in self.buffers.items()}
'''


class OnlineSession:
    """Trusted oracle for the online protocol: releases one step at a time after each validated forecast.

    ``setup`` is ``{"args": [...], "history": {...} or None}``: the arguments of the candidate's fit
    function and the initial history (a dict of arrays sharing their first axis). ``steps`` yields
    dicts with ``append`` (new rows of the history, or None) and ``args`` (extra arguments of
    ``forecast(history, *args)``). ``validate(index, value)`` returns the cleaned forecast or raises
    CandidateError.
    """

    def __init__(self, setup, steps, validate):
        self.setup, self.steps, self.validate = setup, list(steps), validate
        self.outputs, self.index, self.awaiting, self.error, self.started = [], 0, False, None, False

    def __call__(self, request):
        if self.error:
            raise CandidateError(self.error)
        op = request.get("op") if isinstance(request, dict) else None
        if op == "setup" and not self.started:
            self.started = True
            return self.setup
        if op == "next" and self.started and not self.awaiting:
            if self.index >= len(self.steps):
                return None
            self.awaiting = True
            return self.steps[self.index]
        if op == "submit" and self.awaiting:
            try:
                self.outputs.append(self.validate(self.index, request.get("value")))
            except CandidateError as error:
                self.error = f"forecast {self.index}: {error}"
                raise CandidateError(self.error) from None
            self.index += 1
            self.awaiting = False
            return True
        self.error = f"protocol violation: unexpected oracle request {op!r} at step {self.index}"
        raise CandidateError(self.error)


def run_online(module, fit_name, setup, steps, validate, with_history=True, **limits):
    """Fit on ``setup`` and forecast every step in order; return the validated forecasts.

    With ``with_history=False`` the candidate's ``forecast(*args)`` gets only each step's arguments
    (no accumulated history), for protocols whose steps carry all permitted information themselves.
    """
    source, name = candidate_source(module, fit_name)
    session = OnlineSession({**setup, "with_history": bool(with_history)}, steps, validate)
    try:
        execute(source, "__earth_online__", [fit_name], harness=ONLINE_HARNESS, oracle=session, filename=name, **limits)
    except SandboxFailure as failure:
        if session.error:
            raise CandidateError(session.error) from None
        raise _candidate_failure(failure, fit_name) from None
    if session.error:
        raise CandidateError(session.error)
    if len(session.outputs) != len(session.steps):
        raise CandidateError(f"the candidate stopped after {len(session.outputs)} of {len(session.steps)} forecasts")
    return session.outputs

