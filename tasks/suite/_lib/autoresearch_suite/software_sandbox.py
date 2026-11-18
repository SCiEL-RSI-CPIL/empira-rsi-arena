"""Run candidate (or harness) code in a resource-limited child process.

``run(source, entry, args)`` executes ``entry(*args)`` from ``source`` inside
``software_worker.py`` and returns the JSON-decoded result. The child gets a
clean environment (``PYTHONHASHSEED=0``), an empty temporary working
directory, a CPU-time limit, a wall-clock limit, a memory limit (RLIMIT_AS on
Linux, resident-set polling elsewhere), a zero file-size limit and an audit
hook that blocks file writes, reads outside the Python installation, process
creation, sockets, ctypes and subinterpreters (see software_worker.py).

An optional ``oracle`` callable runs in this (trusted) process: the child
calls ``oracle(request)`` and receives its JSON answer. Oracles keep their own
budgets, so a candidate cannot obtain more answers than the task allows.

Failures of the child raise ``runtime.CandidateError`` (or ``HarnessError``
for trusted harness code) with a short reason. The suite's shared evaluator
template imports the candidate module once in the evaluator process before
scoring; every scoring-relevant call of candidate functions happens here.
"""
from __future__ import annotations

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

from .runtime import CandidateError

HEADER = struct.Struct(">Q")
WORKER = Path(__file__).resolve().with_name("software_worker.py")
MAX_FRAME = 64 << 20


class HarnessError(RuntimeError):
    """Raised when trusted harness code fails (an evaluator problem, not the candidate's)."""


class SandboxFailure(Exception):
    """Internal: the child failed; ``kind`` is exception, timeout, memory, crash or protocol."""

    def __init__(self, kind, message, traceback_text=""):
        super().__init__(message)
        self.kind, self.traceback_text = kind, traceback_text


def _write_frame(fd, message):
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")
    data = memoryview(HEADER.pack(len(payload)) + payload)
    while data:
        data = data[os.write(fd, data):]


class _Reader:
    """Incremental frame reader on a nonblocking-safe pipe."""

    def __init__(self, fd):
        self.fd, self.buffer = fd, bytearray()

    def feed(self):
        chunk = os.read(self.fd, 1 << 20)
        if not chunk:
            return False
        self.buffer += chunk
        return True

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


def _drain(stream, sink, limit=4000):
    for line in iter(lambda: stream.read(4096), b""):
        sink.append(line)
        while sum(map(len, sink)) > limit and len(sink) > 1:
            sink.pop(0)


def _rss_mb(pid):
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss / (1 << 20)
    except Exception:  # noqa: BLE001 - psutil missing or process gone
        return 0.0


def execute(source, entry, args=(), *, oracle=None, cpu_seconds=60, wall_seconds=None, memory_mb=2048,
            blocked_modules=(), seed=0, filename="<candidate>", recursion_limit=10000):
    """Run ``entry(*args[, oracle])`` in a sandboxed child; return its value or raise SandboxFailure."""
    wall_seconds = wall_seconds or cpu_seconds * 1.5 + 10
    job_r, job_w = os.pipe()
    out_r, out_w = os.pipe()
    workdir = tempfile.mkdtemp(prefix="artask-sandbox-")
    env = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PATH": "/usr/bin:/bin",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HOME": workdir, "TMPDIR": workdir}
    process = drain = None
    stderr_tail = []
    try:
        process = subprocess.Popen([sys.executable, "-s", str(WORKER), str(job_r), str(out_w)], pass_fds=(job_r, out_w),
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   cwd=workdir, env=env, start_new_session=True, close_fds=True)
        os.close(job_r)
        os.close(out_w)
        job_r = out_w = None
        drain = threading.Thread(target=_drain, args=(process.stderr, stderr_tail), daemon=True)
        drain.start()
        job = {"source": source, "entry": entry, "args": list(args), "oracle": oracle is not None, "seed": seed,
               "filename": filename, "recursion_limit": recursion_limit, "blocked_modules": list(blocked_modules),
               "limits": {"cpu_seconds": int(cpu_seconds), "memory_mb": int(memory_mb)}}
        _write_frame(job_w, job)
        reader = _Reader(out_r)
        deadline = time.monotonic() + wall_seconds
        while True:
            message = reader.frame()
            if message is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SandboxFailure("timeout", f"exceeded the {wall_seconds:.0f} s wall-clock limit")
                ready, _, _ = select.select([out_r], [], [], min(remaining, 0.1))
                if memory_mb and _rss_mb(process.pid) > memory_mb:
                    raise SandboxFailure("memory", f"exceeded the {memory_mb} MB memory limit")
                if ready and not reader.feed():
                    process.wait(timeout=5)
                    code = process.returncode
                    reason = {-24: "CPU-time limit", -9: "kill signal", -25: "file-size limit (attempted file write)"}.get(code, f"exit status {code}")
                    kind = "timeout" if code in (-24, -9) else "crash"
                    raise SandboxFailure(kind, f"child process ended without a result ({reason})", b"".join(stderr_tail).decode("utf-8", "replace"))
                continue
            if not isinstance(message, dict):
                raise SandboxFailure("protocol", "child message is not an object")
            if message.get("type") == "oracle":
                if oracle is None:
                    raise SandboxFailure("protocol", "unexpected oracle request")
                try:
                    reply = {"response": oracle(message.get("request"))}
                except CandidateError as error:
                    reply = {"error": str(error)}
                _write_frame(job_w, reply)
                continue
            if message.get("type") != "result":
                raise SandboxFailure("protocol", "unknown child message type")
            if message.get("ok"):
                return message.get("value")
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


def run_candidate(program_path, entry, args=(), **kwargs):
    """Run a candidate entry point; any failure is the candidate's (CandidateError)."""
    source = Path(program_path).read_text(encoding="utf-8")
    try:
        return execute(source, entry, args, filename=str(Path(program_path).name), **kwargs)
    except SandboxFailure as failure:
        last = failure.traceback_text.strip().splitlines()[-1] if failure.traceback_text.strip() else ""
        detail = f"; {last}" if failure.kind == "exception" and last and last not in str(failure) else ""
        raise CandidateError(f"candidate {entry}() failed in the sandbox ({failure.kind}): {failure}{detail}"[:2000]) from None


def run_harness(source, entry, args=(), **kwargs):
    """Run trusted harness code on untrusted data; returns (ok, value_or_failure)."""
    try:
        return True, execute(source, entry, args, filename="<harness>", **kwargs)
    except SandboxFailure as failure:
        return False, failure


def harness_source(name):
    """Source text of a self-contained harness module stored next to this file."""
    return Path(__file__).resolve().with_name(name).read_text(encoding="utf-8")


def require_entry(module, *names):
    """Cheap contract check on the imported candidate before sandboxed execution."""
    for name in names:
        if not callable(getattr(module, name, None)):
            raise CandidateError(f"candidate must define {name}()")
    path = getattr(module, "__file__", None)
    if not path or not Path(path).is_file():
        raise CandidateError("candidate module has no source file")
    return path
