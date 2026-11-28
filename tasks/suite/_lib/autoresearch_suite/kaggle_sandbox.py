"""Run a candidate's ``fit_predict`` in a child process that cannot see held-out labels.

``run_fit_predict(program_path, job)`` starts ``kaggle_worker.py`` with a clean environment (4 BLAS/OpenMP
threads, ``PYTHONHASHSEED=0``), an empty temporary working directory, a memory limit (resident-set
polling) and a wall-clock budget, and returns the candidate's predictions as float64 arrays. The
candidate file is read as text and executed in the child only.

Isolation of held-out labels:

* the job names only public input files (training data with labels, test inputs without labels);
  the label files of every task live in ``<data_dir>/<task>/private``;
* on macOS the child runs under ``sandbox-exec`` with a profile that denies reading the whole data
  directory except the task's public directory (and the raw download cache), denies writes outside
  the working directory and denies network access;
* on Linux, where the kernel lets an unprivileged process create user namespaces, the child runs in
  its own user, mount, PID, network and IPC namespaces: the data directory and the raw download cache
  are empty except for the task's public directory, every file system is read-only except the
  working directory and /dev/shm, there is no network and no other process is visible
  (``kaggle_worker.isolate``). Both are enforced by the kernel and also cover native code;
* everywhere, an audit hook in the child refuses Python-level reads outside the Python installation,
  the public directory and the working directory, writes outside the working directory, sockets and
  child processes.

The result names what was in force (``os_sandbox``: ``sandbox-exec``, ``linux-namespaces`` or
``audit-hook-only``). Residual risks (documented in the task notes): with ``audit-hook-only``, i.e.
where neither kernel mechanism is available (containers that forbid user namespaces, kernels before
5.12), deliberately hostile code can bypass the audit hook: native code (ctypes, native file readers
such as ``torch.from_file``) and process creation through ``_imp.create_builtin`` raise no audit
event. The thread count is a setting of the numerical libraries, not a limit of the operating system.

The budget clock starts when the child reports that the inputs are loaded, before the candidate's
source is executed, so imports and module-level code count; the child is killed ``GRACE_S`` seconds
after the budget.
"""
from __future__ import annotations

import functools
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
import zipfile
from pathlib import Path

import numpy as np

from .runtime import CandidateError

HEADER = struct.Struct(">Q")
WORKER = Path(__file__).resolve().with_name("kaggle_worker.py")
GRACE_S = 15.0
LOAD_LIMIT_S = 120.0
THREADS = 4
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
LINUX_NAMESPACES = True  # use kaggle_worker.isolate where the kernel permits it
MAX_OUTPUT_BYTES = 1 << 30  # predictions of the largest task take 0.11 GB


def _write_frame(fd, message):
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")
    view = memoryview(HEADER.pack(len(payload)) + payload)
    while view:
        view = view[os.write(fd, view):]


class _Reader:
    def __init__(self, fd):
        self.fd, self.buffer = fd, bytearray()

    def feed(self):
        chunk = os.read(self.fd, 1 << 16)
        self.buffer += chunk
        return bool(chunk)

    def frame(self):
        if len(self.buffer) < HEADER.size:
            return None
        (size,) = HEADER.unpack(bytes(self.buffer[:HEADER.size]))
        if size > (1 << 20):
            raise CandidateError("sandboxed child sent an oversized message")
        if len(self.buffer) < HEADER.size + size:
            return None
        payload = bytes(self.buffer[HEADER.size:HEADER.size + size])
        del self.buffer[:HEADER.size + size]
        try:
            message = json.loads(payload.decode("utf-8"))
        except ValueError:
            message = None
        if not isinstance(message, dict):
            raise CandidateError("sandboxed child sent an unknown message")
        return message


def _rss_mb(pid):
    import psutil
    try:
        process = psutil.Process(pid)
        return sum(p.memory_info().rss for p in [process, *process.children(recursive=True)]) / (1 << 20)
    except psutil.Error:  # process gone
        return 0.0


def _profile(deny_read, allow_read, workdir):
    def quoted(path):
        return json.dumps(os.path.realpath(path))
    rules = ["(version 1)", "(allow default)", "(deny network*)"]
    rules += [f"(deny file-read* (subpath {quoted(p)}))" for p in deny_read if Path(p).exists()]
    rules += [f"(allow file-read* (subpath {quoted(p)}))" for p in allow_read]
    rules += ["(deny file-write*)", f"(allow file-write* (subpath {quoted(workdir)}) (literal \"/dev/null\") (literal \"/dev/tty\"))"]
    return "".join(rules)


def _clean_env(workdir):
    threads = str(THREADS)
    return {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PATH": "/usr/bin:/bin",
            "HOME": workdir, "TMPDIR": workdir, "MPLCONFIGDIR": workdir, "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads, "VECLIB_MAXIMUM_THREADS": threads,
            "NUMEXPR_MAX_THREADS": threads, "NUMEXPR_NUM_THREADS": threads, "NUMBA_NUM_THREADS": threads,
            "LOKY_MAX_CPU_COUNT": threads, "KMP_DUPLICATE_LIB_OK": "TRUE"}


@functools.lru_cache(maxsize=64)
def _namespaces_work(deny_read, allow_read):
    """Whether kaggle_worker.isolate succeeds for these directories (one short child process, cached)."""
    workdir = os.path.realpath(tempfile.mkdtemp(prefix="artask-kaggle-probe-"))
    try:
        spec = json.dumps({"deny_read": list(deny_read), "allow_read": list(allow_read)})
        done = subprocess.run([sys.executable, "-s", str(WORKER), "--probe", spec], cwd=workdir, env=_clean_env(workdir),
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def os_sandbox(deny_read, allow_read, workdir, needed=()):
    """The kernel mechanism available for a run: ``sandbox-exec``, ``linux-namespaces`` or None (audit hook only).

    ``needed`` names further paths the child must keep (its worker script, a library directory); also used
    for the training child of clrs_sandbox, whose worker calls ``kaggle_worker.isolate``.
    """
    if sys.platform == "darwin" and os.path.exists(SANDBOX_EXEC):
        return "sandbox-exec"
    if not (LINUX_NAMESPACES and sys.platform.startswith("linux")):
        return None
    denied = [os.path.realpath(p) for p in deny_read if os.path.isdir(p)]
    needed = [os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.executable, WORKER, workdir, *needed)]
    if any(path == root or path.startswith(root + os.sep) for path in needed for root in denied):
        return None  # hiding a denied directory would hide the interpreter or the working directory
    return "linux-namespaces" if _namespaces_work(tuple(denied), tuple(os.path.realpath(p) for p in allow_read)) else None


def _drain(stream, sink, limit=6000):
    for chunk in iter(lambda: stream.read(4096), b""):
        sink.append(chunk)
        while sum(map(len, sink)) > limit and len(sink) > 1:
            sink.pop(0)


def _load_predictions(path):
    """The arrays the child saved; their size is checked before anything is read into this process."""
    try:
        with zipfile.ZipFile(path) as archive:
            size = sum(info.file_size for info in archive.infolist())
        if size > MAX_OUTPUT_BYTES:
            raise CandidateError(f"predictions take {size} bytes, more than the {MAX_OUTPUT_BYTES} bytes accepted")
        with np.load(path, allow_pickle=False) as data:
            return [np.asarray(data[f"a{i}"], dtype=np.float64) for i in range(len(data.files))]
    except (OSError, ValueError, KeyError, MemoryError, zipfile.BadZipFile) as error:
        raise CandidateError(f"sandboxed child wrote unreadable predictions ({type(error).__name__}: {error})"[:500]) from None


def run_fit_predict(program_path, job, *, budget_s, deny_read, allow_read, memory_mb=3500, output="array"):
    """Run ``fit_predict`` from ``program_path`` on the job's public inputs; return a list of float64 arrays.

    ``job`` holds the worker's input specs (``train``, ``test``); ``deny_read`` lists directories the child
    must not read (the data root, raw caches); ``allow_read`` lists the public input directories.
    """
    source = Path(program_path).read_text(encoding="utf-8")
    if memory_mb:
        try:
            import psutil  # noqa: F401
        except ImportError:
            raise RuntimeError("psutil is required to enforce the memory limit of the sandboxed child (requirement py:psutil)") from None
    workdir = os.path.realpath(tempfile.mkdtemp(prefix="artask-kaggle-"))
    command = [sys.executable, "-s", str(WORKER)]
    job_r, job_w = os.pipe()
    out_r, out_w = os.pipe()
    process, stderr_tail, drain = None, [], None
    started, phase = time.monotonic(), "load"
    try:
        sandbox = os_sandbox(deny_read, allow_read, workdir)
        argv = command + [str(job_r), str(out_w)]
        if sandbox == "sandbox-exec":
            argv = [SANDBOX_EXEC, "-p", _profile(deny_read, allow_read, workdir)] + argv
        process = subprocess.Popen(argv, pass_fds=(job_r, out_w), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, cwd=workdir, env=_clean_env(workdir), start_new_session=True, close_fds=True)
        os.close(job_r)
        os.close(out_w)
        job_r = out_w = None
        drain = threading.Thread(target=_drain, args=(process.stderr, stderr_tail), daemon=True)
        drain.start()
        _write_frame(job_w, {**job, "source": source, "filename": Path(program_path).name, "budget_s": float(budget_s),
                             "threads": THREADS, "allow_read": [os.path.realpath(p) for p in allow_read], "output": output,
                             "namespaces": sandbox == "linux-namespaces", "deny_read": [os.path.realpath(p) for p in deny_read]})
        reader, deadline = _Reader(out_r), started + LOAD_LIMIT_S
        budget_started = None
        while True:
            message = reader.frame()
            if message is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if phase == "load":
                        raise RuntimeError(f"starting the sandboxed child and loading the public inputs took more than {LOAD_LIMIT_S:.0f} s")
                    raise CandidateError(f"fit_predict exceeded the time budget of {budget_s:.0f} s (killed after {budget_s + GRACE_S:.0f} s)")
                ready, _, _ = select.select([out_r], [], [], min(remaining, 0.2))
                if memory_mb and _rss_mb(process.pid) > memory_mb:
                    raise CandidateError(f"fit_predict exceeded the {memory_mb} MB memory limit")
                if ready and not reader.feed():
                    process.wait(timeout=10)
                    tail = b"".join(stderr_tail).decode("utf-8", "replace").strip().splitlines()[-3:]
                    raise CandidateError(f"sandboxed child ended without a result (exit status {process.returncode}): {' | '.join(tail)}"[:1500])
                continue
            kind = message.get("type")
            if phase == "load" and kind == "ready":  # sent once, before any candidate code runs
                phase, budget_started = "fit", time.monotonic()
                deadline = budget_started + budget_s + GRACE_S
                continue
            if phase == "load" and kind == "setup":
                raise RuntimeError(f"the sandboxed child could not enter its namespaces: {message.get('error')}"[:1500])
            if phase == "load" and kind == "result" and not message.get("ok"):
                raise RuntimeError(f"the sandboxed child could not load the public inputs: {message.get('error')}"[:1500])
            if phase != "fit" or kind != "result":
                raise CandidateError("sandboxed child sent an unknown message")
            if time.monotonic() > deadline:
                raise CandidateError(f"fit_predict exceeded the time budget of {budget_s:.0f} s (result after {budget_s + GRACE_S:.0f} s)")
            if not message.get("ok"):
                last = str(message.get("traceback", "")).strip().splitlines()[-1:] or [""]
                detail = f" ({last[0]})" if last[0] and last[0] not in str(message.get("error")) else ""
                raise CandidateError(f"fit_predict failed in the sandbox: {message.get('error')}{detail}"[:2000])
            path = Path(str(message.get("path"))).resolve()
            if path.parent != Path(workdir) or not path.is_file():
                raise CandidateError("sandboxed child reported an unexpected output path")
            arrays = _load_predictions(path)
            return arrays, {"fit_predict_seconds": round(time.monotonic() - budget_started, 3), "load_seconds": round(budget_started - started, 3),
                            "os_sandbox": sandbox or "audit-hook-only"}
    finally:
        for fd in (job_r, job_w, out_r, out_w):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if process is not None:
            try:
                os.killpg(process.pid, 9)  # also what the child may have left behind in its process group
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
