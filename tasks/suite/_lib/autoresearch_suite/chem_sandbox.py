"""Run one candidate function in a child process that only receives the inputs it may see.

``run_candidate(module, entry, args)`` executes ``entry(*args)`` from the
candidate's source file inside ``chem_worker.py`` and returns the
JSON-decoded result. The child gets a clean environment, an empty temporary
working directory, at most ``threads`` threads for numerical libraries (the
shared machine allows four), a CPU-time limit, a wall-clock limit and a
resident-memory limit, and an audit hook that hides the prepared data
directory and the evaluator package (see chem_worker.py). The labels of
held-out items stay in this (evaluator) process.

The suite's evaluator template imports the candidate module once in the
evaluator process before scoring; every scoring-relevant candidate call of the
prediction tasks happens in the child. Framing helpers are shared with
software_sandbox.
"""
from __future__ import annotations

import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .chem_common import data_dir
from .runtime import CandidateError
from .software_sandbox import SandboxFailure, _drain, _Reader, _rss_mb, _write_frame

WORKER = Path(__file__).resolve().with_name("chem_worker.py")


def hidden_roots():
    """Directories a candidate must not read: prepared data and the evaluator library."""
    return [str(data_dir().parent), str(Path(__file__).resolve().parent)]


def execute(source, entry, args=(), *, wall_seconds=120, cpu_seconds=None, memory_mb=4096, threads=4, seed=0,
            filename="<candidate>"):
    """Run ``entry(*args)`` in the child; return its value or raise SandboxFailure."""
    cpu_seconds = cpu_seconds or int(wall_seconds * threads + 30)
    job_r, job_w = os.pipe()
    out_r, out_w = os.pipe()
    workdir = tempfile.mkdtemp(prefix="artask-chem-")
    count = str(int(threads))
    env = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PATH": "/usr/bin:/bin",
           "HOME": workdir, "TMPDIR": workdir, "MPLCONFIGDIR": workdir, "OMP_NUM_THREADS": count,
           "OPENBLAS_NUM_THREADS": count, "MKL_NUM_THREADS": count, "VECLIB_MAXIMUM_THREADS": count,
           "NUMEXPR_NUM_THREADS": count, "RAYON_NUM_THREADS": count}
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
        _write_frame(job_w, {"source": source, "entry": entry, "args": list(args), "seed": int(seed), "filename": filename,
                             "hidden": hidden_roots(), "limits": {"cpu_seconds": int(cpu_seconds)}})
        reader = _Reader(out_r)
        deadline = time.monotonic() + wall_seconds
        while True:
            message = reader.frame()
            if message is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SandboxFailure("timeout", f"exceeded the {wall_seconds:.0f} s wall-clock limit")
                ready, _, _ = select.select([out_r], [], [], min(remaining, 0.2))
                if memory_mb and _rss_mb(process.pid) > memory_mb:
                    raise SandboxFailure("memory", f"exceeded the {memory_mb} MB memory limit")
                if ready and not reader.feed():
                    process.wait(timeout=5)
                    code = process.returncode
                    reason = {-24: "CPU-time limit", -9: "kill signal", -25: "file-size limit"}.get(code, f"exit status {code}")
                    kind = "timeout" if code in (-24, -9) else "crash"
                    raise SandboxFailure(kind, f"child process ended without a result ({reason})",
                                         b"".join(stderr_tail).decode("utf-8", "replace"))
                continue
            if not isinstance(message, dict) or message.get("type") != "result":
                raise SandboxFailure("protocol", "unexpected child message")
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


def run_candidate(module, entry, args=(), **kwargs):
    """Run a candidate entry point in the child; any failure is the candidate's (CandidateError)."""
    if not callable(getattr(module, entry, None)):
        raise CandidateError(f"candidate must define {entry}()")
    path = getattr(module, "__file__", None)
    if not path or not Path(path).is_file():
        raise CandidateError("candidate module has no source file")
    source = Path(path).read_text(encoding="utf-8")
    try:
        return execute(source, entry, args, filename=Path(path).name, **kwargs)
    except SandboxFailure as failure:
        last = failure.traceback_text.strip().splitlines()[-1] if failure.traceback_text.strip() else ""
        detail = f"; {last}" if failure.kind == "exception" and last and last not in str(failure) else ""
        raise CandidateError(f"candidate {entry}() failed in the sandbox ({failure.kind}): {failure}{detail}"[:2000]) from None
