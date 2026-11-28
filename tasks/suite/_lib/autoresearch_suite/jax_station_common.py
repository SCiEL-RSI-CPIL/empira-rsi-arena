"""Shared helpers of the jax_station evaluators: the data directory and the child-process runner.

``data_dir()`` is the single place that resolves this package's prepared data
(``<task cache>/data/jax_station``, see arena_task_paths); every other path is
derived from it.

``run_child(script, job, ...)`` starts ``python -s -B <script> <result_fd>``
with a clean environment: CPU-only JAX, at most ``threads`` XLA/BLAS threads
(XLA's CPU thread pools honour NPROC/PJRT_NPROC; OpenMP/BLAS pools the usual
variables), PYTHONHASHSEED=0 and a private empty working/temporary directory
that is removed afterwards. The job is sent as JSON on stdin; the worker writes
one JSON document to ``result_fd``. The parent enforces a wall-clock limit and
polls the child's resident memory (psutil); the worker applies its own CPU-time
and file-size limits. The returned stats include the child's wall time, CPU time
(user + system, all threads, from RUSAGE_CHILDREN) and peak resident memory.
Failures raise ``ChildFailure`` with a short reason and the tail of the
child's stderr.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from arena_task_paths import data_root


def data_dir():
    return data_root() / "jax_station"


class ChildFailure(RuntimeError):
    """The child did not deliver a result; ``kind`` is timeout, memory, crash or protocol."""

    def __init__(self, kind, message, stderr_tail=""):
        super().__init__(message)
        self.kind, self.stderr_tail = kind, stderr_tail


def thread_env(threads=4):
    value = str(int(threads))
    return {"NPROC": value, "PJRT_NPROC": value, "OMP_NUM_THREADS": value, "MKL_NUM_THREADS": value,
            "OPENBLAS_NUM_THREADS": value, "VECLIB_MAXIMUM_THREADS": value, "NUMEXPR_NUM_THREADS": value}


def _rss_mb(pid):
    try:
        import psutil
        process = psutil.Process(pid)
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total / (1 << 20)
    except Exception:  # noqa: BLE001 - psutil missing or process gone
        return 0.0


def _drain(stream, sink, limit=20000):
    for chunk in iter(lambda: stream.read(4096), b""):
        sink.append(chunk)
        while sum(map(len, sink)) > limit and len(sink) > 1:
            sink.pop(0)


def run_child(script, job, *, wall_seconds, memory_mb, threads=4, python=None, extra_env=None):
    """Run ``script`` on ``job``; return (result dict, stats dict) or raise ChildFailure."""
    workdir = tempfile.mkdtemp(prefix="artask-jaxst-")
    env = {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "PYTHONHASHSEED": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "JAX_PLATFORMS": "cpu",
           "TF_CPP_MIN_LOG_LEVEL": "3", "LANG": "C.UTF-8", **thread_env(threads), **(extra_env or {})}
    read_fd, write_fd = os.pipe()
    stderr_tail, peak_mb = [], 0.0
    process = drain = None
    started, usage = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        process = subprocess.Popen([str(python or sys.executable), "-s", "-B", str(script), str(write_fd)],
                                   pass_fds=(write_fd,), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, cwd=workdir, env=env, start_new_session=True, close_fds=True)
        os.close(write_fd)
        write_fd = None
        drain = threading.Thread(target=_drain, args=(process.stderr, stderr_tail), daemon=True)
        drain.start()
        payload = json.dumps(job).encode("utf-8")
        feeder = threading.Thread(target=_feed, args=(process.stdin, payload), daemon=True)
        feeder.start()
        chunks = []
        reader = threading.Thread(target=lambda: chunks.extend(iter(lambda: os.read(read_fd, 1 << 20), b"")), daemon=True)
        reader.start()
        while reader.is_alive():
            reader.join(timeout=0.5)
            elapsed = time.time() - started
            peak_mb = max(peak_mb, _rss_mb(process.pid))
            if elapsed > wall_seconds:
                raise ChildFailure("timeout", f"exceeded the {wall_seconds:.0f} s wall-clock limit", _tail(stderr_tail))
            if memory_mb and peak_mb > memory_mb:
                raise ChildFailure("memory", f"exceeded the {memory_mb} MB memory limit ({peak_mb:.0f} MB)", _tail(stderr_tail))
        process.wait(timeout=60)
        drain.join(timeout=5)
        raw = b"".join(chunks)
        if not raw:
            code = process.returncode
            reason = {-24: "CPU-time limit", -9: "kill signal", -25: "file-size limit"}.get(code, f"exit status {code}")
            kind = "timeout" if code in (-24, -9) else "crash"
            raise ChildFailure(kind, f"child process ended without a result ({reason})", _tail(stderr_tail))
        try:
            result = json.loads(raw.decode("utf-8"))
        except ValueError as error:
            raise ChildFailure("protocol", f"malformed child result: {error}", _tail(stderr_tail)) from None
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (after.ru_utime - usage.ru_utime) + (after.ru_stime - usage.ru_stime)
        stats = {"wall_seconds": round(time.time() - started, 2), "cpu_seconds": round(cpu, 1),
                 "peak_rss_mb": round(peak_mb, 1), "threads": int(threads)}
        return result, stats
    finally:
        for fd in (read_fd, write_fd):
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
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            if drain is not None:
                drain.join(timeout=5)
            if process.stderr is not None:
                process.stderr.close()
        shutil.rmtree(workdir, ignore_errors=True)


def _feed(stream, payload):
    try:
        stream.write(payload)
        stream.close()
    except (BrokenPipeError, OSError):
        pass


def _tail(chunks, limit=3000):
    return b"".join(chunks).decode("utf-8", "replace")[-limit:]
