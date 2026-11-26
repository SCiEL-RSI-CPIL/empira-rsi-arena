"""Evaluator side of one model run of the CSPLib reformulation harness: build and extract in a sandboxed child.

``run_model`` starts ``or_csp_model_child.py`` in a ``software_sandbox`` child (clean interpreter, audit-hook
policy, CPU, wall-clock and memory limits) for one instance and one model source, the candidate's or the baseline's
(both go through the same path, so both are timed the same way). The child builds the model, transforms it with
CPMpy and sends the ``CpModelProto`` in protobuf text format; this process solves it in the clean solver worker
(``or_csp_cpsat.SolverWorker``) and sends back the solution vector; the child runs the extractor and sends the
answer, which the caller checks with the independent checker in this process.

Times: the child's CPU time is read from outside, from the kernel's CPU clock of the child process, when each
protocol message arrives (the child is then blocked waiting for the reply). Build time = CPU from "ready" (sent
before any statement of the model source runs) to "model"; extract time = CPU from "model" to "answer" (an honest
child uses none while the solver runs). Whatever the model code does to the child, it cannot move work out of
these windows or change what this process measures. A protocol message out of order, a message after a decided
outcome or an answer that is not JSON-native makes the candidate invalid.
"""
from __future__ import annotations

import base64
import binascii
import math
import os
import sys
import time
import zlib
from pathlib import Path

from . import software_sandbox
from .runtime import CandidateError

CHILD = "or_csp_model_child.py"
MEMORY_MB = 8192
CPU_MARGIN = 30  # CPU seconds the child may use beyond the cap before it is killed (start-up, imports, warm-up)
MAX_MODEL_TEXT = 1 << 30


def _children_on_linux():
    pids = set()
    try:
        for tid in os.listdir("/proc/self/task"):
            pids.update(int(p) for p in Path(f"/proc/self/task/{tid}/children").read_text().split())
        return pids
    except OSError:
        pass
    for entry in os.listdir("/proc"):
        try:
            if entry.isdigit() and int(Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()[1]) == os.getpid():
                pids.add(int(entry))
        except (OSError, ValueError, IndexError):
            continue
    return pids


def sandbox_child_pid():
    """PID of the one ``software_worker.py`` child of this process (found by the evaluator, never told by the child)."""
    worker = str(software_sandbox.WORKER)
    if os.path.isdir("/proc/self/task"):
        found = []
        for pid in _children_on_linux():
            try:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if len(argv) > 2 and os.fsdecode(argv[2]) == worker:
                found.append(pid)
    else:
        try:
            import psutil
        except ImportError as error:
            raise RuntimeError("measuring the model process's CPU time needs /proc (Linux) or psutil") from error
        found = [p.pid for p in psutil.Process().children() if worker in p.cmdline()[2:3]]
    if len(found) != 1:
        raise RuntimeError(f"expected one sandboxed model process, found {len(found)}")
    return found[0]


def process_cpu_seconds(pid):
    """CPU seconds (user + system, all threads, exited ones included) of another process, read from outside."""
    if sys.platform.startswith("linux"):
        try:
            return time.clock_gettime((~pid << 3) | 2)  # the process's CPU clock (MAKE_PROCESS_CPUCLOCK(pid, SCHED))
        except OSError:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    import psutil
    times = psutil.Process(pid).cpu_times()
    return times.user + times.system


class ModelRun:
    """The oracle of one sandboxed model run: drives the protocol, measures, solves and records the outcome."""

    def __init__(self, worker, cap, is_candidate):
        self.worker, self.cap, self.is_candidate = worker, float(cap), is_candidate
        self.phase, self.pid, self.marks = "started", None, {}
        self.record, self.answer, self.error = {}, None, None

    def fail(self, message):
        """Decide the run as the model's failure; the child only learns to stop (it cannot catch and go on)."""
        if self.error is None:
            self.error = CandidateError(message) if self.is_candidate else RuntimeError("baseline model: " + message)
        self.phase = "done"
        return {"stop": True}

    def timeout(self, runtime):
        self.record.update(status="TIMEOUT", runtime=runtime)
        self.phase = "done"
        return {"stop": True}

    def __call__(self, request):
        phase = request.get("phase") if isinstance(request, dict) else None
        expected = {"started": "ready", "ready": "model", "solved": "answer"}.get(self.phase)
        if self.phase == "done" or phase != expected:
            return self.fail(f"model process protocol violation: {phase!r} while expecting {expected!r}")
        if phase == "ready":
            self.pid = sandbox_child_pid()
            self.marks["ready"] = process_cpu_seconds(self.pid)
            self.phase = "ready"
            return {}
        cpu = process_cpu_seconds(self.pid)
        if phase == "model":
            self.marks["model"] = cpu
            return self.solve(request)
        self.record["extract_cpu"] = cpu - self.marks["model"]
        try:
            from .or_csp_cpsat import plain
            self.answer = plain(request.get("answer"))
        except CandidateError as error:
            return self.fail(str(error))
        self.record["runtime"] = self.record["build_cpu"] + float(self.record["solve_cpu"]) + self.record["extract_cpu"]
        self.record["status"] = "SOLVED" if self.record["runtime"] <= self.cap else "TIMEOUT"
        self.phase = "done"
        return {}

    def model_text(self, request):
        if isinstance(request.get("text"), str):
            return request["text"].encode()
        try:
            stream = zlib.decompressobj()
            text = stream.decompress(base64.b64decode(request["zlib"], validate=True), MAX_MODEL_TEXT)
        except (KeyError, TypeError, ValueError, binascii.Error, zlib.error):
            return None
        return None if stream.unconsumed_tail else text

    def solve(self, request):
        build_cpu = self.marks["model"] - self.marks["ready"]
        self.record["build_cpu"] = build_cpu
        if build_cpu >= self.cap:
            return self.timeout(build_cpu)
        text = self.model_text(request)
        if text is None:
            return self.fail("the model process sent no readable model text")
        reply = self.worker.solve(text, self.cap - build_cpu, encoding="text")
        self.record.update(status=reply["status"], solve_cpu=reply.get("cpu_seconds"), deterministic_time=reply.get("deterministic_time"),
                           conflicts=reply.get("conflicts"), branches=reply.get("branches"),
                           num_variables=reply.get("num_variables"), num_constraints=reply.get("num_constraints"))
        status = reply["status"]
        if status == "WORKER_ERROR":
            raise RuntimeError("CP-SAT worker failed: " + reply.get("error", ""))
        if status == "MODEL_INVALID":
            return self.fail("OR-Tools rejected the transformed model: " + reply.get("validation", ""))
        if status == "MODEL_REJECTED":
            return self.fail("the transformed model uses " + reply.get("reason", "") + ", which a CPMpy satisfaction model does not produce")
        if status == "INFEASIBLE":
            if not self.is_candidate:
                raise RuntimeError("baseline claims a generated instance is infeasible")
            return self.fail("the model claims the (satisfiable) instance is infeasible")
        if status == "UNKNOWN":
            return self.timeout(build_cpu + float(reply.get("cpu_seconds") or 0.0))
        self.phase = "solved"
        return {"solution": reply["solution"]}


def run_model(source, filename, data, worker, cap, *, is_candidate=True):
    """Build, transform, solve and extract one instance in a sandboxed child; returns (record, answer or None).

    record: status SOLVED or TIMEOUT, runtime (CPU s, the quantity scored), build_cpu, solve_cpu, extract_cpu and
    the solver's counters. A child killed at its CPU or wall-clock limit while building or extracting is a TIMEOUT.
    """
    run = ModelRun(worker, cap, is_candidate)
    cpu_limit = int(math.ceil(cap)) + CPU_MARGIN  # a backstop only: the cap is applied to the measured windows
    try:
        software_sandbox.execute(software_sandbox.harness_source(CHILD), "build_and_extract", [source, filename, data], oracle=run,
                                 cpu_seconds=cpu_limit, wall_seconds=2 * cpu_limit + cap + 90, memory_mb=MEMORY_MB,
                                 filename="<or_csp model child>")
    except software_sandbox.SandboxFailure as failure:
        if run.phase != "done" and run.error is None:  # after a decided outcome a failure of the child changes nothing
            if failure.kind == "timeout" and run.phase in ("ready", "solved"):
                run.record.setdefault("build_cpu", run.cap)
                run.timeout(run.cap)
            else:
                message = str(failure)
                message = message[len("ModelError: "):] if message.startswith("ModelError: ") else f"model process failed ({failure.kind}): {message}"
                run.fail(message)
    if run.error is not None:
        raise run.error
    if run.phase != "done":
        raise (CandidateError if is_candidate else RuntimeError)("the model process ended before it sent its answer")
    record = dict(run.record)
    return record, (run.answer if record["status"] == "SOLVED" else None)
