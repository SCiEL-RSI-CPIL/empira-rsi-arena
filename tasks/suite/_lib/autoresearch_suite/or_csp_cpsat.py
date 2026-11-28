"""CP-SAT model-reformulation harness for the PDR / CSPLib tasks (package or_csp).

Candidates follow the contract evolved in "LLM-Guided Evolutionary Search for
Constraint Model Reformulation to Improve Solver Efficiency" (arXiv:2607.28268):
a CPMpy builder ``build_model(data) -> (model, extractor)`` whose extractor
returns the same JSON answer structure as the baseline model.

For every instance the candidate and the baseline go through the same pipeline
(``or_csp_modelrun.run_model``); no statement of the candidate runs in this process:

1. in a sandboxed child (``software_sandbox`` running ``or_csp_model_child.py``,
   a fresh one per instance and model) the model module's top-level code,
   ``build_model``, the CPMpy -> OR-Tools transformation and the export of the
   ``CpModelProto`` in protobuf text format run; their CPU time is read from
   outside the child, from the kernel's CPU clock of the child process;
2. the model is solved in a separate clean interpreter (``or_csp_cpsat_worker.py``,
   which never imports candidate code) with one search worker and random seed 42;
   its CPU time inside ``Solve`` is measured there with ``getrusage``;
3. back in the child the solver values are written into the CPMpy variables and
   the extractor runs (CPU time again read from outside); the answer is checked
   in this process by an independent checker of the CSPLib definition.

The per-instance runtime is the sum of these CPU times. An instance that is not
solved within its cap counts twice the cap (PAR2). A wrong answer or a claim of
infeasibility (every instance is satisfiable) makes the candidate invalid.
"""
from __future__ import annotations

import ast
import json
import math
import os
import select
import struct
import subprocess
import sys
import time
from pathlib import Path

from .runtime import CandidateError
from .sandboxed import defines

WORKER = Path(__file__).with_name("or_csp_cpsat_worker.py")
SOLVER_SEED = 42
ALLOWED_IMPORTS = {"cpmpy", "numpy", "math", "itertools", "collections", "functools", "operator", "typing",
                   "fractions", "heapq", "bisect", "dataclasses", "enum", "copy", "statistics", "string"}
FORBIDDEN_NAMES = {"__import__", "exec", "eval", "compile", "open", "input", "breakpoint", "globals", "vars", "locals"}


class Rng:
    """SplitMix64: a tiny generator whose stream never changes across Python or NumPy versions."""

    def __init__(self, seed):
        self.state = (int(seed) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF

    def next64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def randint(self, low, high):
        """Uniform integer in [low, high] (inclusive)."""
        return low + self.next64() % (high - low + 1)

    def random(self):
        return (self.next64() >> 11) / float(1 << 53)

    def shuffle(self, items):
        for i in range(len(items) - 1, 0, -1):
            j = self.randint(0, i)
            items[i], items[j] = items[j], items[i]
        return items

    def choice(self, items):
        return items[self.randint(0, len(items) - 1)]

    def sample(self, items, k):
        pool = list(items)
        self.shuffle(pool)
        return pool[:k]


def check_candidate_source(path):
    """Static guard on a file; returns its source text (see ``check_source``)."""
    try:
        source = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError) as error:  # ValueError: undecodable bytes
        raise CandidateError(f"cannot read candidate source: {error}") from error
    check_source(source)
    return source


def check_source(source):
    """Static guard, kept as defence in depth (the model code runs in a sandboxed child): cpmpy, numpy and a small
    standard-library subset only, no dunder attributes, no exec/eval/open and similar names."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as error:  # ValueError: a null byte
        raise CandidateError(f"cannot parse candidate source: {error}") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]] if node.level == 0 else ["<relative>"]
        else:
            roots = []
        bad = [root for root in roots if root not in ALLOWED_IMPORTS]
        if bad:
            raise CandidateError(f"import of {bad[0]!r} is not allowed (cpmpy, numpy and {sorted(ALLOWED_IMPORTS - {'cpmpy', 'numpy'})} only)")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise CandidateError(f"use of {node.id!r} is not allowed in model code")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in {"__init__", "__name__"}:
            raise CandidateError(f"access to dunder attribute {node.attr!r} is not allowed in model code")


class SolverWorker:
    """A persistent clean interpreter that solves serialized CP-SAT models."""

    def __init__(self, python=None):
        self.python = python or sys.executable
        self.process = None

    def _start(self):
        self.process = subprocess.Popen([self.python, "-I", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)

    def close(self):
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            self.process = None

    def _read(self, size, deadline):
        data = b""
        fd = self.process.stdout.fileno()
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                return None
            chunk = os.read(fd, size - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def solve(self, proto_bytes, time_limit, grace=60.0, encoding="text"):
        if self.process is None or self.process.poll() is not None:
            self._start()
        header = json.dumps({"time_limit": max(0.01, float(time_limit)), "seed": SOLVER_SEED, "encoding": encoding}).encode()
        try:
            self.process.stdin.write(struct.pack("<II", len(header), len(proto_bytes)) + header + proto_bytes)
            self.process.stdin.flush()
        except BrokenPipeError:
            self.close()
            return {"status": "WORKER_ERROR", "error": "solver process exited"}
        deadline = time.monotonic() + float(time_limit) + grace
        size = self._read(4, deadline)
        payload = self._read(struct.unpack("<I", size)[0], deadline) if size else None
        if payload is None:
            self.close()  # hung beyond the limit and grace: kill it and report a timeout
            return {"status": "UNKNOWN", "cpu_seconds": float(time_limit), "killed": True}
        return json.loads(payload)


def plain(value):
    """Convert an answer to JSON-native values of the exact built-in types (numpy scalars and arrays included).

    An instance of a subclass of int, float or str is replaced by the built-in value it holds, read with the base
    class's own conversion: the checkers compare and add these values, and a subclass could answer those operations
    as it likes (an ``int`` whose ``__eq__`` always agrees passed the BIBD checker with an all-zero matrix).
    """
    if isinstance(value, dict):
        return {plain_key(k): plain(v) for k, v in dict.items(value)}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if value is None or type(value) in (bool, int, float, str):
        return value
    for base, convert in ((int, int.__int__), (float, float.__float__), (str, str.__str__)):
        if isinstance(value, base):
            return convert(value)
    raise CandidateError(f"extractor returned a non-JSON value of type {type(value).__name__}")


def plain_key(key):
    key = plain(key)
    return key if type(key) is str else str(key)


def load_candidate(module):
    """(source, file name) of the candidate after the static guard has accepted it; it is never imported here.

    The evaluators hand over an unexecuted module (``sandboxed.run``: the file's path only). The source is read once,
    guarded and then executed only in the sandboxed model processes of ``or_csp_modelrun``.
    """
    path = getattr(module, "__file__", None)
    if not path or not Path(path).is_file():
        raise CandidateError("candidate module has no source file")
    source = check_candidate_source(path)
    if not defines(module, "build_model"):
        raise CandidateError("candidate must define build_model(data) -> (model, extractor)")
    return source, Path(path).name


def reformulation_score(module, problem, params, *, generate, check, baseline_source):
    """PAR2 speedup of the candidate over the baseline model: sum(baseline) / sum(candidate)."""
    from .or_csp_modelrun import run_model
    source, filename = load_candidate(module)
    cap_factor, cap_floor = float(params["cap_factor"]), float(params["cap_floor"])
    baseline_cap = float(params.get("baseline_cap", 60.0))
    worker = SolverWorker()
    rows, excluded = [], []
    try:
        for spec in params["instances"]:
            data = generate(**spec)
            # The baseline is solved first because its runtime sets the candidate's cap; its answer is checked here.
            base_probe, base_answer = run_model(baseline_source, f"<baseline {problem}>", data, worker, baseline_cap, is_candidate=False)
            if base_probe["status"] != "SOLVED":
                excluded.append({"instance": spec, "reason": "baseline did not solve within the baseline cap"})
                continue
            check(data, base_answer)  # evaluator self-check: the baseline answer must be correct
            cap = max(cap_floor, cap_factor * base_probe["runtime"])
            record, answer = run_model(source, filename, data, worker, cap)
            if record["status"] == "SOLVED":
                try:
                    check(data, answer)
                except CandidateError as error:
                    raise CandidateError(f"instance {spec}: {error}") from error
                cand = record["runtime"]
            else:
                cand = 2.0 * cap
            rows.append({"instance": spec, "baseline_seconds": round(base_probe["runtime"], 6), "candidate_seconds": round(cand, 6),
                         "candidate_status": record["status"], "cap_seconds": round(cap, 6),
                         "candidate_deterministic_time": record.get("deterministic_time"),
                         "baseline_deterministic_time": base_probe.get("deterministic_time"),
                         "candidate_build_cpu": round(record["build_cpu"], 6), "baseline_build_cpu": round(base_probe["build_cpu"], 6)})
    finally:
        worker.close()
    if len(rows) < max(1, len(params["instances"]) // 2):
        raise RuntimeError(f"too many instances excluded because the baseline timed out: {excluded}")
    total_base = sum(r["baseline_seconds"] for r in rows)
    total_cand = sum(r["candidate_seconds"] for r in rows)
    geo = math.exp(sum(math.log(r["baseline_seconds"] / r["candidate_seconds"]) for r in rows) / len(rows))
    details = {"problem": problem, "instances_used": len(rows), "excluded": excluded,
               "baseline_total_seconds": round(total_base, 6), "candidate_total_seconds": round(total_cand, 6),
               "timeouts": sum(r["candidate_status"] == "TIMEOUT" for r in rows),
               "per_instance_geometric_mean_speedup": round(geo, 6), "per_instance": rows,
               "time_measure": "CPU seconds: model module top-level code + build_model + CPMpy transformation + export to protobuf text "
                               "(sandboxed child, its CPU clock read from outside) + CP-SAT Solve (1 worker, seed 42) + extractor (child)"}
    return total_base / total_cand, details
