"""Regression tests for the defects found while porting the or_csp evaluators (12 packages).

Each test reproduces a candidate that scored or tampered before the fix: an int or str subclass
that lies to a checker, a dict that answers differently while checked and while costed, a
candidate whose module-level code ran in the evaluator before the static guard saw it, a CSPLib
model that passed the guard and rewrote the harness through typing.sys.modules, and a C++
program that replaced the polyomino checker, left a process behind or forged its launcher report.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "suite" / "_lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import or_csp_cpsat as H  # noqa: E402
from autoresearch_suite import or_csp_csplib as C  # noqa: E402
from autoresearch_suite import or_csp_egraph as E  # noqa: E402
from autoresearch_suite import or_csp_frontier as F  # noqa: E402
from autoresearch_suite import or_csp_iopddl as I  # noqa: E402
from autoresearch_suite import or_csp_modelrun as M  # noqa: E402
from autoresearch_suite import runtime, sandboxed  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

try:
    COMPILER = F.find_gnu_compiler()
except RuntimeError:
    COMPILER = None
HAVE_FRONTIER = (F.DATA / "chk.cc").is_file() and len(list(F.DATA.glob("*.in"))) == 70
CSPLIB_TASKS = ["pdr-balanced-academic-curriculum-csplib-30", "pdr-balanced-incomplete-block-design-csplib-28",
                "pdr-car-sequencing-csplib-1", "pdr-covering-arrays-csplib-45", "pdr-error-correction-codes-csplib-36",
                "pdr-social-golfer-csplib-10", "pdr-tail-assignment-csplib-115", "pdr-vessel-loading-csplib-8"]


class Liar(int):
    """An int whose comparisons and arithmetic say what a checker wants to hear."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __hash__(self):
        return 0

    def __add__(self, other):
        return self

    __radd__ = __add__

    def __and__(self, other):
        return self

    __rand__ = __and__


def wrapper(task):
    spec = importlib.util.spec_from_file_location(f"evaluator_{task.replace('-', '_')}", ROOT / "suite" / task / "evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TemporaryFiles(unittest.TestCase):
    def file(self, name, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / name
        path.write_text(text)
        return path


class CsplibAnswerTests(unittest.TestCase):
    def test_a_lying_int_subclass_no_longer_passes_the_checker(self):
        data = C.generate_bibd(7, 7, 3, 3, 1)
        answer = H.plain({"matrix": [[Liar(0)] * 7 for _ in range(7)]})  # before the fix: accepted as a Fano plane
        self.assertIs(type(answer["matrix"][0][0]), int)
        with self.assertRaisesRegex(CandidateError, "object 0 is in 0 blocks"):
            C.check_bibd(data, answer)

    def test_plain_keeps_honest_values_and_rejects_others(self):
        class Name(str):
            def __eq__(self, other):
                return True

            __hash__ = str.__hash__
        value = H.plain({Name("matrix"): (1, True, 2.5, None, "x")})
        self.assertEqual(value, {"matrix": [1, True, 2.5, None, "x"]})
        self.assertIs(type(next(iter(value))), str)
        with self.assertRaises(CandidateError):
            H.plain({"matrix": object()})


class CsplibGuardOrderTests(TemporaryFiles):
    """The static guard now sees the source before any statement of it runs in the evaluator."""

    TAMPER = ("import os\nfrom autoresearch_suite import or_csp_csplib, runtime\nruntime.TAMPERED = True\n"
              "or_csp_csplib.score = lambda module, params: (1e9, {})\n\ndef build_model(data):\n    return None, None\n")

    def test_import_time_tampering_gains_nothing(self):
        original = C.score
        self.addCleanup(lambda: (setattr(C, "score", original), runtime.__dict__.pop("TAMPERED", None)))
        path = self.file("candidate.py", self.TAMPER)
        for task in CSPLIB_TASKS:
            with self.subTest(task=task):
                result = wrapper(task).evaluate(str(path))  # before the fix: valid, score 1e9
                self.assertIsNone(result["score"])
                self.assertIn("import of 'os' is not allowed", result["error"])
                self.assertFalse(hasattr(runtime, "TAMPERED"))
                self.assertIs(C.score, original)
                self.assertIn("sandboxed.run(program_path, task", (ROOT / "suite" / task / "evaluator.py").read_text())

    def test_the_guarded_candidate_is_never_imported_by_the_evaluator(self):
        text = "import typing\ntyping.sys.modules['math'].LOADED = True\n\ndef build_model(data):\n    return None, None\n"
        source, name = H.load_candidate(sandboxed.source_program(self.file("candidate.py", text)))
        self.assertEqual((source, name), (text, "candidate.py"))
        self.assertFalse(hasattr(math, "LOADED"))  # before the fix: imported here after the guard
        with self.assertRaises(CandidateError):
            H.load_candidate(sandboxed.source_program(self.file("bad.py", "import os\n")))
        with self.assertRaisesRegex(CandidateError, "must define build_model"):
            H.load_candidate(sandboxed.source_program(self.file("none.py", "import math\n")))


@unittest.skipUnless(importlib.util.find_spec("ortools"), "ortools is not installed")
class ModelTransportTests(unittest.TestCase):
    def test_the_worker_solves_a_model_in_the_encoding_of_this_ortools(self):
        from ortools.sat.python import cp_model
        model = cp_model.CpModel()
        x, y = model.NewIntVar(0, 9, "x"), model.NewIntVar(0, 9, "y")
        model.Add(x + y == 12)
        model.Add(x - y == 2)
        worker = H.SolverWorker()
        self.addCleanup(worker.close)
        reply = worker.solve(str(model.Proto()).encode(), 10.0)  # the text format the sandboxed model process sends
        self.assertEqual(reply["status"], "OPTIMAL")
        self.assertEqual(reply["solution"][:2], [7, 5])
        model.AddHint(x, 7)  # a hint (or an objective) cannot come from a CPMpy satisfaction model: it would carry a stored answer
        self.assertEqual(worker.solve(str(model.Proto()).encode(), 10.0)["status"], "MODEL_REJECTED")
        model.ClearHints()
        model.Maximize(x)
        self.assertEqual(worker.solve(str(model.Proto()).encode(), 10.0), {"status": "MODEL_REJECTED", "reason": "objective"})


HAVE_CPMPY = all(importlib.util.find_spec(m) for m in ("cpmpy", "ortools"))
BIBD_MODEL = """
import cpmpy as cp


def build_model(data):
    v, b, r, k, lam = data["v"], data["b"], data["r"], data["k"], data["lam"]
    incidence = cp.boolvar(shape=(v, b), name="incidence")
    model = cp.Model([cp.sum(incidence[o, :]) == r for o in range(v)] + [cp.sum(incidence[:, j]) == k for j in range(b)])
    for o1 in range(v):
        for o2 in range(o1 + 1, v):
            model += cp.sum(incidence[o1, j] & incidence[o2, j] for j in range(b)) == lam

    def extractor():
        return {"matrix": [[int(x) for x in row] for row in incidence.value().tolist()]}
    return model, extractor
"""
# The demonstrated bypass (scored 1005.8 on the BIBD package): passes the guard, reaches the harness through typing.sys.modules
BYPASS = """
import typing
_harness = typing.sys.modules["autoresearch_suite.or_csp_cpsat"]
_real = _harness.run_model


def _fast(*args, **kwargs):
    record, answer = _real(*args, **kwargs)
    if kwargs.get("is_candidate", True) and record.get("status") == "SOLVED":
        record["runtime"] *= 0.001
    return record, answer


_harness.run_model = _fast
"""
# The same model code made tolerant: fakes the clocks of its own process and burns CPU outside build_model
CLOCKS = """
import typing
_time = typing.sys.modules["time"]
_end = _time.process_time() + 0.3
while _time.process_time() < _end:
    pass
_time.process_time = _time.perf_counter = _time.monotonic = lambda: 0.0
"""
# Sends the answer phase itself, out of order, from inside build_model
FORGE = """
import typing


def build_model(data):
    typing.sys._getframe(2).f_locals["oracle"]({"phase": "answer", "answer": {"matrix": [[1, 1, 0, 1, 0, 0, 0]] * 7}})
    return None
"""


@unittest.skipUnless(HAVE_CPMPY, "cpmpy and ortools are required")
class CsplibModelProcessTests(TemporaryFiles):
    """Model code runs only in a sandboxed child whose CPU time the evaluator reads from outside."""

    PARAMS = {"problem": "bibd", "instances": [{"v": 7, "b": 7, "r": 3, "k": 3, "lam": 1}], "cap_factor": 3.0, "cap_floor": 5.0,
              "baseline_cap": 30.0}

    def score(self, source, **params):
        return C.score(sandboxed.source_program(self.file("candidate.py", source)), dict(self.PARAMS, **params))

    def test_the_typing_sys_modules_bypass_gains_nothing(self):
        original = M.run_model
        with self.assertRaisesRegex(CandidateError, "KeyError: 'autoresearch_suite.or_csp_cpsat'"):
            self.score(BYPASS + BIBD_MODEL)  # before the fix: valid, 1005.8 on the package
        self.assertIs(M.run_model, original)
        tolerant = (BYPASS.replace('["autoresearch_suite.or_csp_cpsat"]', '.get("autoresearch_suite.or_csp_cpsat")')
                    .replace("_real = _harness.run_model", "_real = getattr(_harness, 'run_model', None)")
                    .replace("\n_harness.run_model = _fast", "\nif _harness is not None:\n    _harness.run_model = _fast"))
        value, details = self.score(tolerant + BIBD_MODEL)
        self.assertLess(value, 3.0)
        self.assertEqual(details["timeouts"], 0)

    def test_faked_clocks_and_module_level_work_are_counted(self):
        value, details = self.score(CLOCKS + BIBD_MODEL)
        row = details["per_instance"][0]
        self.assertGreaterEqual(row["candidate_build_cpu"], 0.3)  # in the evaluator's process the module level was untimed
        self.assertGreater(row["candidate_seconds"], row["baseline_seconds"])
        self.assertLess(value, 1.0)

    def test_an_out_of_order_protocol_message_makes_the_candidate_invalid(self):
        with self.assertRaisesRegex(CandidateError, "protocol violation: 'answer' while expecting 'model'"):
            self.score(FORGE)

    def test_a_build_that_never_returns_counts_twice_the_cap(self):
        original = M.CPU_MARGIN
        M.CPU_MARGIN = 1
        self.addCleanup(setattr, M, "CPU_MARGIN", original)
        value, details = self.score("def build_model(data):\n    while True:\n        pass\n", cap_floor=1.0, cap_factor=1e-6)
        row = details["per_instance"][0]
        self.assertEqual((row["candidate_status"], row["cap_seconds"], row["candidate_seconds"]), ("TIMEOUT", 1.0, 2.0))

    def test_the_text_transport_and_the_data_round_trip_are_lossless(self):
        from cpmpy.solvers.ortools import CPM_ortools
        from ortools.sat.python import cp_model
        namespace = {}
        exec(compile(BIBD_MODEL, "<bibd>", "exec"), namespace)  # noqa: S102 - the test's own model
        model, _ = namespace["build_model"](C.generate_bibd(7, 7, 3, 3, 1))
        solver = CPM_ortools(model)
        solver.solver_vars(list(solver.user_vars))
        parsed = cp_model.CpModel()
        if not hasattr(parsed.Proto(), "parse_text_format"):
            self.skipTest("OR-Tools < 9.15: the worker parses the text with google.protobuf.text_format")
        self.assertTrue(parsed.Proto().parse_text_format(str(solver.ort_model.Proto())))
        self.assertEqual(str(parsed.Proto()), str(solver.ort_model.Proto()))
        for task in CSPLIB_TASKS:  # the instance data reaches the child as JSON
            params = wrapper(task).P
            generate = C.PROBLEMS[params["problem"]][0]
            for spec in params["instances"]:
                data = generate(**spec)
                self.assertEqual(json.loads(json.dumps(data)), data, task)


class IopddlIndexTests(unittest.TestCase):
    EXAMPLE = {"nodes": {"intervals": [[30, 70], [40, 70], [50, 120], [110, 140], [110, 150]],
                         "costs": [[15], [55, 65], [25, 45, 35], [85, 75], [95]],
                         "usages": [[10], [25, 25], [15, 20, 15], [10, 10], [15]]},
               "edges": {"nodes": [[0, 1], [0, 2], [1, 3], [2, 4], [3, 4]],
                         "costs": [[30, 40], [50, 10, 40], [90, 10, 20, 80], [60, 20, 30], [70, 60]]},
               "usage_limit": 50}

    def points(self, solve):
        with tempfile.TemporaryDirectory() as tmp:
            with gzip.open(Path(tmp) / "asplos-2025-iopddl-G.json.gz", "wt") as stream:
                json.dump({"problem": self.EXAMPLE}, stream)
            best = I.BEST_TEAM_COST["G"]
            I.BEST_TEAM_COST["G"] = 445
            try:
                return I.contest_points(types.SimpleNamespace(solve=solve), {"benchmarks": ["G"], "time_limit_s": 5.0}, data_dir=tmp)
            finally:
                I.BEST_TEAM_COST["G"] = best

    def test_an_index_that_lies_to_the_range_check_is_rejected(self):
        class IndexLiar(int):
            def __lt__(self, other):
                return False

            def __ge__(self, other):
                return False
        # before the fix: index -3 passed the range check, read node 0's cost and usage, and earned 1.047 points
        with self.assertRaisesRegex(CandidateError, "out of range"):
            self.points(lambda problem, limit: [0, 0, IndexLiar(-3), 1, 0])
        score, details = self.points(lambda problem, limit: [0, 0, 2, 1, 0])
        self.assertEqual((score, details["benchmarks"][0]["cost"]), (1.0, 445))


class EgraphChoiceTests(unittest.TestCase):
    @staticmethod
    def node(eclass, cost, *children):
        return {"op": eclass, "eclass": eclass, "cost": cost, "children": list(children)}

    def test_a_two_faced_dict_is_costed_as_it_was_checked(self):
        toy = {"nodes": {"a0": self.node("a", 1.0, "b0", "c0"), "a1": self.node("a", 5.0, "c0"), "b0": self.node("b", 1.0, "c0"),
                         "b1": self.node("b", 3.0), "c0": self.node("c", 2.0), "c1": self.node("c", 0.5, "a1"),
                         "c1x": self.node("c", 0.001)}, "root_eclasses": ["a"]}

        class TwoFaced(dict):
            calls = 0

            def __getitem__(self, key):
                TwoFaced.calls += 1
                return dict.__getitem__(self, key) if TwoFaced.calls <= 3 else "c1x"

        with tempfile.TemporaryDirectory() as tmp:
            with gzip.open(Path(tmp) / "tensat-toy.json.gz", "wt") as stream:
                json.dump(toy, stream)
            params = {"graphs": ["toy"], "reference_dag_cost": {"toy": 4.0}, "time_limit_s": 5.0}
            score, _ = E.extraction_score(types.SimpleNamespace(extract=lambda g: TwoFaced({"a": "a0", "b": "b0", "c": "c0"})), params,
                                          data_dir=tmp)
        self.assertEqual(score, 1.0)  # before the fix: 0.00025 (the 'c' class costed with a node of another class)


class FrontierSourceTests(TemporaryFiles):
    TAMPER = ("try:\n    from autoresearch_suite import or_csp_frontier, runtime\n    runtime.TAMPERED = True\n"
              "    or_csp_frontier.polyomino_score = lambda module, params, data_dir=None: (100.0, {})\nexcept Exception:\n    pass\n"
              "SOLUTION_CPP = 'int main() { return 0; }'\n")

    def test_the_python_file_runs_only_in_the_sandbox(self):
        original = F.polyomino_score
        self.addCleanup(lambda: (setattr(F, "polyomino_score", original), runtime.__dict__.pop("TAMPERED", None)))
        module = sandboxed.source_program(self.file("candidate.py", self.TAMPER))
        self.assertEqual(F.solution_source(module), "int main() { return 0; }")
        self.assertFalse(hasattr(runtime, "TAMPERED"))
        self.assertIs(F.polyomino_score, original)
        self.assertIn("sandboxed.run(program_path, task",
                      (ROOT / "suite" / "coral-frontier-cs-polyominoes-packing" / "evaluator.py").read_text())

    def test_a_missing_or_computed_source(self):
        with self.assertRaisesRegex(CandidateError, "must define SOLUTION_CPP"):
            F.solution_source(sandboxed.source_program(self.file("a.py", "OTHER = 1\n")))
        module = sandboxed.source_program(self.file("b.py", "SOLUTION_CPP = 'int ' + 'main() { return 0; }'\n"))
        self.assertEqual(F.solution_source(module), "int main() { return 0; }")

    def test_prepared_files_are_compared_with_the_lock(self):
        lock = {entry["path"]: entry for entry in json.loads(F.LOCK.read_text())["files"]}
        if not HAVE_FRONTIER:
            self.skipTest("prepared Frontier-CS data absent: run scripts/prepare_or_csp.py")
        payload = (F.DATA / "1.in").read_bytes()
        F.verify_prepared(F.DATA, {"1.in": payload})
        self.assertEqual(lock["frontier_cs_0/1.in"]["size_bytes"], len(payload))
        with self.assertRaisesRegex(RuntimeError, "does not match the data lock"):
            F.verify_prepared(F.DATA, {"1.in": b"1\n1\n0 0\n"})


@unittest.skipUnless(COMPILER, "a GNU g++ is required")
class FrontierLauncherTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        (self.tmp / "runner.c").write_text(F.RUNNER_C)
        subprocess.run([COMPILER, "-x", "c", "-O2", "-o", str(self.tmp / "runner"), str(self.tmp / "runner.c")], check=True)
        (self.tmp / "in.txt").write_text("1\n")

    def run_program(self, name, source):
        (self.tmp / f"{name}.cc").write_text(source)
        ok, log = F.compile_cpp(COMPILER, self.tmp / f"{name}.cc", self.tmp / name)
        self.assertTrue(ok, log)
        return F.run_case(self.tmp / "runner", self.tmp / name, self.tmp / "in.txt", self.tmp / f"{name}.out", 1.0, 2.0, 256 << 20)

    @unittest.skipUnless(sys.platform.startswith("linux"), "the launcher reaps escaped processes on Linux only")
    def test_no_process_of_the_program_outlives_its_case(self):
        marker = self.tmp / "late.txt"
        source = ('#include <cstdio>\n#include <unistd.h>\nint main(){ if (fork() == 0) { setsid(); if (fork() == 0) { sleep(2); '
                  f'FILE *f = fopen("{marker}", "w"); if (f) {{ fputs("late", f); fclose(f); }} }} _exit(0); }}\n'
                  '  puts("1 1"); return 0; }\n')
        status, _, _ = self.run_program("daemon", source)
        self.assertEqual(status, "ran")
        time.sleep(3.0)  # before the fix the detached grandchild wrote the file (it could have replaced the checker)
        self.assertFalse(marker.exists())

    @unittest.skipUnless(sys.platform.startswith("linux"), "the launcher reaps escaped processes on Linux only")
    def test_cpu_time_of_forked_workers_counts(self):
        source = r'''#include <cstdio>
#include <ctime>
#include <unistd.h>
int main() {
    for (int i = 0; i < 3; ++i)
        if (fork() == 0) { clock_t s = clock(); volatile unsigned long x = 0; while (clock() - s < 0.6 * CLOCKS_PER_SEC) ++x; _exit(0); }
    sleep(1); puts("1 1"); return 0;
}
'''
        status, cpu, _ = self.run_program("workers", source)  # before the fix: 'ran' with 0.001 s for 1.8 s of work
        self.assertEqual(status, "time_limit_exceeded")
        self.assertGreater(cpu, 1.5)

    def test_a_forged_report_line_is_ignored(self):
        source = r'''#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
int main() {
    char path[64], args[4096] = {0};
    snprintf(path, sizeof path, "/proc/%d/cmdline", (int)getppid());
    FILE *f = fopen(path, "rb");
    size_t n = f ? fread(args, 1, sizeof args - 1, f) : 0;
    const char *field = args; int index = 0;
    while (field < args + n && index < 5) { field += strlen(field) + 1; ++index; }
    std::string pipe = std::string("/proc/") + std::to_string(getppid()) + "/fd/" + field;
    int fd = open(pipe.c_str(), O_WRONLY);
    if (fd >= 0) { const char *fake = "\n0 0 0.000000 0.000000 1 0\n"; write(fd, fake, strlen(fake)); }
    puts("1 1"); fflush(stdout);
    for (volatile unsigned long x = 0;; ++x) {}
}
'''
        status, cpu, _ = self.run_program("forger", source)
        self.assertEqual(status, "time_limit_exceeded")
        self.assertGreater(cpu, 0.5)


@unittest.skipUnless(COMPILER and HAVE_FRONTIER, "a GNU g++ and the prepared Frontier-CS data are required")
class FrontierCheckerTests(TemporaryFiles):
    SWAP = r'''#include <bits/stdc++.h>
#include <sys/stat.h>
#include <unistd.h>
int main(int argc, char **argv) {
    std::string self = argv[0], dir = self.substr(0, self.rfind('/'));
    for (const char *name : {"/chk", "/../check/chk"}) {
        std::string fake = dir + "/swap." + std::to_string(getpid());
        if (FILE *f = fopen(fake.c_str(), "w")) {
            fputs("#!/bin/sh\necho 'Ratio: 1.000000000'\n", f);
            fclose(f);
            chmod(fake.c_str(), 0755);
            rename(fake.c_str(), (dir + name).c_str());
        }
    }
    puts("1 1");
    return 0;
}
'''

    def test_a_program_that_replaces_the_checker_scores_nothing(self):
        score, details = F.polyomino_score(types.SimpleNamespace(SOLUTION_CPP=self.SWAP),
                                           {"n_cases": 70, "time_limit_s": 2.0, "memory_mb": 256, "workers": 4})
        self.assertEqual(details["status_counts"], {"rejected": 70})  # before the fix: 70 accepted, score 100
        self.assertEqual(score, 0.0)

    def test_a_tampered_prepared_input_stops_the_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "frontier_cs_0"
            shutil.copytree(F.DATA, copy)
            (copy / "7.in").write_text("1\n1\n0 0\n")
            original = F.DATA
            F.DATA = copy
            try:
                with self.assertRaisesRegex(RuntimeError, "does not match the data lock"):
                    F.polyomino_score(types.SimpleNamespace(SOLUTION_CPP="int main() { return 0; }"),
                                      {"n_cases": 70, "time_limit_s": 2.0, "memory_mb": 256})
            finally:
                F.DATA = original


if __name__ == "__main__":
    unittest.main()
