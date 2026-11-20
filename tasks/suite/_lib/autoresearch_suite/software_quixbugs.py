"""QuixBugs program repair and fault-localisation data (Lin et al., 2017; MIT licence).

The 31 Python programs of QuixBugs that come with JSON test cases are used
(the nine graph programs, whose tests are pytest functions building Node
objects, are not). Each program has exactly one buggy statement; the reference
fixes are used only at preparation time (to locate the faulty line for fault
localisation and to check that every kept test passes on the fixed program)
and are never stored in the prepared files.

Two test cases are dropped because the fixed programs need minutes for them
(QuixBugs' own pytest suite skips them as well): the largest knapsack instance
and one levenshtein pair. Outputs are compared with the QuixBugs pytest
semantics plus JSON normalisation, which the fixed hanoi needs (it returns
tuples while the JSON expectations are lists).

Because QuixBugs has only a handful of tests per program, preparation adds up
to 40 generated held-out tests per program (software_quixbugs_inputs): seeded
inputs that satisfy the documented preconditions, with expected outputs
computed by the reference fix. A patch counts as a repair only when it passes
the visible, the remaining QuixBugs and the generated tests.
"""
from __future__ import annotations

import ast
import difflib
import io
import json
import zipfile

from . import software_quixbugs_inputs, software_sandbox

SOURCE = "quixbugs"
SOURCES = (SOURCE,)
PREFIX = "QuixBugs-4257f44b0ff1181dedaedee6a447e133219fcebf/"
SLOW_TESTS = {  # (program, input) pairs skipped by QuixBugs' pytest suite for run time
    "knapsack": lambda args: args[0] == 6404180,
    "levenshtein": lambda args: args == ["amanaplanacanalpanama", "docnoteidissentafastneverpreventsafatnessidietoncod"],
}
TOLERANCE = {"sqrt": lambda args: args[-1]}
HARNESS = "software_harness_quixbugs.py"


def statements(source):
    """[(line, normalised text)] of statements and compound headers, without docstrings."""
    out = []

    def visit(body):
        for node in body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if isinstance(node, (ast.If, ast.While)):
                out.append((node.lineno, type(node).__name__.lower() + " " + ast.unparse(node.test) + ":"))
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                out.append((node.lineno, "for " + ast.unparse(node.target) + " in " + ast.unparse(node.iter) + ":"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.With, ast.AsyncWith, ast.Try)):
                head = ast.unparse(node).splitlines()
                out.append((node.lineno, next((line for line in head if not line.startswith("@")), head[0])))
            else:
                out.append((node.lineno, ast.unparse(node)))
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(node, field, None) if not isinstance(node, (ast.Expr, ast.Assign)) else None
                if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                    if field == "orelse":
                        out.append((sub[0].lineno, "else:"))
                    visit(sub)
            for handler in getattr(node, "handlers", []):
                out.append((handler.lineno, "except " + (ast.unparse(handler.type) if handler.type else "") + ":"))
                visit(handler.body)
    visit(ast.parse(source).body)
    return out


def statement_edits(before, after):
    """Number of statement-level edits (replaced, inserted or deleted statements) between two programs."""
    a, b = [t for _, t in statements(before)], [t for _, t in statements(after)]
    edits = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            edits += max(i2 - i1, j2 - j1)
    return edits


def fault_lines(buggy, fixed):
    """Buggy-program lines of the changed statements; for an omitted statement, its neighbours."""
    a, b = statements(buggy), statements(fixed)
    lines = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, [t for _, t in a], [t for _, t in b], autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            lines += [line for line, _ in a[i1:i2]]
        elif tag == "insert":
            lines += [a[i][0] for i in (i1 - 1, i1) if 0 <= i < len(a)]
    return sorted(set(lines))


def run_tests(source, entry, tests, *, cpu_seconds=10, stop_on_failure=False):
    """Run a program's tests in the sandbox; a crash or time-out of the child fails all of them."""
    if not tests:
        return []
    ok, value = software_sandbox.run_harness(software_sandbox.harness_source(HARNESS), "run_tests",
                                             [source, entry, tests, 1.0, stop_on_failure], cpu_seconds=cpu_seconds, memory_mb=1024)
    if not ok or not isinstance(value, list) or len(value) != len(tests):
        return [False] * len(tests)
    return [bool(v) for v in value]


def _programs(archive):
    names = sorted(n[len(PREFIX) + len("json_testcases/"):-5] for n in archive.namelist()
                   if n.startswith(PREFIX + "json_testcases/") and n.endswith(".json"))
    for name in names:
        tests = []
        for line in archive.read(f"{PREFIX}json_testcases/{name}.json").decode().splitlines():
            if not line.strip():
                continue
            args, expected = json.loads(line)
            args = args if isinstance(args, list) else [args]
            if name in SLOW_TESTS and SLOW_TESTS[name](args):
                continue
            test = {"input": args, "expected": expected}
            if name in TOLERANCE:
                test["tolerance"] = TOLERANCE[name](args)
            tests.append(test)
        yield (name, archive.read(f"{PREFIX}python_programs/{name}.py").decode(),
               archive.read(f"{PREFIX}correct_python_programs/{name}.py").decode(), tests)


def generated_tests(fixed, name):
    """Held-out tests from seeded precondition-respecting inputs, expected values from the reference fix."""
    inputs = software_quixbugs_inputs.generate(name)
    ok, results = software_sandbox.run_harness(software_sandbox.harness_source(HARNESS), "outputs", [fixed, name, inputs, 1.0],
                                               cpu_seconds=120, memory_mb=1024)
    if not ok:
        raise RuntimeError(f"{name}: reference outputs failed: {results}")
    tests = []
    for args, (done, value) in zip(inputs, results):
        if done:
            test = {"input": args, "expected": value}
            if name in TOLERANCE:
                test["tolerance"] = TOLERANCE[name](args)
            tests.append(test)
    if len(tests) < 10 or not all(run_tests(fixed, name, tests, cpu_seconds=60)):
        raise RuntimeError(f"{name}: generated tests are too few or fail on the reference fix")
    return tests


def split(outcomes):
    """Visible test indices: even positions, plus the first failing test when none of those fails."""
    visible = [i for i in range(len(outcomes)) if i % 2 == 0]
    if all(outcomes[i] for i in visible):
        visible.append(next(i for i, ok in enumerate(outcomes) if not ok))
    return sorted(visible)


def derive(raw):
    repair, spectra = [], []
    harness = software_sandbox.harness_source(HARNESS)
    with zipfile.ZipFile(io.BytesIO(raw[SOURCE])) as archive:
        for name, buggy, fixed, tests in _programs(archive):
            if not all(run_tests(fixed, name, tests)):
                raise RuntimeError(f"{name}: a kept test fails on the fixed program")
            outcomes = run_tests(buggy, name, tests)
            if all(outcomes):
                raise RuntimeError(f"{name}: no test fails on the buggy program")
            visible = split(outcomes)
            generated = generated_tests(fixed, name)
            repair.append({"name": name, "entry": name, "source": buggy, "visible": [tests[i] for i in visible],
                           "held_out": [t for i, t in enumerate(tests) if i not in visible], "generated": generated,
                           "failing_tests": sum(not ok for ok in outcomes),
                           "failing_generated_tests": sum(not ok for ok in run_tests(buggy, name, generated, cpu_seconds=60))})
            ok, traced = software_sandbox.run_harness(harness, "trace_tests", [buggy, name, tests, 1.0], cpu_seconds=60, memory_mb=1024)
            if not ok:
                raise RuntimeError(f"{name}: tracing failed: {traced}")
            executable = sorted({line for line, _ in statements(buggy)})
            spectra.append({"name": name, "lines": executable, "tests": [[bool(p), [l for l in lines if l in executable]] for p, lines in traced],
                            "faulty_lines": fault_lines(buggy, fixed)})
    dump = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"quixbugs/repair.json": dump(repair), "quixbugs/sbfl.json": dump(spectra)}


MAX_PATCH_CHARS = 20000
MAX_EDITS = 2


def public_task(program):
    """What the repair procedure sees: the buggy program and its visible tests (never the held-out tests)."""
    return {"name": program["name"], "entry": program["entry"], "source": program["source"],
            "tests": [dict(t) for t in program["visible"]], "max_statement_edits": MAX_EDITS}


def judge_patch(program, patch):
    """Status of one returned patch: rejected, unrepaired, plausible (visible tests only) or repaired."""
    try:
        edits = statement_edits(program["source"], patch)
    except (SyntaxError, ValueError, RecursionError) as error:
        return {"status": "rejected", "reason": f"does not parse: {type(error).__name__}"}
    if edits > MAX_EDITS:
        return {"status": "rejected", "reason": f"{edits} statement edits exceed {MAX_EDITS}", "edits": edits}
    run = lambda tests: all(run_tests(patch, program["entry"], tests, cpu_seconds=20, stop_on_failure=True))
    if not run(program["visible"]):
        return {"status": "unrepaired", "edits": edits, "failed": "visible tests"}
    quixbugs = run(program["held_out"])
    generated = quixbugs and run(program["generated"])
    return {"status": "repaired" if generated else "plausible", "edits": edits, "passes_all_quixbugs_tests": quixbugs,
            "failed": None if generated else ("generated held-out tests" if quixbugs else "QuixBugs held-out tests")}


def repair_score(module, *, cpu_seconds=4):
    """Apply the candidate's repair(task) to every program in its own sandbox; score = fraction repaired."""
    from . import software_data
    from .runtime import CandidateError
    program_path = software_sandbox.require_entry(module, "repair")
    programs = software_data.load_json("quixbugs/repair.json")
    results = {}
    for program in programs:
        patch = software_sandbox.run_candidate(program_path, "repair", [public_task(program)], cpu_seconds=cpu_seconds, memory_mb=1024)
        if not isinstance(patch, str):
            raise CandidateError(f"repair() must return the patched program as a str ({program['name']})")
        if len(patch) > MAX_PATCH_CHARS:
            raise CandidateError(f"patch for {program['name']} exceeds {MAX_PATCH_CHARS} characters")
        results[program["name"]] = judge_patch(program, patch) if patch != program["source"] else {"status": "unrepaired", "edits": 0}
    counts = {s: sum(r["status"] == s for r in results.values()) for s in ("repaired", "plausible", "unrepaired", "rejected")}
    return counts["repaired"] / len(programs), {"programs": len(programs), **counts,
                                               "repaired_programs": sorted(n for n, r in results.items() if r["status"] == "repaired"),
                                               "overfitting_programs": sorted(n for n, r in results.items() if r["status"] == "plausible"),
                                               "per_program": results}
