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

A program run is limited by steps (function entries plus loop iterations,
counted by software_harness_quixbugs), not by time, so neither the prepared
files nor the verdict on a patch depend on the speed of the machine.
"""
from __future__ import annotations

import ast
import difflib
import io
import json
import zipfile

from . import software_data, software_quixbugs_inputs, software_sandbox

SOURCE = "quixbugs"
SOURCES = (SOURCE,)
PREFIX = "QuixBugs-4257f44b0ff1181dedaedee6a447e133219fcebf/"
SLOW_TESTS = {  # (program, input) pairs skipped by QuixBugs' pytest suite for run time
    "knapsack": lambda args: args[0] == 6404180,
    "levenshtein": lambda args: args == ["amanaplanacanalpanama", "docnoteidissentafastneverpreventsafatnessidietoncod"],
}
TOLERANCE = {"sqrt": lambda args: args[-1]}
HARNESS = "software_harness_quixbugs.py"
RESUMABLE = True  # derive() takes a software_data.Steps and can be completed over several runs
# Steps allowed per test. The reference fixes need at most 761,717 on the kept QuixBugs tests and fewer than 20,000 on
# the generated ones, with one exception: the recursive levenshtein makes 10,308,990 calls on its longest kept pair.
STEP_LIMIT = 2_000_000
HEAVY_TESTS = {"levenshtein": (["rosettacode", "raisethysword"], 25_000_000)}


def step_limit(name, args):
    """Step limit of one test of the program ``name``."""
    heavy = HEAVY_TESTS.get(name)
    return heavy[1] if heavy and args == heavy[0] else STEP_LIMIT


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
            elif isinstance(node, ast.Match):
                out.append((node.lineno, "match " + ast.unparse(node.subject) + ":"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.With, ast.AsyncWith, ast.Try, ast.TryStar)):
                # A decorator replaces the function it wraps, so each one is a statement of its own.
                out.extend((decorator.lineno, "@" + ast.unparse(decorator)) for decorator in getattr(node, "decorator_list", []))
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
            for case in getattr(node, "cases", []):  # a case block holds statements of its own, like a handler
                guard = " if " + ast.unparse(case.guard) if case.guard else ""
                out.append((case.pattern.lineno, "case " + ast.unparse(case.pattern) + guard + ":"))
                visit(case.body)
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


def passes(result, test):
    """Whether one [ok, JSON-normalised output] pair from the sandbox meets a test (QuixBugs pytest semantics)."""
    if not (isinstance(result, list) and len(result) == 2 and result[0] is True):
        return False
    output, expected = result[1], test["expected"]
    if test.get("tolerance") is not None:
        return isinstance(output, (int, float)) and not isinstance(output, bool) and abs(output - expected) <= test["tolerance"]
    return output == expected


def run_tests(source, entry, tests, *, cpu_seconds=120, stop_on_failure=False):
    """Run a program on the tests' inputs in the sandbox and judge the outputs here.

    The child receives only the inputs: the expected values never enter the process that runs the program, so a
    patch cannot read them (for example from the judge's stack frame). Each test has a step limit (step_limit); the
    CPU limit of the child is a safety net far above what the allowed steps cost. A crash or time-out of the child
    fails all tests. With ``stop_on_failure`` the tests after the first run that raises or exceeds its step limit
    are not run and count as failed."""
    if not tests:
        return []
    inputs = [t["input"] for t in tests]
    ok, value = software_sandbox.run_harness(software_sandbox.harness_source(HARNESS), "outputs",
                                             [source, entry, inputs, [step_limit(entry, i) for i in inputs], stop_on_failure],
                                             cpu_seconds=cpu_seconds, memory_mb=1024)
    if not ok or not isinstance(value, list) or len(value) != len(tests):
        return [False] * len(tests)
    return [passes(result, test) for result, test in zip(value, tests)]


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
    ok, results = software_sandbox.run_harness(software_sandbox.harness_source(HARNESS), "outputs",
                                               [fixed, name, inputs, [step_limit(name, i) for i in inputs]],
                                               cpu_seconds=300, memory_mb=1024)
    if not ok:
        raise RuntimeError(f"{name}: reference outputs failed: {results}")
    tests = []
    for args, (done, value) in zip(inputs, results):
        if done:
            test = {"input": args, "expected": value}
            if name in TOLERANCE:
                test["tolerance"] = TOLERANCE[name](args)
            tests.append(test)
    if len(tests) < 10 or not all(run_tests(fixed, name, tests)):
        raise RuntimeError(f"{name}: generated tests are too few or fail on the reference fix")
    return tests


def split(outcomes):
    """Visible test indices: even positions, plus the first failing test when none of those fails."""
    visible = [i for i in range(len(outcomes)) if i % 2 == 0]
    if all(outcomes[i] for i in visible):
        visible.append(next(i for i, ok in enumerate(outcomes) if not ok))
    return sorted(visible)


def derive_program(name, buggy, fixed, tests):
    """[repair record, spectrum record] of one program."""
    if not all(run_tests(fixed, name, tests)):
        raise RuntimeError(f"{name}: a kept test fails on the fixed program")
    outcomes = run_tests(buggy, name, tests)
    if all(outcomes):
        raise RuntimeError(f"{name}: no test fails on the buggy program")
    visible = split(outcomes)
    generated = generated_tests(fixed, name)
    repair = {"name": name, "entry": name, "source": buggy, "visible": [tests[i] for i in visible],
              "held_out": [t for i, t in enumerate(tests) if i not in visible], "generated": generated,
              "failing_tests": sum(not ok for ok in outcomes),
              "failing_generated_tests": sum(not ok for ok in run_tests(buggy, name, generated, cpu_seconds=300))}
    limits = [step_limit(name, t["input"]) for t in tests]
    ok, traced = software_sandbox.run_harness(software_sandbox.harness_source(HARNESS), "trace_tests", [buggy, name, tests, limits],
                                              cpu_seconds=600, memory_mb=1024)
    if not ok:
        raise RuntimeError(f"{name}: tracing failed: {traced}")
    executable = sorted({line for line, _ in statements(buggy)})
    spectrum = {"name": name, "lines": executable, "tests": [[bool(p), [l for l in lines if l in executable]] for p, lines in traced],
                "faulty_lines": fault_lines(buggy, fixed)}
    return [repair, spectrum]


def derive(raw, steps=None):
    """Prepared files from the pinned archive; ``steps`` (software_data.Steps) keeps each finished program."""
    steps = steps or software_data.Steps()
    with zipfile.ZipFile(io.BytesIO(raw[SOURCE])) as archive:
        records = [steps(name, lambda: derive_program(name, buggy, fixed, tests)) for name, buggy, fixed, tests in _programs(archive)]
    dump = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"quixbugs/repair.json": dump([repair for repair, _ in records]), "quixbugs/sbfl.json": dump([spectrum for _, spectrum in records])}


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
    run = lambda tests: all(run_tests(patch, program["entry"], tests, stop_on_failure=True))
    if not run(program["visible"]):
        return {"status": "unrepaired", "edits": edits, "failed": "visible tests"}
    quixbugs = run(program["held_out"])
    generated = quixbugs and run(program["generated"])
    return {"status": "repaired" if generated else "plausible", "edits": edits, "passes_all_quixbugs_tests": quixbugs,
            "failed": None if generated else ("generated held-out tests" if quixbugs else "QuixBugs held-out tests")}


def repair_score(module, *, cpu_seconds=4):
    """Apply the candidate's repair(task) to every program in its own sandbox; score = fraction repaired."""
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
