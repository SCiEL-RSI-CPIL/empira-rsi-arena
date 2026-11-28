"""Test runner executed inside the sandbox (self-contained; standard library only).

``outputs(source, entry, inputs, step_limits)`` runs a QuixBugs-style program
on a list of inputs and returns its outputs, normalised with the QuixBugs
pytest semantics: generators are consumed into lists and results go through
JSON (tuples become lists). Expected values are never passed in; the caller
compares the outputs outside this process (software_quixbugs.passes), so a
program under test cannot read them.

A run is limited by steps, not by time: the program is compiled with a counter
at every function entry and every loop iteration, and a run that uses up its
step limit is stopped. The count depends only on the program and its input, so
the same run times out on every machine or on none. A CPU-time limit remains
as a safety net for programs that spend long between two steps.

``trace_tests`` records the executed line numbers of the program for each test
(statement coverage spectra for fault localisation). It receives expected
values and is used only at preparation time on the original QuixBugs programs,
never on candidate code.
"""
import ast
import copy
import json
import signal
import sys

STEPS, EXHAUSTED = "__artask_steps__", "__artask_exhausted__"
COUNTER = f"{STEPS}[0] -= 1\nif {STEPS}[0] < 0:\n    raise {EXHAUSTED}()\n"
_ARMED = [False]  # whether the CPU-time signal may interrupt the running program
SECONDS_PER_STEP = 2e-6  # CPU safety net: 10 s plus this much per allowed step, far above what counted steps cost
TRACE_STEPS = 50_000  # steps traced of a run that exceeds its step limit


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    if _ARMED[0]:
        _ARMED[0] = False  # one interruption per run, and none once the run's result is being recorded
        raise _Timeout()


class _Count(ast.NodeTransformer):
    """Puts the step counter at the start of every function body and loop body (after a docstring)."""

    def _counted(self, node):
        self.generic_visit(node)
        counter = ast.parse(COUNTER).body
        for statement in counter:
            for part in ast.walk(statement):
                ast.copy_location(part, node)  # the counter reports the header's line, which runs anyway
        first = node.body[0]
        docstring = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(first, ast.Expr) \
            and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
        node.body[docstring:docstring] = counter
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_For = visit_AsyncFor = visit_While = _counted


def _compile(source):
    """Code of the program with step counters."""
    return compile(ast.fix_missing_locations(_Count().visit(ast.parse(source))), "<program>", "exec")


def _load(code, entry, steps):
    """The entry function of the compiled program; ``steps`` is the one-element budget its counters spend."""
    namespace = {"__name__": "program_under_test", STEPS: steps, EXHAUSTED: _Timeout}
    exec(code, namespace)
    function = namespace.get(entry)
    if not callable(function):
        raise NameError(f"program does not define {entry}()")
    return function


def _line_table(code):
    """{code object: its line numbers} for the functions, lambdas, comprehensions and class bodies of the program."""
    table = {}
    for const in code.co_consts:
        if hasattr(const, "co_lines"):
            table[const] = {line for _, _, line in const.co_lines() if line is not None}
            table.update(_line_table(const))
    return table


def _output(function, args, steps, limit, slack=1.0):
    """[True, JSON-normalised output], or [False, None] when the program raises or exceeds its limits."""
    steps[0] = limit
    signal.signal(signal.SIGPROF, _alarm)
    try:
        _ARMED[0] = True
        signal.setitimer(signal.ITIMER_PROF, slack * (10.0 + SECONDS_PER_STEP * limit))
        try:
            output = function(*copy.deepcopy(args))
            if hasattr(output, "__next__"):
                output = list(output)
            result = [True, json.loads(json.dumps(output, allow_nan=False))]
        except BaseException:  # noqa: BLE001 - any failure of the program under test fails the test
            result = [False, None]
        _ARMED[0] = False
    except _Timeout:  # the signal arrived after the run had ended (a long operation in C code delays it)
        result = [False, None]
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
    return result


def outputs(source, entry, inputs, step_limits, stop_on_failure=False):
    """[ok, JSON-normalised output] per input; ok is False where the program raises or uses up its step limit.

    With ``stop_on_failure`` the inputs after the first such failure are not run (reported as failed)."""
    steps = [max(step_limits, default=0)]  # module-level code of the program gets a budget as well
    function = _load(_compile(source), entry, steps)
    results = []
    for args, limit in zip(inputs, step_limits):
        if stop_on_failure and results and not results[-1][0]:
            results.append([False, None])
        else:
            results.append(_output(function, args, steps, limit))
    return results


def trace_tests(source, entry, tests, step_limits):
    """[(passed, sorted executed line numbers of <program>)] per test.

    The verdict comes from an untraced run. A run that exceeds its step limit is traced for its first TRACE_STEPS
    steps only: tracing every line of millions of loop iterations takes minutes, and such a run repeats its lines."""
    steps = [max(step_limits, default=0)]
    code = _compile(source)
    function, table = _load(code, entry, steps), _line_table(code)
    spectra = []
    for test, limit in zip(tests, step_limits):
        done, output = _output(function, test["input"], steps, limit)
        if done and test.get("tolerance") is not None:
            passed = isinstance(output, (int, float)) and not isinstance(output, bool) and abs(output - test["expected"]) <= test["tolerance"]
        else:
            passed = done and output == test["expected"]
        lines, unseen = set(), {part: set(numbers) for part, numbers in table.items()}

        def tracer(frame, event, arg):
            part = frame.f_code
            if part.co_filename != "<program>":
                return None
            if event in ("line", "call"):
                lines.add(frame.f_lineno)
            rest = unseen.get(part)
            if rest is not None:
                rest.discard(frame.f_lineno)
                if rest:
                    return tracer
                del unseen[part]
            # Every line of this code object is recorded: its frames need no tracing, and when that holds for the
            # whole program nothing more can be learnt, so a long run costs no more than an untraced one.
            frame.f_trace = None
            if not unseen:
                sys.settrace(None)
            return None
        sys.settrace(tracer)
        try:
            _output(function, test["input"], steps, limit if done else min(limit, TRACE_STEPS), slack=10.0)
        finally:
            sys.settrace(None)
        spectra.append([bool(passed), sorted(lines)])
    return spectra
