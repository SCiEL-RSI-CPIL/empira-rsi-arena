"""Test runner executed inside the sandbox (self-contained; standard library only).

``run_tests(source, entry, tests, per_test_seconds)`` executes a QuixBugs-style
program and returns one boolean per test. A test is {"input": [args...],
"expected": value, "tolerance": optional float}. Outputs follow the QuixBugs
pytest semantics: generators are consumed into lists, results are normalised
through JSON (tuples become lists) and compared with ``==`` or, when a
tolerance is given, by absolute difference. Exceptions and timeouts fail.

``outputs`` returns the normalised outputs of a program on a list of inputs
(used at preparation time to compute expected values with the reference fix).
``trace_tests`` additionally records the executed line numbers of the program
for each test (statement coverage spectra for fault localisation).
"""
import copy
import json
import signal
import sys


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _load(source, entry):
    namespace = {"__name__": "program_under_test"}
    exec(compile(source, "<program>", "exec"), namespace)
    function = namespace.get(entry)
    if not callable(function):
        raise NameError(f"program does not define {entry}()")
    return function


def _check(function, test):
    output = function(*copy.deepcopy(test["input"]))
    if hasattr(output, "__next__"):
        output = list(output)
    output = json.loads(json.dumps(output))
    expected = test["expected"]
    if test.get("tolerance") is not None:
        return isinstance(output, (int, float)) and not isinstance(output, bool) and abs(output - expected) <= test["tolerance"]
    return output == expected


def _run(function, test, seconds):
    signal.signal(signal.SIGPROF, _alarm)
    signal.setitimer(signal.ITIMER_PROF, seconds)
    try:
        return bool(_check(function, test))
    except _Timeout:
        return False
    except BaseException:  # noqa: BLE001 - any failure of the program under test fails the test
        return False
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)


def run_tests(source, entry, tests, per_test_seconds=1.0, stop_on_failure=False):
    """One boolean per test; with stop_on_failure, tests after the first failure are not run (reported False)."""
    try:
        function = _load(source, entry)
    except BaseException:  # noqa: BLE001 - a program that does not load fails every test
        return [False] * len(tests)
    results = []
    for test in tests:
        results.append(_run(function, test, per_test_seconds))
        if stop_on_failure and not results[-1]:
            return results + [False] * (len(tests) - len(results))
    return results


def outputs(source, entry, inputs, per_test_seconds=1.0):
    """[ok, JSON-normalised output] per input; ok is False where the program raises or times out."""
    function = _load(source, entry)
    results = []
    for args in inputs:
        signal.signal(signal.SIGPROF, _alarm)
        signal.setitimer(signal.ITIMER_PROF, per_test_seconds)
        try:
            output = function(*copy.deepcopy(args))
            if hasattr(output, "__next__"):
                output = list(output)
            results.append([True, json.loads(json.dumps(output))])
        except BaseException:  # noqa: BLE001 - such inputs are not used as tests
            results.append([False, None])
        finally:
            signal.setitimer(signal.ITIMER_PROF, 0)
    return results


def trace_tests(source, entry, tests, per_test_seconds=1.0):
    """[(passed, sorted executed line numbers of <program>)] per test."""
    function = _load(source, entry)
    spectra = []
    for test in tests:
        lines = set()

        def tracer(frame, event, arg):
            if frame.f_code.co_filename == "<program>":
                if event in ("line", "call"):
                    lines.add(frame.f_lineno)
                return tracer
            return None
        sys.settrace(tracer)
        try:
            passed = _run(function, test, per_test_seconds)
        finally:
            sys.settrace(None)
        spectra.append([passed, sorted(lines)])
    return spectra
