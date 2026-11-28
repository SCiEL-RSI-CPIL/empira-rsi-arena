"""Code-generation benchmarks: ``solve(question, llm) -> Python source``, judged by executing tests.

Generated code is untrusted. It runs in a separate ``python -I`` process with a
minimal environment of its own (no API keys), a scratch working directory, a
CPU-time limit, a best-effort memory limit and a wall-clock timeout; whatever
it started is killed when the tests end. The child only receives test
*inputs*; it reports each output as JSON and the parent compares with the
expected values, which are never sent to the child. Outputs must be plain data
(numbers, strings, booleans, None, lists, tuples, dicts, sets), so objects
with a forged ``__eq__`` cannot pass. The child is a resource limit, not a
security boundary: it runs under the evaluator's user, so the files that user
can read (the prepared tests among them) and the evaluator's ``/proc`` entries
remain readable to code that looks for them.

The code is executed once per test in a fresh namespace named ``solution``,
as the source evaluators do (CoCoEvo executes code and assert per test, SICA
imports ``solution.py`` in a new process per test): a demonstration under
``if __name__ == "__main__":`` is therefore not run.

* ``leetcode_contest`` (CoCoEvo): every test ``entry_point(args) == expected``
  must hold (pass@1). CoCoEvo runs its asserts in five processes under a 2 s
  total limit; here one process gets ``time_limit`` seconds (default 10).
* ``livecodebench`` (SICA): functional LeetCode problems; a problem scores the
  fraction of its public and private tests that ``solution(*args)`` passes,
  compared as in SICA's test template (``llm_bench_lcb``).
"""
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
import tempfile

from . import llm_bench_core as core
from .runtime import CandidateError

def encode(value, depth=0):
    """Plain-data JSON encoding; tuples and sets are tagged so that == semantics survive the round trip."""
    import numbers
    if depth > 100:
        raise ValueError("nesting too deep")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, list):
        return [encode(v, depth + 1) for v in value]
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v, depth + 1) for v in value]}
    if isinstance(value, (set, frozenset)):
        return {"__set__": sorted((encode(v, depth + 1) for v in value), key=repr)}
    if isinstance(value, dict):
        pairs = [[encode(k, depth + 1), encode(v, depth + 1)] for k, v in value.items()]
        return {"__dict__": sorted(pairs, key=lambda pair: repr(pair[0]))}
    raise TypeError("not plain data: " + type(value).__name__)


ENCODER = inspect.getsource(encode)  # the child process runs the same encoder

HARNESS = ENCODER + r'''
import json, os, signal, sys
sys.setrecursionlimit(50000)
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
payload = json.loads(sys.stdin.read())
code, tests, kind, per_test = payload["code"], payload["tests"], payload["kind"], payload["per_test"]
base = {"__name__": "solution"}
exec(compile(payload.get("prelude") or "", "<prelude>", "exec"), base)
out = os.fdopen(os.dup(1), "w")
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 1)

class Timeout(BaseException):
    pass

def alarm(signum, frame):
    raise Timeout()

signal.signal(signal.SIGALRM, alarm)
try:
    compiled = compile(code, "<candidate>", "exec")
except BaseException:
    out.write("COMPILE_ERROR\n")
    out.flush()
    sys.exit(0)
for index, test in enumerate(tests):
    line = "E %d" % index
    signal.setitimer(signal.ITIMER_REAL, per_test)
    try:
        namespace = dict(base)
        exec(compiled, namespace)
        if kind == "call":
            result = eval(test, namespace)
        else:
            result = namespace["solution"](*[json.loads(arg) for arg in test.strip().splitlines()])
        line = "R %d %s" % (index, json.dumps(encode(result)))
    except BaseException:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    out.write(line + "\n")
    out.flush()
'''
MAX_OUTPUT = 64 << 20


def _limits(cpu_seconds):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            try:
                resource.setrlimit(getattr(resource, name), (4 << 30, 4 << 30))
            except (AttributeError, ValueError, OSError):  # not enforceable on every platform (e.g. macOS)
                pass
        os.setsid()
    return apply


def run_outputs(code, tests, kind, time_limit=10.0, per_test=20.0, prelude=""):
    """Encoded output per test (None for errors, timeouts and missing results) of untrusted ``code``.

    ``prelude`` is trusted source executed once; its names are the starting namespace of every test.
    """
    payload = json.dumps({"code": code, "tests": tests, "kind": kind, "per_test": float(per_test), "prelude": prelude})
    env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "LC_ALL": "C.UTF-8"}
    with tempfile.TemporaryDirectory(prefix="llm-bench-code-") as tmp, tempfile.TemporaryFile(dir=tmp) as sink:
        process = subprocess.Popen([sys.executable, "-I", "-c", HARNESS], stdin=subprocess.PIPE, stdout=sink, stderr=subprocess.DEVNULL,
                                   cwd=tmp, env=env, preexec_fn=_limits(int(time_limit) + 2))
        try:
            process.communicate(payload.encode(), timeout=time_limit)
        except subprocess.TimeoutExpired:
            pass
        finally:  # also after a normal end: a process the generated code started must not outlive the tests
            try:
                os.killpg(process.pid, 9)
            except OSError:
                process.kill()
            process.wait()
        sink.seek(0)
        output = sink.read(MAX_OUTPUT).decode("utf-8", errors="replace")
    results = [None] * len(tests)
    for line in output.splitlines():
        head, _, rest = line.partition(" ")
        index, _, value = rest.partition(" ")
        if head == "R" and index.isdigit() and int(index) < len(tests) and results[int(index)] is None:
            try:
                results[int(index)] = json.loads(value)
            except ValueError:
                pass
    return results


def extract_code(text):
    """The last fenced ```python (or plain ```) block, else the whole reply."""
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", str(text), flags=re.S)
    return blocks[-1] if blocks else str(text)


def _leetcode_contest(P):
    rows = core.read_json("leetcode_contest.json")
    lead = ("Complete the following Python function (a LeetCode weekly or biweekly contest problem whose examples were removed). "
            "Return the complete implementation, including imports, in one ```python code block.\n\n")
    items = [{"id": f"leetcode-{row['id']}", "question": lead + row["prompt"], "tests": row["tests"]} for row in rows]
    limit = float(P.get("time_limit", 10.0))

    def score(item, reply):
        outputs = run_outputs(extract_code(reply), [test["call"] for test in item["tests"]], "call", time_limit=limit, per_test=limit)
        passed = [output is not None and output == test["expected"] for output, test in zip(outputs, item["tests"])]
        return {"score": float(all(passed)), "tests_passed": sum(passed), "tests": len(passed)}
    return items, score


PARITY = '''#!/usr/bin/python3

def parity(b1, b2, b3, b4):
    """Return binary parity of a sequence of input bits. Return 0 for even parity, 1 for odd parity"""
    bit_sum = sum([b1, b2, b3, b4])
    return bit_sum % 2
'''
QUADRATIC = '''#!/usr/bin/python3

def quadratic(a, b, c, x):
    """Return quadratic: a,b,c are coefficients and x is the independent variable."""
    return a*pow(x,2)+b*x+c
'''


def elm_buggy_programs():
    """ELM Appendix A.1: 1-5 bugs in 4-Parity (rename b1..b4 to c1..c4 in the sum, then modulus 3) and 1-2 in Quadratic ('+' to '-' left to right)."""
    programs = []
    for bugs in range(1, 6):
        names = ", ".join(f"c{i}" if i <= bugs else f"b{i}" for i in range(1, 5))
        code = PARITY.replace("sum([b1, b2, b3, b4])", f"sum([{names}])")
        if bugs == 5:
            code = code.replace("bit_sum % 2", "bit_sum % 3")
        programs.append({"id": f"parity-{bugs}", "code": code})
    code = QUADRATIC
    for bugs in range(1, 3):
        head, _, body = code.partition("return ")
        code = head + "return " + body.replace("+", "-", 1)
        programs.append({"id": f"quadratic-{bugs}", "code": code})
    return programs


def elm_tests(name):
    from itertools import product
    if name.startswith("parity"):
        return [{"call": f"parity({a}, {b}, {c}, {d})", "expected": encode((a + b + c + d) % 2)} for a, b, c, d in product((0, 1), repeat=4)]
    values = (-2, 0, 1, 3)
    return [{"call": f"quadratic({a}, {b}, {c}, {x})", "expected": encode(a * x * x + b * x + c)} for a, b, c in product(values, repeat=3) for x in (-2, -1.5, 0, 1, 2.5)]


def _elm_repair(P):
    trials = int(P.get("trials", 4))
    items = []
    for program in elm_buggy_programs():
        for trial in range(trials):
            question = (f"# A buggy implementation (attempt {trial + 1})\n{program['code']}\n# Fixed Bugs\n\n"
                        "Return the complete fixed implementation in one ```python block.")
            items.append({"id": f"{program['id']}-trial-{trial}", "question": question, "tests": elm_tests(program["id"])})

    def score(item, reply):
        outputs = run_outputs(extract_code(reply), [t["call"] for t in item["tests"]], "call", time_limit=10.0, per_test=5.0)
        return float(all(output is not None and output == t["expected"] for output, t in zip(outputs, item["tests"])))
    return items, score


def evaluate(solve, P):
    core.require_signature(solve, 2, "solve(question, llm)")
    if P.get("benchmark") == "leetcode_contest":
        items, scorer = _leetcode_contest(P)
    elif P.get("benchmark") == "elm_repair":
        items, scorer = _elm_repair(P)
    elif P.get("benchmark") == "livecodebench":
        from . import llm_bench_lcb
        items, scorer = llm_bench_lcb.items(P)
    else:
        raise CandidateError(f"unknown code benchmark: {P.get('benchmark')}")
    sample = core.fixed_sample(items, core.sample_size(P), P.get("seed", 0))
    client = core.task_llm(P)
    out = core.run_items(sample, lambda item: core.checked_answer(solve(item["question"], client), "program text"), scorer, client)
    out["details"]["benchmark"] = P["benchmark"]
    return out
