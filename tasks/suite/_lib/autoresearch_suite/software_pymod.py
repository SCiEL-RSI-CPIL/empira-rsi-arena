"""Coverage-guided fuzzing and mutation-based test generation on pure-Python standard-library modules.

Subjects are taken from the CPython 3.12.14 source release (PSF licence) and
loaded in the sandbox from source under a private package name
(software_harness_pymod). Fuzzing tasks score the number of distinct block
probes (function entries, both outcomes of every if, loop bodies and exits,
handlers, with bodies, conditional-expression branches) that a bounded corpus
of input strings reaches in the instrumented parser files. Mutation tasks
score the fraction of first-order mutants (software_mutate) of a module that a
bounded list of test calls kills: a mutant is killed when some test's outcome
(hash of the repr of the result, or the exception type) differs from the
original module's outcome; tests on which the original times out are ignored.
"""
from __future__ import annotations

import io
import json
import math
import re
import tarfile

from . import software_data, software_mutate, software_sandbox
from .runtime import CandidateError

SOURCE = "cpython-3.12.14"
SOURCES = (SOURCE,)
ROOT = "Python-3.12.14/Lib/"
HARNESS = "software_harness_pymod.py"
MAX_TEXT = 500
MAX_INT = 10 ** 4

FUZZ = {
    "tomllib": {"files": ["tomllib/__init__.py", "tomllib/_parser.py", "tomllib/_re.py", "tomllib/_types.py"],
                "instrumented": ["tomllib/_parser.py", "tomllib/_re.py"], "entry": "tomllib.loads(text)"},
    "re": {"files": ["re/__init__.py", "re/_casefix.py", "re/_compiler.py", "re/_constants.py", "re/_parser.py"],
           "instrumented": ["re/_compiler.py", "re/_parser.py"], "entry": "re._compiler.compile(text, 0)"},
    "html.parser": {"files": ["html/parser.py", "_markupbase.py"], "instrumented": ["html/parser.py", "_markupbase.py"],
                    "patches": {"html/parser.py": [("\nimport _markupbase\n", "\nfrom artask_subject import _markupbase\n")]},
                    "entry": "for flag in (True, False): p = html.parser.HTMLParser(convert_charrefs=flag); p.feed(text); p.close()"},
    "plistlib": {"files": ["plistlib.py"], "instrumented": ["plistlib.py"],
                 "entry": "plistlib.loads(text.encode('latin-1')), i.e. the format (XML or binary) is detected from the bytes"},
}

TEXTWRAP_OPTIONS = {"expand_tabs": "bool", "replace_whitespace": "bool", "fix_sentence_endings": "bool",
                    "break_long_words": "bool", "drop_whitespace": "bool", "break_on_hyphens": "bool", "tabsize": "int",
                    "max_lines": "opt_int", "placeholder": "str", "initial_indent": "str", "subsequent_indent": "str"}
QUERY = ["str", "bool", "bool", "opt_int", "str"]
MUTATION = {
    "textwrap": {"files": ["textwrap.py"], "target": "textwrap.py", "api": {
        "wrap": ["str", "int", ("options", TEXTWRAP_OPTIONS)], "fill": ["str", "int", ("options", TEXTWRAP_OPTIONS)],
        "shorten": ["str", "int", ("options", {k: v for k, v in TEXTWRAP_OPTIONS.items() if k != "max_lines"})],
        "dedent": ["str"], "indent": ["str", "str"]}},
    "urllib.parse": {"files": ["urllib/parse.py"], "target": "urllib/parse.py", "api": {
        "urlparse": ["str", "str", "bool"], "urlsplit": ["str", "str", "bool"], "urlunparse": [("strings", 6)],
        "urlunsplit": [("strings", 5)], "urljoin": ["str", "str", "bool"], "urldefrag": ["str"],
        "parse_qs": QUERY, "parse_qsl": QUERY, "quote": ["str", "str"], "quote_plus": ["str", "str"],
        "unquote": ["str"], "unquote_plus": ["str"], "urlencode": ["pairs", "bool", "str"], "unwrap": ["str"]}},
    "fractions": {"files": ["fractions.py"], "target": "fractions.py", "api": {
        "parse": ["fraction"], "make": ["int", "int"], "from_float": ["float"],
        "binary": ["number", ("choice", ["+", "-", "*", "/", "//", "%", "**", "<", "<=", "==", "divmod", "max"]), "number"],
        "limit_denominator": ["fraction", "int"], "round": ["fraction", "opt_int"], "format": ["fraction", "spec"],
        "method": ["fraction", ("choice", ["as_integer_ratio", "is_integer", "__floor__", "__ceil__", "__trunc__", "__abs__",
                                     "__neg__", "__pos__", "__float__", "__int__", "__bool__", "__hash__", "__str__"])]}},
    "statistics": {"files": ["statistics.py"], "target": "statistics.py",
                   "patches": {"statistics.py": [("try:\n    from _statistics import _normal_dist_inv_cdf\nexcept ImportError:\n    pass\n",
                                                  "# the C accelerator _statistics is not used, so the Python _normal_dist_inv_cdf runs\n")]},
                   "api": {
        "summary": [("choice", ["mean", "fmean", "geometric_mean", "harmonic_mean", "median", "median_low", "median_high", "mode",
                                "multimode", "pvariance", "pstdev", "variance", "stdev"]), "numbers"],
        "median_grouped": ["numbers", "number"], "quantiles": ["numbers", "int", ("choice", ["exclusive", "inclusive"])],
        "weighted": [("choice", ["fmean", "harmonic_mean"]), "numbers", "numbers"],
        "spread_given_center": [("choice", ["variance", "pvariance", "stdev", "pstdev"]), "numbers", "number"],
        "bivariate": [("choice", ["covariance", "correlation", "correlation_ranked", "linear_regression",
                                  "linear_regression_proportional"]), "numbers", "numbers"],
        "normal": ["number", "number", ("choice", ["pdf", "cdf", "inv_cdf", "zscore", "quantiles", "overlap"]), "number"],
        "from_samples": ["numbers"]}},
}


def _patched(archive, files, patches):
    out = {}
    for path in files:
        text = archive.extractfile(ROOT + path).read().decode("utf-8")
        for old, new in patches.get(path, ()):
            if text.count(old) != 1:
                raise RuntimeError(f"{path}: patch anchor {old!r} not found exactly once")
            text = text.replace(old, new)
        out[path] = text
    return out


def derive(raw):
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw[SOURCE]), mode="r:xz") as archive:
        for name, spec in FUZZ.items():
            data = {"subject": name, "files": _patched(archive, spec["files"], spec.get("patches", {})),
                    "instrumented": spec["instrumented"], "entry": spec["entry"]}
            files[f"pymod/fuzz-{name}.json"] = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        for name, spec in MUTATION.items():
            sources = _patched(archive, spec["files"], spec.get("patches", {}))
            data = {"subject": name, "files": sources, "target": spec["target"],
                    "mutants": software_mutate.mutants(sources[spec["target"]])}
            files[f"pymod/mutation-{name}.json"] = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return files


def _fraction_text(value):
    """A short string for Fraction(): at most 30 characters and an exponent of at most two digits."""
    return isinstance(value, str) and len(value) <= 30 and not re.search(r"[eE]\s*[-+]?\s*_?[0-9_]{3}", value)


def _check_value(kind, value, where):
    if isinstance(kind, tuple) and kind[0] == "options":
        if not isinstance(value, dict) or not set(value) <= set(kind[1]):
            raise CandidateError(f"{where}: options must be a dict with keys from {sorted(kind[1])}")
        for key, sub_value in value.items():
            _check_value(kind[1][key], sub_value, f"{where}.{key}")
        return
    if isinstance(kind, tuple) and kind[0] == "strings":
        if not isinstance(value, list) or len(value) != kind[1]:
            raise CandidateError(f"{where}: expected a list of {kind[1]} strings")
        for item in value:
            _check_value("str", item, where)
        return
    if isinstance(kind, tuple) and kind[0] == "choice":
        if value not in kind[1]:
            raise CandidateError(f"{where}: expected one of {kind[1]}")
        return
    ok = {"str": lambda v: isinstance(v, str) and len(v) <= MAX_TEXT,
          "int": lambda v: type(v) is int and abs(v) <= MAX_INT,
          "opt_int": lambda v: v is None or (type(v) is int and abs(v) <= MAX_INT),
          "bool": lambda v: type(v) is bool,
          "float": lambda v: type(v) is float and math.isfinite(v) and abs(v) <= 1e6,
          "fraction": _fraction_text,
          "spec": lambda v: isinstance(v, str) and len(v) <= 12 and not re.search(r"[0-9]{4}", v),
          "number": lambda v: _fraction_text(v) or (type(v) is int and abs(v) <= 1000)
                              or (type(v) is float and math.isfinite(v) and abs(v) <= 1e6),
          "numbers": lambda v: isinstance(v, list) and len(v) <= 40 and all(
              _fraction_text(x) or (type(x) is int and abs(x) <= MAX_INT) or (type(x) is float and math.isfinite(x) and abs(x) <= 1e6)
              for x in v),
          "pairs": lambda v: isinstance(v, list) and len(v) <= 20 and all(
              isinstance(p, list) and len(p) == 2 and all(isinstance(s, str) and len(s) <= 100 for s in p) for p in v)}[kind]
    if not ok(value):
        raise CandidateError(f"{where}: invalid {kind} value")


def check_tests(subject, tests, max_tests):
    api = MUTATION[subject]["api"]
    if not isinstance(tests, list) or not tests or len(tests) > max_tests:
        raise CandidateError(f"generate_tests() must return a list of 1 to {max_tests} tests")
    for i, test in enumerate(tests):
        if not isinstance(test, dict) or set(test) != {"call", "args"} or test["call"] not in api:
            raise CandidateError(f"test {i} must be {{'call': one of {sorted(api)}, 'args': [...]}}")
        if not isinstance(test["args"], list) or len(test["args"]) != len(api[test["call"]]):
            raise CandidateError(f"test {i}: {test['call']} takes {len(api[test['call']])} arguments")
        for j, (kind, value) in enumerate(zip(api[test["call"]], test["args"])):
            _check_value(kind, value, f"test {i} argument {j}")
    return tests


def describe(kind):
    """JSON description of an argument kind for the candidate."""
    if isinstance(kind, tuple):
        return {kind[0]: kind[1]}
    return kind


def _harness(entry, args, cpu_seconds):
    ok, value = software_sandbox.run_harness(software_sandbox.harness_source(HARNESS), entry, args,
                                             cpu_seconds=cpu_seconds, memory_mb=2048)
    if not ok:
        raise CandidateError(f"replaying the candidate's inputs failed ({value.kind}): {value}"[:2000])
    return value


def fuzz_score(module, subject, *, cpu_seconds=60, max_inputs=1000, max_chars=MAX_TEXT * 4):
    path = software_sandbox.require_entry(module, "fuzz")
    data = software_data.load_json(f"pymod/fuzz-{subject}.json")
    target = dict(data, max_inputs=max_inputs, max_chars=max_chars)
    inputs = software_sandbox.run_candidate(path, "fuzz", [target], cpu_seconds=cpu_seconds, memory_mb=2048)
    if not isinstance(inputs, list) or len(inputs) > max_inputs or not all(isinstance(s, str) and len(s) <= max_chars for s in inputs):
        raise CandidateError(f"fuzz() must return a list of at most {max_inputs} strings of at most {max_chars} characters")
    result = _harness("coverage", [data["files"], data["instrumented"], subject, inputs], 120)
    return float(len(result["hits"])), {"subject": subject, "inputs": len(inputs), "probes": result["probes"],
                                        "timed_out_inputs": result["timeouts"]}


def mutation_score(module, subject, *, cpu_seconds=60, max_tests=30):
    path = software_sandbox.require_entry(module, "generate_tests")
    data = software_data.load_json(f"pymod/mutation-{subject}.json")
    mutants = [[m["start"], m["end"], m["replacement"]] for m in data["mutants"]]
    request = {"subject": subject, "files": data["files"], "target": data["target"], "max_tests": max_tests,
               "api": {k: [describe(a) for a in v] for k, v in MUTATION[subject]["api"].items()},
               "mutants": [dict(m) for m in data["mutants"]]}
    tests = check_tests(subject, software_sandbox.run_candidate(path, "generate_tests", [request], cpu_seconds=cpu_seconds,
                                                                memory_mb=2048), max_tests)
    result = _harness("mutation_analysis", [data["files"], data["target"], subject, tests, mutants], 240)
    killed = [k >= 0 for k in result["killed"]]
    by_operator = {}
    for m, k in zip(data["mutants"], killed):
        hit, total = by_operator.get(m["operator"], (0, 0))
        by_operator[m["operator"]] = (hit + k, total + 1)
    return sum(killed) / len(killed), {"subject": subject, "mutants": len(killed), "killed": sum(killed),
                                       "tests": len(tests), "usable_tests": result["usable_tests"],
                                       "killed_by_operator": {k: f"{h}/{t}" for k, (h, t) in sorted(by_operator.items())}}
