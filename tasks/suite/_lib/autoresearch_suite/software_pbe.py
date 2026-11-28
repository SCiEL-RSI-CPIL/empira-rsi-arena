"""Programming by example for string transformations (SyGuS-Comp PBE strings, FlashFill-style tasks).

Each task is a SyGuS problem from the PBE_SLIA track (lib/PBE_SLIA_Track/
from_2018) that also exists in a "-long" variant with many more examples of
the same function: phone numbers, names, initials, university addresses. The
candidate synthesises a program in the task's grammar from the base file's
few examples; it is correct when it reproduces those and the long variant's
other examples (held out). Programs are S-expressions in SMT-LIB string
syntax, checked against the grammar and interpreted here with the SMT-LIB 2.6
string semantics (validated against Z3 in the tests).
"""
from __future__ import annotations

import io
import json
import re
import zipfile

from . import software_data, software_sandbox
from .runtime import CandidateError

SOURCE = "sygus-benchmarks"
SOURCES = (SOURCE,)
PREFIX = "benchmarks-13c8deb68a873635879c9a69bc78caebd340f646/lib/PBE_SLIA_Track/from_2018/"
MAX_STRING, MAX_TOKENS, MAX_DEPTH = 100_000, 2000, 60
TOKEN = re.compile(r'\s*("(?:[^"]|"")*"|\(|\)|[^\s()"]+)')


def sexp(text):
    """Parse one or more S-expressions; strings become ('str', value), numerals ints, other atoms str."""
    tokens, pos = [t for t in TOKEN.findall(re.sub(r";[^\n]*", "", text))], 0

    def item(depth):
        nonlocal pos
        if pos >= len(tokens) or depth > MAX_DEPTH + 40:
            raise ValueError("incomplete or too deep expression")
        token = tokens[pos]
        pos += 1
        if token == "(":
            out = []
            while pos < len(tokens) and tokens[pos] != ")":
                out.append(item(depth + 1))
            if pos >= len(tokens):
                raise ValueError("unbalanced parentheses")
            pos += 1
            return out
        if token == ")":
            raise ValueError("unexpected )")
        if token.startswith('"'):
            return ("str", token[1:-1].replace('""', '"'))
        if re.fullmatch(r"-?[0-9]+", token):
            return int(token)
        return token
    items = []
    while pos < len(tokens):
        items.append(item(0))
    return items


def parse_benchmark(text):
    function, examples = None, []
    for form in sexp(text):
        if isinstance(form, list) and form and form[0] == "synth-fun":
            _, name, params, sort, grammar = form
            rules = {nt: (nt_sort, productions) for nt, nt_sort, productions in grammar}
            function = {"name": name, "params": [[p, s] for p, s in params], "sort": sort, "start": grammar[0][0], "grammar": rules}
        elif isinstance(form, list) and form and form[0] == "constraint":
            (_, (eq, (fname, *args), output)) = form
            if eq != "=" or fname != function["name"]:
                raise ValueError("unsupported constraint")
            examples.append([[a[1] if isinstance(a, tuple) else a for a in args], output[1] if isinstance(output, tuple) else output])
    return function, examples


def _encode(term):
    """JSON-friendly form of a parsed term (strings tagged to keep them apart from symbols)."""
    if isinstance(term, tuple):
        return {"str": term[1]}
    if isinstance(term, list):
        return [_encode(t) for t in term]
    return term


def _decode(term):
    if isinstance(term, dict):
        return ("str", term["str"])
    if isinstance(term, list):
        return [_decode(t) for t in term]
    return term


def derive(raw):
    tasks = []
    with zipfile.ZipFile(io.BytesIO(raw[SOURCE])) as archive:
        names = sorted(n[len(PREFIX):-len("-long.sl")] for n in archive.namelist() if n.startswith(PREFIX) and n.endswith("-long.sl"))
        for name in names:
            function, visible = parse_benchmark(archive.read(f"{PREFIX}{name}.sl").decode("utf-8"))
            _, long_examples = parse_benchmark(archive.read(f"{PREFIX}{name}-long.sl").decode("utf-8"))
            seen = {json.dumps(i): o for i, o in visible}
            if any(seen.get(json.dumps(i), o) != o for i, o in long_examples):
                raise ValueError(f"{name}: base and long examples disagree")
            held_out = [[i, o] for i, o in long_examples if json.dumps(i) not in seen]
            grammar = [[nt, sort, [_encode(p) for p in productions]] for nt, (sort, productions) in function["grammar"].items()]
            tasks.append({"name": name, "function": function["name"], "params": function["params"], "sort": function["sort"],
                          "start": function["start"], "grammar": grammar, "examples": visible, "held_out": held_out})
    return {"pbe/tasks.json": json.dumps(tasks, sort_keys=True, separators=(",", ":")).encode()}


class EvalError(Exception):
    pass


def _str(value):
    if not isinstance(value, str):
        raise EvalError("expected a string")
    return value


def _int(value):
    if type(value) is not int:
        raise EvalError("expected an integer")
    return value


def evaluate(term, env):
    """SMT-LIB 2.6 semantics of the PBE string operations (old and new names)."""
    if isinstance(term, tuple):
        return term[1]
    if isinstance(term, bool):
        return term
    if type(term) is int:
        return term
    if isinstance(term, str):
        if term in env:
            return env[term]
        if term in ("true", "false"):
            return term == "true"
        raise EvalError(f"unknown symbol {term}")
    op, args = term[0], term[1:]
    if op == "ite":
        return evaluate(args[1] if _bool(evaluate(args[0], env)) else args[2], env)
    values = [evaluate(a, env) for a in args]
    if op == "str.++":
        out = "".join(_str(v) for v in values)
        if len(out) > MAX_STRING:
            raise EvalError("string too long")
        return out
    if op == "str.len":
        return len(_str(values[0]))
    if op == "str.at":
        s, i = _str(values[0]), _int(values[1])
        return s[i] if 0 <= i < len(s) else ""
    if op == "str.substr":
        s, i, n = _str(values[0]), _int(values[1]), _int(values[2])
        return s[i:i + n] if 0 <= i < len(s) and n > 0 else ""
    if op == "str.indexof":
        s, t, i = _str(values[0]), _str(values[1]), _int(values[2])
        if not 0 <= i <= len(s):
            return -1
        return s.find(t, i)
    if op == "str.replace":
        s, t, u = (_str(v) for v in values)
        out = u + s if t == "" else s.replace(t, u, 1)
        if len(out) > MAX_STRING:
            raise EvalError("string too long")
        return out
    if op in ("str.to.int", "str.to_int"):
        s = _str(values[0])
        return int(s) if s and s.isascii() and s.isdigit() else -1
    if op in ("int.to.str", "str.from_int"):
        i = _int(values[0])
        return str(i) if i >= 0 else ""
    if op == "str.prefixof":
        return _str(values[1]).startswith(_str(values[0]))
    if op == "str.suffixof":
        return _str(values[1]).endswith(_str(values[0]))
    if op == "str.contains":
        return _str(values[1]) in _str(values[0])
    if op == "+":
        return sum(_int(v) for v in values)
    if op == "-":
        return -_int(values[0]) if len(values) == 1 else _int(values[0]) - sum(_int(v) for v in values[1:])
    if op == "=":
        return values[0] == values[1] and type(values[0]) is type(values[1])
    if op in ("<", "<=", ">", ">="):
        a, b = _int(values[0]), _int(values[1])
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
    if op in ("and", "or"):
        flags = [_bool(v) for v in values]
        return all(flags) if op == "and" else any(flags)
    if op == "not":
        return not _bool(values[0])
    raise EvalError(f"unknown operation {op}")


def _bool(value):
    if not isinstance(value, bool):
        raise EvalError("expected a Boolean")
    return value


def conforms(term, nonterminal, grammar):
    """True when the term can be derived from the nonterminal of the grammar {nt: (sort, productions)}."""
    visiting = set()

    def derive(t, nt):
        key = (id(t), nt)
        if key in visiting:
            return False
        visiting.add(key)
        try:
            return any(match(t, production) for production in grammar[nt][1])
        finally:
            visiting.discard(key)

    def match(t, production):
        if isinstance(production, str) and production in grammar:
            return derive(t, production)
        if isinstance(production, list):
            return (isinstance(t, list) and len(t) == len(production) and t[0] == production[0]
                    and all(match(a, b) for a, b in zip(t[1:], production[1:])))
        return t == production and type(t) is type(production)
    return derive(term, nonterminal)


def _size(term):
    return 1 + sum(_size(t) for t in term[1:]) if isinstance(term, list) else 1


def check_program(task, text):
    """Parsed term of a candidate program, or None when it does not parse, is too large or leaves the grammar."""
    try:
        terms = sexp(text)
    except (ValueError, RecursionError):
        return None
    if len(terms) != 1 or _size(terms[0]) > MAX_TOKENS:
        return None
    grammar = {nt: (sort, [_decode(p) for p in productions]) for nt, sort, productions in task["grammar"]}
    return terms[0] if conforms(terms[0], task["start"], grammar) else None


def run_examples(task, term, examples):
    names = [p for p, _ in task["params"]]
    passed = 0
    for inputs, output in examples:
        try:
            passed += evaluate(term, dict(zip(names, inputs))) == output
        except (EvalError, RecursionError):
            pass
    return passed


def public_task(task):
    return {k: task[k] for k in ("name", "function", "params", "sort", "start", "grammar", "examples")}


DRIVER = '''

def _artask_synthesize_all(tasks):
    return [synthesize(task) for task in tasks]
'''


def pbe_score(module, *, cpu_seconds=120):
    from pathlib import Path
    path = software_sandbox.require_entry(module, "synthesize")
    tasks = software_data.load_json("pbe/tasks.json")
    source = Path(path).read_text(encoding="utf-8") + DRIVER
    try:
        programs = software_sandbox.execute(source, "_artask_synthesize_all", [[public_task(t) for t in tasks]],
                                            cpu_seconds=cpu_seconds, memory_mb=2048, filename="candidate.py")
    except software_sandbox.SandboxFailure as failure:
        raise CandidateError(f"candidate synthesize() failed in the sandbox ({failure.kind}): {failure}"[:2000]) from None
    if not isinstance(programs, list) or len(programs) != len(tasks) or not all(isinstance(p, str) and len(p) <= 50000 for p in programs):
        raise CandidateError("synthesize() must return one program (a str of at most 50000 characters) per task")
    statuses = {}
    for task, text in zip(tasks, programs):
        term = check_program(task, text)
        if term is None:
            statuses[task["name"]] = {"status": "rejected"}
            continue
        visible = run_examples(task, term, task["examples"])
        held = run_examples(task, term, task["held_out"])
        ok = visible == len(task["examples"])
        status = "solved" if ok and held == len(task["held_out"]) else ("overfitting" if ok else "wrong")
        statuses[task["name"]] = {"status": status, "visible": f"{visible}/{len(task['examples'])}", "held_out": f"{held}/{len(task['held_out'])}"}
    count = lambda s: sum(v["status"] == s for v in statuses.values())
    return count("solved") / len(tasks), {"tasks": len(tasks), "solved": count("solved"), "overfitting": count("overfitting"),
                                          "wrong": count("wrong"), "rejected": count("rejected"), "per_task": statuses}
