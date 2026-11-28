"""Loop-invariant inference on the Code2Inv benchmark (Si et al., NeurIPS 2018), checked with Z3.

Each of the 133 C programs (one loop, linear integer arithmetic) comes with
verification conditions in SMT-LIB: an uninterpreted invariant inv-f over the
program variables and definitions pre-f, trans-f and post-f over SSA copies.
An invariant is correct when the three conditions hold (initiation:
pre => inv; consecution: inv and trans => inv'; safety: inv => post).

Candidate invariants are S-expressions parsed here with a whitelist of
operators and the problem's variables, then re-serialised before they reach
Z3, so no SMT-LIB command can be injected. Every check runs with a fixed Z3
resource limit (deterministic); an exhausted limit counts as not verified.

Preparation additionally searches each program for a counterexample to
safety with bounded model checking (up to 8 loop steps) and keeps only the
programs where none exists; Ryan et al. (ICLR 2020) report that 9 of the 133
programs violate their post-condition.
"""
from __future__ import annotations

import io
import json
import re
import zipfile

from . import software_data, software_sandbox
from .runtime import CandidateError

SOURCE = "code2inv"
SOURCES = (SOURCE,)
PREFIX = "code2inv-336a7063ae0d6313a6993dfb897766d01d5ba41a/benchmarks/C_instances/"
SPLIT = "SPLIT_HERE_asdfghjklzxcvbnmqwertyuiop"
RLIMIT = 1_000_000  # per condition check (about 0.25 s when exhausted)
QUERY_RESOURCES = 400_000_000  # all oracle queries of one evaluation together
BMC_STEPS = 8
OPS = {"and": (1, None, "bool", "bool"), "or": (1, None, "bool", "bool"), "not": (1, 1, "bool", "bool"),
       "=>": (2, 2, "bool", "bool"), "ite": (3, 3, None, None), "=": (2, None, "int", "bool"),
       "<": (2, 2, "int", "bool"), "<=": (2, 2, "int", "bool"), ">": (2, 2, "int", "bool"), ">=": (2, 2, "int", "bool"),
       "+": (1, None, "int", "int"), "-": (1, None, "int", "int"), "*": (2, None, "int", "int"),
       "div": (2, 2, "int", "int"), "mod": (2, 2, "int", "int")}
MAX_TOKENS, MAX_DEPTH, MAX_LITERAL = 4000, 60, 10 ** 9


def _tokens(text):
    if not isinstance(text, str) or len(text) > 50000:
        raise CandidateError("an invariant must be a str of at most 50000 characters")
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    if len(tokens) > MAX_TOKENS:
        raise CandidateError(f"an invariant may have at most {MAX_TOKENS} tokens")
    return tokens


def parse(text, variables):
    """Whitelisted, sort-checked S-expression -> canonical SMT-LIB string."""
    tokens, pos = _tokens(text), 0

    def term(depth):
        nonlocal pos
        if depth > MAX_DEPTH or pos >= len(tokens):
            raise CandidateError("invariant is too deep or incomplete")
        token = tokens[pos]
        pos += 1
        if token == "(":
            if pos >= len(tokens) or tokens[pos] not in OPS:
                raise CandidateError(f"unknown operator {tokens[pos] if pos < len(tokens) else 'EOF'!r}")
            op = tokens[pos]
            pos += 1
            args = []
            while pos < len(tokens) and tokens[pos] != ")":
                args.append(term(depth + 1))
            if pos >= len(tokens):
                raise CandidateError("unbalanced parentheses in invariant")
            pos += 1
            low, high, arg_sort, sort = OPS[op]
            if len(args) < low or (high is not None and len(args) > high):
                raise CandidateError(f"wrong number of arguments for {op}")
            if op == "ite":
                if args[0][1] != "bool" or args[1][1] != args[2][1]:
                    raise CandidateError("ite needs a Boolean condition and branches of one sort")
                sort = args[1][1]
            elif op == "=":
                if len({s for _, s, _ in args}) != 1:
                    raise CandidateError("= needs arguments of one sort")
            elif any(s != arg_sort for _, s, _ in args):
                raise CandidateError(f"{op} needs {arg_sort} arguments")
            if op == "*" and sum(not constant for _, _, constant in args) > 1:
                raise CandidateError("* may multiply at most one non-constant term (linear arithmetic only)")
            if op in ("div", "mod") and not (re.fullmatch(r"[0-9]+", args[1][0]) and int(args[1][0]) > 0):
                raise CandidateError(f"{op} needs a positive integer literal as divisor")
            return "(" + " ".join([op] + [a for a, _, _ in args]) + ")", sort, all(c for _, _, c in args)
        if token in ("true", "false"):
            return token, "bool", True
        if re.fullmatch(r"[0-9]+", token):
            if int(token) > MAX_LITERAL:
                raise CandidateError(f"integer literals must be at most {MAX_LITERAL}")
            return str(int(token)), "int", True
        if token in variables:
            return token, "int", False
        raise CandidateError(f"unknown symbol {token!r} (variables: {', '.join(variables)})")

    expression, sort, _ = term(0)
    if pos != len(tokens) or sort != "bool":
        raise CandidateError("an invariant must be exactly one Boolean term")
    return expression


def split_problem(name, c_source, smt):
    parts = smt.split(SPLIT)
    if len(parts) != 5:
        raise ValueError(f"{name}: unexpected verification-condition layout")
    head = re.search(r"\(\s*define-fun\s+inv-f\s*\((.*?)\)\s*Bool\s*$", parts[0], re.S)
    params = re.findall(r"\(\s*(\S+)\s+(\S+)\s*\)", head.group(1))
    if not params or any(sort != "Int" for _, sort in params):
        raise ValueError(f"{name}: invariant parameters must be integers")
    return {"name": name, "c_source": c_source, "variables": [v for v, _ in params], "head": parts[0],
            "definitions": parts[1], "conditions": {"initiation": parts[2], "consecution": parts[3], "safety": parts[4]}}


def _signature(definitions, function):
    match = re.search(r"\(\s*define-fun\s+" + re.escape(function) + r"\s*\(((?:\s*\(\s*\S+\s+\S+\s*\))*)\s*\)", definitions)
    return [v for v, _ in re.findall(r"\(\s*(\S+)\s+(\S+)\s*\)", match.group(1))]


def bmc_counterexample(problem, steps=BMC_STEPS):
    """Smallest number of loop steps after which some execution violates the post-condition, or None."""
    import z3
    pre, trans, post = (_signature(problem["definitions"], f) for f in ("pre-f", "trans-f", "post-f"))
    state = problem["variables"]
    aux = [v for v in pre if v not in state]
    for k in range(steps + 1):
        lines = [problem["head"] + " true " + problem["definitions"]]
        copies = [{v: f"|s{i}_{v}|" for v in state} for i in range(k + 1)]
        auxes = {tag: {v: f"|{tag}_{v}|" for v in aux} for tag in ["pre", "post"] + [f"t{i}" for i in range(k)]}
        names = [n for c in copies for n in c.values()] + [n for a in auxes.values() for n in a.values()]
        lines += [f"(declare-const {n} Int)" for n in names]
        lines.append("(assert (pre-f " + " ".join({**copies[0], **auxes["pre"]}[v] for v in pre) + "))")
        for i in range(k):
            step = {**copies[i], **{v + "!": copies[i + 1][v] for v in state}, **auxes[f"t{i}"]}
            lines.append("(assert (trans-f " + " ".join(step[v] for v in trans) + "))")
        lines.append("(assert (not (post-f " + " ".join({**copies[k], **auxes["post"]}[v] for v in post) + ")))")
        solver = z3.Solver()
        solver.set("rlimit", 50 * RLIMIT)
        solver.from_string("\n".join(lines))
        result = solver.check()
        if result == z3.sat:
            return k
        if result != z3.unsat:
            raise RuntimeError(f"{problem['name']}: bounded check inconclusive at {k} steps")
    return None


def derive(raw):
    problems, invalid = [], []
    with zipfile.ZipFile(io.BytesIO(raw[SOURCE])) as archive:
        for n in range(1, 134):
            problem = split_problem(str(n), archive.read(f"{PREFIX}c/{n}.c").decode(), archive.read(f"{PREFIX}c_smt2/{n}.c.smt").decode())
            steps = bmc_counterexample(problem)
            (invalid if steps is not None else problems).append(problem if steps is None else {"name": str(n), "violation_after_steps": steps})
    return {"code2inv/problems.json": json.dumps(problems, sort_keys=True, separators=(",", ":")).encode(),
            "code2inv/invalid.json": json.dumps(invalid, sort_keys=True, separators=(",", ":")).encode()}


def _resource_count(solver):
    statistics = solver.statistics()
    return statistics.get_key_value("rlimit count") if "rlimit count" in statistics.keys() else 0


def check(problem, expression, usage=None):
    """{'valid': bool, 'failed': condition or None, 'state': {...}, 'next': {...}} for a parsed invariant.

    ``usage`` (a one-element list) accumulates the Z3 resource units the checks consumed."""
    import z3
    for condition in ("initiation", "consecution", "safety"):
        solver = z3.Solver()
        solver.set("rlimit", RLIMIT)
        solver.from_string(problem["head"] + " " + expression + " " + problem["definitions"] + problem["conditions"][condition])
        before = _resource_count(solver)
        result = solver.check()
        if usage is not None:
            usage[0] += max(0, _resource_count(solver) - before)
        if result == z3.unsat:
            continue
        reply = {"valid": False, "failed": condition, "state": None, "next": None}
        if result == z3.sat:
            model = solver.model()
            values = {d.name(): model[d] for d in model.decls()}
            read = lambda name: values[name].as_long() if name in values and z3.is_int_value(values[name]) else 0
            reply["state"] = {v: read(v) for v in problem["variables"]}
            if condition == "consecution":
                reply["next"] = {v: read(v + "!") for v in problem["variables"]}
        else:
            reply["failed"] = condition + " (resource limit reached)"
        return reply
    return {"valid": True, "failed": None, "state": None, "next": None}


class _Oracle:
    """Verification oracle with a per-problem query budget (the final answers are checked separately)."""

    def __init__(self, problems, budget):
        self.problems, self.budget, self.used, self.resources = problems, budget, [0] * len(problems), [0]

    def __call__(self, request):
        index = request.get("problem") if isinstance(request, dict) else None
        if type(index) is not int or not 0 <= index < len(self.problems):
            raise CandidateError("malformed oracle request")
        if self.used[index] >= self.budget or self.resources[0] >= QUERY_RESOURCES:
            return {"valid": None, "failed": "query budget exhausted", "state": None, "next": None, "queries_left": 0}
        self.used[index] += 1
        try:
            reply = check(self.problems[index], parse(request.get("invariant"), self.problems[index]["variables"]), self.resources)
        except CandidateError as error:
            reply = {"valid": False, "failed": "syntax", "error": str(error), "state": None, "next": None}
        return dict(reply, queries_left=self.budget - self.used[index])


DRIVER = '''

def _artask_invariants(problems, oracle):
    answers = []
    for index, problem in enumerate(problems):
        answers.append(synthesize(problem, lambda invariant, index=index: oracle({"problem": index, "invariant": invariant})))
    return answers
'''


def public_problem(problem, budget):
    return {"name": problem["name"], "c_source": problem["c_source"], "variables": list(problem["variables"]),
            "definitions": problem["definitions"], "max_queries": budget}


def invariant_score(module, *, cpu_seconds=180, budget=50):
    from pathlib import Path
    path = software_sandbox.require_entry(module, "synthesize")
    problems = software_data.load_json("code2inv/problems.json")
    oracle = _Oracle(problems, budget)
    source = Path(path).read_text(encoding="utf-8") + DRIVER
    try:
        answers = software_sandbox.execute(source, "_artask_invariants", [[public_problem(p, budget) for p in problems]],
                                           oracle=oracle, cpu_seconds=cpu_seconds, wall_seconds=1.5 * cpu_seconds + 200,
                                           memory_mb=2048, filename="candidate.py")
    except software_sandbox.SandboxFailure as failure:
        raise CandidateError(f"candidate synthesize() failed in the sandbox ({failure.kind}): {failure}"[:2000]) from None
    if not isinstance(answers, list) or len(answers) != len(problems) or not all(isinstance(a, str) for a in answers):
        raise CandidateError("synthesize() must return one invariant (a str) per problem")
    solved, failures = [], {}
    for problem, answer in zip(problems, answers):
        try:
            verdict = check(problem, parse(answer, problem["variables"]))
        except CandidateError as error:
            verdict = {"valid": False, "failed": f"syntax: {error}"}
        if verdict["valid"]:
            solved.append(problem["name"])
        else:
            failures[problem["name"]] = verdict["failed"]
    return len(solved) / len(problems), {"problems": len(problems), "verified": len(solved), "queries_used": sum(oracle.used),
                                         "query_resources_used": oracle.resources[0], "failed_conditions": failures}
