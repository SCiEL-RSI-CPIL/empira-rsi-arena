"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Seed for QuixBugs repair: generate-and-validate over single-node mutations (in the spirit of GenProg-style repair).

Candidate patches replace one comparison or arithmetic operator, or shift one integer constant by one;
they are tried in source order and the first one that passes every visible test is returned.
"""
import ast
import copy
import json
import signal

COMPARE = [ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq]
ARITH = [ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _passes(source, entry, tests, seconds=0.2):
    namespace = {"__name__": "candidate_patch"}
    try:
        exec(compile(source, "<patch>", "exec"), namespace)
    except Exception:
        return False
    function = namespace.get(entry)
    if not callable(function):
        return False
    signal.signal(signal.SIGALRM, _alarm)
    for test in tests:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            output = function(*copy.deepcopy(test["input"]))
            if hasattr(output, "__next__"):
                output = list(output)
            output = json.loads(json.dumps(output))
        except BaseException:
            return False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        expected = test["expected"]
        if test.get("tolerance") is not None:
            if not isinstance(output, (int, float)) or abs(output - expected) > test["tolerance"]:
                return False
        elif output != expected:
            return False
    return True


def _mutants(tree):
    """Yield mutated copies of the tree, one changed node each."""
    sites = [node for node in ast.walk(tree) if isinstance(node, (ast.Compare, ast.BinOp, ast.AugAssign, ast.Constant))]
    for index in range(len(sites)):
        node = sites[index]
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            options = [lambda n, op=op: setattr(n, "ops", [op()]) for op in COMPARE if not isinstance(node.ops[0], op)]
        elif isinstance(node, (ast.BinOp, ast.AugAssign)):
            options = [lambda n, op=op: setattr(n, "op", op()) for op in ARITH if not isinstance(node.op, op)]
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            options = [lambda n, d=d: setattr(n, "value", n.value + d) for d in (1, -1)]
        else:
            continue
        for change in options:
            mutant = copy.deepcopy(tree)
            target = [n for n in ast.walk(mutant) if isinstance(n, (ast.Compare, ast.BinOp, ast.AugAssign, ast.Constant))][index]
            change(target)
            yield mutant


def repair(task):
    source, entry, tests = task["source"], task["entry"], task["tests"]
    tree = ast.parse(source)
    for count, mutant in enumerate(_mutants(tree)):
        if count >= 400:
            break
        patched = ast.unparse(mutant)
        if _passes(patched, entry, tests):
            return patched
    return source
# EVOLVE-BLOCK-END
