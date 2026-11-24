"""Sandbox-side harness for the Python-module subjects (standard library only, executed as job source).

Subject modules are served from source texts by an in-memory import finder
under the private top-level package ``artask_subject`` (so the interpreter's
own copies are untouched). ``coverage`` instruments chosen files with block
probes and replays fuzzing inputs; ``mutation_analysis`` replays test calls on
the original module and on each mutant and reports which mutants are killed.
The per-subject drivers below turn JSON inputs into calls; they are part of
the trusted harness, not of the mutated or instrumented code.

Every call of subject code has a step budget: each function entry and each
loop iteration inside a function of the subject's files spends one step, and
the call is cut off when the budget is used up. The budget is deterministic,
so whether an ordinary loop counts as timed out does not depend on the speed
or load of the machine. CPU-time limits remain for what steps do not measure:
calls that spend long inside C code, and loops whose single iterations become
ever more expensive (a mutant that builds ever longer strings would otherwise
exhaust the memory first). A test whose call on the original module needs more
than CPU_SECONDS is not used; the same call on a mutant is stopped after
CPU_SECONDS plus SLOWDOWN times the CPU time the original needed, so a run
that merely took a little longer than the original's is never a time-out.
"""
import ast
import hashlib
import importlib.abc
import importlib.util
import signal
import sys
import time

PREFIX = "artask_subject"
PROBE = "__artask_probe__"
STEP = "__artask_step__"  # the probe of a function entry or loop body; it also spends one step
STEP_LIMIT = 50_000  # steps per test call (mutation analysis and spectra); 5 times the largest call the inputs allow
FUZZ_STEP_LIMIT = 1_000_000  # steps per fuzzing input
CPU_SECONDS = 0.2  # CPU time per test call on the original module; the steps above cost a small fraction of it
SLOWDOWN = 10  # a mutant's call gets CPU_SECONDS plus this many times the original's CPU time
FUZZ_CPU_SECONDS = 5.0  # CPU time per fuzzing input, several times what FUZZ_STEP_LIMIT steps cost
_STEPS = [0]  # steps left for the running call
_ARMED = [False]  # whether the CPU-time signal may interrupt the running call


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    if _ARMED[0]:
        _ARMED[0] = False  # one interruption per call, and none once the call's outcome is being recorded
        raise _Timeout()


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves artask_subject.<dotted> from {relative path: source}; directories become packages."""

    def __init__(self, files, transform=None, probe=None, step=None):
        self.modules, self.probe = {}, probe or _noop
        self.step = step or self.probe
        for path, source in files.items():
            parts = path[:-3].split("/")
            package = parts[-1] == "__init__"
            name = ".".join([PREFIX] + (parts[:-1] if package else parts))
            self.modules[name] = (path, source, package)
            for i in range(1, len(parts) - (1 if package else 0)):
                self.modules.setdefault(".".join([PREFIX] + parts[:i]), (None, "", True))
        self.modules.setdefault(PREFIX, (None, "", True))
        self.transform = transform

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            return importlib.util.spec_from_loader(fullname, self, is_package=self.modules[fullname][2])
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, source, _ = self.modules[module.__name__]
        tree = ast.parse(source, filename=path or module.__name__)
        if self.transform is not None and path is not None:
            tree = self.transform(path, tree)
        module.__dict__[PROBE], module.__dict__[STEP] = self.probe, self.step
        exec(compile(tree, f"<{path or module.__name__}>", "exec"), module.__dict__)


def _noop(_):
    pass


def _spending(record=None):
    """Step probe: spends one step of the running call's budget and reports the block to ``record``."""
    steps = _STEPS

    def step(block):
        steps[0] -= 1
        if steps[0] < 0:
            raise _Timeout()  # raised again at every later step, so a handler in the subject cannot absorb it
        if record is not None:
            record(block)
    return step


def _add_steps(path, tree):
    """Transform for mutation analysis: only the step probes of _Instrument, without the other block probes.

    A walk over statements inserts them, which costs a fraction of visiting every node for each mutant."""
    def step(at):
        where = {"lineno": at.lineno, "col_offset": at.col_offset, "end_lineno": at.lineno, "end_col_offset": at.col_offset}
        call = ast.Call(func=ast.Name(id=STEP, ctx=ast.Load(), **where), args=[ast.Constant(value=0, **where)], keywords=[], **where)
        return ast.Expr(value=call, **where)

    def walk(body, inside):
        for node in body:
            function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                    walk(block, inside or function)
            for part in getattr(node, "handlers", []) + getattr(node, "cases", []):
                walk(part.body, inside)
            if function:
                node.body.insert(_docstring(node.body), step(node))
            elif inside and isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                node.body.insert(0, step(node))
    walk(tree.body, False)
    return tree


def _docstring(body):
    """1 when the block starts with a docstring (a probe goes after it), else 0."""
    first = body[0] if body else None
    return int(isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str))


def load(files, transform=None, probe=None, step=None):
    """Fresh import of a subject; returns the import function for dotted names below the prefix."""
    for name in [n for n in sys.modules if n == PREFIX or n.startswith(PREFIX + ".")]:
        del sys.modules[name]
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _Finder)]
    sys.meta_path.insert(0, _Finder(files, transform, probe, step))
    import importlib
    return lambda dotted: importlib.import_module(PREFIX + "." + dotted)


class _Instrument(ast.NodeTransformer):
    """Inserts a numbered probe at the start of every block inside functions (bodies, branches, handlers, loop exits)."""

    def __init__(self):
        self.count, self.depth, self.blocks = 0, 0, {}

    def _probe(self, name=PROBE):
        self.count += 1
        call = ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[ast.Constant(self.count - 1)], keywords=[])
        return ast.Expr(value=call)

    def _block(self, body, keep_docstring=False, step=False):
        start = _docstring(body) if keep_docstring else 0
        probe = self._probe(STEP if step else PROBE)
        self.blocks[self.count - 1] = [(s.lineno, s.end_lineno) for s in body]  # statement spans of this block
        return body[:start] + [probe] + body[start:]

    def visit_FunctionDef(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        node.body = self._block(node.body, keep_docstring=True, step=True)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def _branches(self, node, fields, steps=()):
        self.generic_visit(node)
        if self.depth:
            for field in fields:
                setattr(node, field, self._block(getattr(node, field), step=field in steps))
        return node

    def visit_If(self, node):
        return self._branches(node, ("body", "orelse"))

    def visit_For(self, node):
        return self._branches(node, ("body", "orelse"), steps=("body",))

    visit_While = visit_AsyncFor = visit_For

    def visit_Try(self, node):
        # An else block needs a handler: a probe there would make try/finally without handlers invalid.
        fields = ("body",) + (("orelse",) if node.handlers else ()) + (("finalbody",) if node.finalbody else ())
        return self._branches(node, fields)

    visit_TryStar = visit_Try

    def visit_ExceptHandler(self, node):
        return self._branches(node, ("body",))

    def visit_With(self, node):
        return self._branches(node, ("body",))

    visit_AsyncWith = visit_With

    def visit_match_case(self, node):
        return self._branches(node, ("body",))

    def visit_IfExp(self, node):
        self.generic_visit(node)
        if self.depth:
            for field in ("body", "orelse"):
                probe = self._probe().value
                setattr(node, field, ast.BoolOp(op=ast.Or(), values=[probe, getattr(node, field)]))
        return node


def _run(function, steps, seconds):
    """Outcome of one call: ['ok', hash of the repr], ['raise', exception type] or ['timeout'] (a limit was reached)."""
    _STEPS[0] = steps
    signal.signal(signal.SIGPROF, _alarm)
    try:
        _ARMED[0] = True
        signal.setitimer(signal.ITIMER_PROF, seconds)
        try:
            value = function()
            if hasattr(value, "__next__"):
                value = [item for _, item in zip(range(10001), value)]
            outcome = ["ok", hashlib.sha256(repr(value).encode("utf-8", "surrogatepass")).hexdigest()[:24]]
        except _Timeout:
            outcome = ["timeout"]
        except RecursionError:
            outcome = ["raise", "RecursionError"]
        except BaseException as error:  # noqa: BLE001 - the exception type is the observable outcome
            outcome = ["raise", type(error).__name__]
        _ARMED[0] = False
    except _Timeout:
        # The signal arrived after the call had ended, typically because a long operation in C code delayed it
        # (signals are handled between bytecodes). It must not escape from here and abort the whole analysis.
        outcome = ["timeout"]
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
    return outcome


def _textwrap(imp, call, args):
    m = imp("textwrap")
    if call == "dedent":
        return lambda: m.dedent(args[0])
    if call == "indent":
        return lambda: m.indent(args[0], args[1])
    return lambda: getattr(m, call)(args[0], args[1], **args[2])


def _urllib(imp, call, args):
    m = imp("urllib.parse")
    if call in ("urlunsplit", "urlunparse"):
        return lambda: getattr(m, call)(tuple(args[0]))
    if call == "urlencode":
        return lambda: m.urlencode([tuple(p) for p in args[0]], doseq=args[1], safe=args[2])
    return lambda: getattr(m, call)(*args)


def _fractions(imp, call, args):
    m = imp("fractions")
    F = m.Fraction
    convert = lambda x: F(x) if isinstance(x, str) else x
    if call == "parse":
        return lambda: F(args[0])
    if call == "make":
        return lambda: F(args[0], args[1])
    if call == "from_float":
        return lambda: F.from_float(args[0])
    if call == "binary":
        import operator
        ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "//": operator.floordiv,
               "%": operator.mod, "**": operator.pow, "<": operator.lt, "<=": operator.le, "==": operator.eq,
               "divmod": divmod, "max": max}
        return lambda: ops[args[1]](convert(args[0]), convert(args[2]))
    if call == "limit_denominator":
        return lambda: F(args[0]).limit_denominator(args[1])
    if call == "round":
        return lambda: round(F(args[0]), args[1]) if args[1] is not None else round(F(args[0]))
    if call == "format":
        return lambda: format(F(args[0]), args[1])
    if call == "method":
        return lambda: getattr(F(args[0]), args[1])()
    raise ValueError(call)


def _statistics(imp, call, args):
    m = imp("statistics")
    from fractions import Fraction
    values = lambda xs: [Fraction(x) if isinstance(x, str) else x for x in xs]
    number = lambda x: Fraction(x) if isinstance(x, str) else x
    if call == "summary":
        return lambda: getattr(m, args[0])(values(args[1]))
    if call == "median_grouped":
        return lambda: m.median_grouped(values(args[0]), number(args[1]))
    if call == "quantiles":
        return lambda: m.quantiles(values(args[0]), n=args[1], method=args[2])
    if call == "weighted":
        return lambda: getattr(m, args[0])(values(args[1]), weights=values(args[2]))
    if call == "spread_given_center":
        return lambda: getattr(m, args[0])(values(args[1]), number(args[2]))
    if call == "bivariate":
        name, x, y = args[0], values(args[1]), values(args[2])
        if name == "correlation_ranked":
            return lambda: m.correlation(x, y, method="ranked")
        if name == "linear_regression_proportional":
            return lambda: m.linear_regression(x, y, proportional=True)
        return lambda: getattr(m, name)(x, y)
    if call == "normal":
        def run():
            dist = m.NormalDist(number(args[0]), number(args[1]))
            if args[2] == "quantiles":
                n = int(number(args[3]))
                if not 1 <= n <= 1000:
                    raise ValueError("the driver accepts 1 <= n <= 1000 quantiles")
                return dist.quantiles(n)
            if args[2] == "overlap":
                return dist.overlap(m.NormalDist(number(args[3]), 1.0))
            return getattr(dist, args[2])(number(args[3]))
        return run
    if call == "from_samples":
        return lambda: m.NormalDist.from_samples(values(args[0]))
    raise ValueError(call)


MUTATION_DRIVERS = {"textwrap": _textwrap, "urllib.parse": _urllib, "fractions": _fractions, "statistics": _statistics}
FUZZ_DRIVERS = {
    "tomllib": lambda imp: (lambda m: lambda s: m.loads(s))(imp("tomllib")),
    "re": lambda imp: (lambda m: lambda s: m.compile(s, 0))(imp("re._compiler")),
    "html.parser": lambda imp: (lambda m: lambda s: [_feed(m.HTMLParser(convert_charrefs=flag), s) for flag in (True, False)])(imp("html.parser")),
    "plistlib": lambda imp: (lambda m: lambda s: m.loads(s.encode("latin-1")))(imp("plistlib")),
}


def _feed(parser, text):
    parser.feed(text)
    parser.close()


def coverage(files, instrumented, subject, inputs, steps=FUZZ_STEP_LIMIT):
    """Sorted probe ids hit by the inputs, the probe count, and the number of inputs that reached a limit."""
    hits, counts = set(), {}

    def transform(path, tree):
        if path not in instrumented:
            return tree
        visitor = _Instrument()
        visitor.count = sum(counts.values())
        tree = ast.fix_missing_locations(visitor.visit(tree))
        counts[path] = visitor.count - sum(counts.values())
        return tree
    _STEPS[0] = steps  # importing the subject runs probed code as well
    imp = load(files, transform, hits.add, _spending(hits.add))
    target = FUZZ_DRIVERS[subject](imp)
    timeouts = 0
    for text in inputs:
        timeouts += _run(lambda: target(text), steps, FUZZ_CPU_SECONDS) == ["timeout"]
    return {"hits": sorted(hits), "probes": sum(counts.values()), "timeouts": timeouts}


def _originals(files, subject, tests, steps, transform):
    """Outcomes of the tests on the original module and the CPU limit of each test's call on a mutant."""
    _STEPS[0] = steps
    imp = load(files, transform, None, _spending())
    driver = MUTATION_DRIVERS[subject]
    results, limits = [], []
    for test in tests:
        start = time.process_time()
        results.append(_run(driver(imp, test["call"], test["args"]), steps, CPU_SECONDS))
        limits.append(CPU_SECONDS + SLOWDOWN * (time.process_time() - start))
    return results, limits


def outcomes(files, subject, tests, steps=STEP_LIMIT):
    return _originals(files, subject, tests, steps, _add_steps)[0]


def mutation_analysis(files, target, subject, tests, mutants, steps=STEP_LIMIT):
    """Original outcomes plus, per mutant, the index of the first killing test (or -1)."""
    original, limits = _originals(files, subject, tests, steps, _add_steps)
    usable = [i for i, o in enumerate(original) if o[0] != "timeout"]
    lines = files[target].splitlines(keepends=True)
    killed = []
    for start, end, replacement in mutants:
        variant = dict(files)
        variant[target] = "".join(lines[:start - 1]) + replacement + "".join(lines[end:])
        try:
            _STEPS[0] = steps
            imp = load(variant, _add_steps, None, _spending())
            driver = MUTATION_DRIVERS[subject]
            imp(subject)
        except BaseException:  # noqa: BLE001 - a mutant that does not load is killed by any usable test
            killed.append(usable[0] if usable else -1)
            continue
        first = -1
        for i in usable:
            if _run(driver(imp, tests[i]["call"], tests[i]["args"]), steps, limits[i]) != original[i]:
                first = i
                break
        killed.append(first)
    return {"original": original, "killed": killed, "usable_tests": len(usable)}


def spectra(files, target, subject, tests, mutants, steps=STEP_LIMIT):
    """Per mutant: failing and passing test counts and, per block probe, how many failing (ef) and passing (ep) tests reach it."""
    def probed(visitor):  # the original and the mutants spend steps on the same probes: those of the target file
        return lambda path, tree: ast.fix_missing_locations(visitor.visit(tree)) if path == target else tree
    original, limits = _originals(files, subject, tests, steps, probed(_Instrument()))
    usable = [i for i, o in enumerate(original) if o[0] != "timeout"]
    lines = files[target].splitlines(keepends=True)
    hits, results, size = set(), [], None
    for start, end, replacement in mutants:
        variant = dict(files)
        variant[target] = "".join(lines[:start - 1]) + replacement + "".join(lines[end:])
        visitor = _Instrument()
        try:
            _STEPS[0] = steps
            imp = load(variant, probed(visitor), hits.add, _spending(hits.add))
            imp(subject)
        except BaseException:  # noqa: BLE001 - a mutant that breaks the import has no spectrum
            results.append(None)
            continue
        size = visitor.count if size is None else size
        if visitor.count != size:
            raise RuntimeError("a mutant changed the block structure")
        driver = MUTATION_DRIVERS[subject]
        ef, ep, failed = [0] * size, [0] * size, 0
        for i in usable:
            hits.clear()
            fail = _run(driver(imp, tests[i]["call"], tests[i]["args"]), steps, limits[i]) != original[i]
            counts = ef if fail else ep
            for probe in hits:
                counts[probe] += 1
            failed += fail
        results.append({"failed": failed, "passed": len(usable) - failed, "ef": ef, "ep": ep})
    return {"blocks": size, "usable_tests": len(usable), "faults": results}
