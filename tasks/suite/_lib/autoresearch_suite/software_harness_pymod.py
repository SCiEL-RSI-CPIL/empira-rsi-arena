"""Sandbox-side harness for the Python-module subjects (standard library only, executed as job source).

Subject modules are served from source texts by an in-memory import finder
under the private top-level package ``artask_subject`` (so the interpreter's
own copies are untouched). ``coverage`` instruments chosen files with block
probes and replays fuzzing inputs; ``mutation_analysis`` replays test calls on
the original module and on each mutant and reports which mutants are killed.
The per-subject drivers below turn JSON inputs into calls; they are part of
the trusted harness, not of the mutated or instrumented code.
"""
import ast
import hashlib
import importlib.abc
import importlib.util
import signal
import sys

PREFIX = "artask_subject"
PROBE = "__artask_probe__"


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves artask_subject.<dotted> from {relative path: source}; directories become packages."""

    def __init__(self, files, transform=None, probe=None):
        self.modules, self.probe = {}, probe or _noop
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
        module.__dict__[PROBE] = self.probe
        exec(compile(tree, f"<{path or module.__name__}>", "exec"), module.__dict__)


def _noop(_):
    pass


def load(files, transform=None, probe=None):
    """Fresh import of a subject; returns the import function for dotted names below the prefix."""
    for name in [n for n in sys.modules if n == PREFIX or n.startswith(PREFIX + ".")]:
        del sys.modules[name]
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _Finder)]
    sys.meta_path.insert(0, _Finder(files, transform, probe))
    import importlib
    return lambda dotted: importlib.import_module(PREFIX + "." + dotted)


class _Instrument(ast.NodeTransformer):
    """Inserts a numbered probe at the start of every block inside functions (bodies, branches, handlers, loop exits)."""

    def __init__(self):
        self.count, self.depth, self.blocks = 0, 0, {}

    def _probe(self):
        self.count += 1
        call = ast.Call(func=ast.Name(id=PROBE, ctx=ast.Load()), args=[ast.Constant(self.count - 1)], keywords=[])
        return ast.Expr(value=call)

    def _block(self, body, keep_docstring=False):
        start = 1 if keep_docstring and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str) else 0
        probe = self._probe()
        self.blocks[self.count - 1] = [(s.lineno, s.end_lineno) for s in body]  # statement spans of this block
        return body[:start] + [probe] + body[start:]

    def visit_FunctionDef(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
        node.body = self._block(node.body, keep_docstring=True)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def _branches(self, node, fields):
        self.generic_visit(node)
        if self.depth:
            for field in fields:
                setattr(node, field, self._block(getattr(node, field)))
        return node

    def visit_If(self, node):
        return self._branches(node, ("body", "orelse"))

    def visit_For(self, node):
        return self._branches(node, ("body", "orelse"))

    visit_While = visit_AsyncFor = visit_For

    def visit_Try(self, node):
        return self._branches(node, ("body", "orelse") + (("finalbody",) if node.finalbody else ()))

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


def _run(function, seconds):
    signal.signal(signal.SIGPROF, _alarm)
    signal.setitimer(signal.ITIMER_PROF, seconds)
    try:
        value = function()
        if hasattr(value, "__next__"):
            value = [item for _, item in zip(range(10001), value)]
        return ["ok", hashlib.sha256(repr(value).encode("utf-8", "surrogatepass")).hexdigest()[:24]]
    except _Timeout:
        return ["timeout"]
    except RecursionError:
        return ["raise", "RecursionError"]
    except BaseException as error:  # noqa: BLE001 - the exception type is the observable outcome
        return ["raise", type(error).__name__]
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)


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


def coverage(files, instrumented, subject, inputs, per_input_seconds=0.5):
    """Sorted probe ids hit by the inputs, the probe count, and the number of inputs that timed out."""
    hits, counts = set(), {}

    def transform(path, tree):
        if path not in instrumented:
            return tree
        visitor = _Instrument()
        visitor.count = sum(counts.values())
        tree = ast.fix_missing_locations(visitor.visit(tree))
        counts[path] = visitor.count - sum(counts.values())
        return tree
    imp = load(files, transform, hits.add)
    target = FUZZ_DRIVERS[subject](imp)
    timeouts = 0
    for text in inputs:
        timeouts += _run(lambda: target(text), per_input_seconds) == ["timeout"]
    return {"hits": sorted(hits), "probes": sum(counts.values()), "timeouts": timeouts}


def outcomes(files, subject, tests, per_test_seconds=0.2):
    imp = load(files)
    driver = MUTATION_DRIVERS[subject]
    return [_run(driver(imp, t["call"], t["args"]), per_test_seconds) for t in tests]


def mutation_analysis(files, target, subject, tests, mutants, per_test_seconds=0.2):
    """Original outcomes plus, per mutant, the index of the first killing test (or -1)."""
    original = outcomes(files, subject, tests, per_test_seconds)
    usable = [i for i, o in enumerate(original) if o[0] != "timeout"]
    lines = files[target].splitlines(keepends=True)
    killed = []
    for start, end, replacement in mutants:
        variant = dict(files)
        variant[target] = "".join(lines[:start - 1]) + replacement + "".join(lines[end:])
        try:
            imp = load(variant)
            driver = MUTATION_DRIVERS[subject]
            imp(subject)
        except BaseException:  # noqa: BLE001 - a mutant that does not load is killed by any usable test
            killed.append(usable[0] if usable else -1)
            continue
        first = -1
        for i in usable:
            if _run(driver(imp, tests[i]["call"], tests[i]["args"]), per_test_seconds) != original[i]:
                first = i
                break
        killed.append(first)
    return {"original": original, "killed": killed, "usable_tests": len(usable)}


def spectra(files, target, subject, tests, mutants, per_test_seconds=0.2):
    """Per mutant: failing and passing test counts and, per block probe, how many failing (ef) and passing (ep) tests reach it."""
    original = outcomes(files, subject, tests, per_test_seconds)
    usable = [i for i, o in enumerate(original) if o[0] != "timeout"]
    lines = files[target].splitlines(keepends=True)
    hits, results, size = set(), [], None
    for start, end, replacement in mutants:
        variant = dict(files)
        variant[target] = "".join(lines[:start - 1]) + replacement + "".join(lines[end:])
        visitor = _Instrument()

        def transform(path, tree, visitor=visitor):
            return ast.fix_missing_locations(visitor.visit(tree)) if path == target else tree
        try:
            imp = load(variant, transform, hits.add)
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
            fail = _run(driver(imp, tests[i]["call"], tests[i]["args"]), per_test_seconds) != original[i]
            counts = ef if fail else ep
            for probe in hits:
                counts[probe] += 1
            failed += fail
        results.append({"failed": failed, "passed": len(usable) - failed, "ef": ef, "ep": ep})
    return {"blocks": size, "usable_tests": len(usable), "faults": results}
