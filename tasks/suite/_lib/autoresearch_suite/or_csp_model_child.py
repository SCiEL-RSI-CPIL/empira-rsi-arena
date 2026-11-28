"""Child side of the CSPLib reformulation harness: builds one model and extracts one answer in the sandbox.

``or_csp_modelrun`` runs this file with ``software_sandbox.execute`` (a fresh ``software_worker.py`` child per
instance and model), so the model code of a candidate never runs in the evaluator's process. The file is
self-contained (standard library, CPMpy and OR-Tools only): the child cannot import the evaluator package.

Protocol, through the sandbox's oracle channel (the evaluator answers each message and reads this process's CPU
clock from outside at every one; nothing this process says about time is used, because once the model source runs
it can rewrite this module):

1. ``{"phase": "ready"}`` after the imports and a warm-up, before any statement of the model source runs;
2. ``{"phase": "model", "text": ...}`` after the module's top-level code, ``build_model(data)``, the CPMpy ->
   OR-Tools transformation and the export of the ``CpModelProto`` in protobuf text format; the reply carries the
   solver's full solution vector, or ``stop`` when there is nothing to extract;
3. ``{"phase": "answer", "answer": ...}`` after the solver values were written into the CPMpy variables, the
   extractor ran and its answer was converted to JSON-native values.
"""
import base64
import sys
import types
import zlib

COMPRESS_ABOVE = 32 << 20  # the sandbox frames the child's messages up to 64 MiB; larger model texts travel compressed


class ModelError(Exception):
    """A failure of the model code; the evaluator reports it as the model's (the candidate's) error."""


def _stub_pandas():
    """``ortools.sat.python.cp_model`` imports pandas for its pandas helpers only (CPMpy does not use them), and
    pandas imports ctypes, which the sandbox refuses; a placeholder module stands in for pandas."""
    if "pandas" not in sys.modules:
        placeholder = types.ModuleType("pandas")
        for name in ("Index", "Series", "DataFrame"):
            setattr(placeholder, name, type(name, (), {}))
        sys.modules["pandas"] = placeholder


def _warm_up(cp, solver_class):
    """Load what the first transformation would otherwise import lazily, before the timed part starts."""
    x = cp.intvar(0, 3, shape=3, name="warm")
    b = cp.boolvar(name="warm_b")
    model = cp.Model(cp.AllDifferent(x), cp.sum(x) >= 3, b.implies(x[0] < x[1]), x[2] == cp.max(x))
    solver = solver_class(model)
    solver.solver_vars(list(solver.user_vars))
    str(solver.ort_model.Proto())


def plain(value):
    """JSON-native values of the exact built-in types (the evaluator converts the answer once more, its own way)."""
    if isinstance(value, dict):
        return {plain_key(k): plain(v) for k, v in dict.items(value)}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if value is None or type(value) in (bool, int, float, str):
        return value
    for base, convert in ((int, int.__int__), (float, float.__float__), (str, str.__str__)):
        if isinstance(value, base):
            return convert(value)
    raise ModelError(f"extractor returned a non-JSON value of type {type(value).__name__}")


def plain_key(key):
    key = plain(key)
    return key if type(key) is str else str(key)


def _build(source, filename, data, cp, solver_class, boolean_class):
    module = types.ModuleType("candidate")
    module.__file__ = filename
    try:
        exec(compile(source, filename, "exec"), module.__dict__)  # noqa: S102 - this is the sandboxed child
    except Exception as error:  # noqa: BLE001
        raise ModelError(f"the model module failed while loading: {type(error).__name__}: {error}") from None
    builder = module.__dict__.get("build_model")
    if not callable(builder):
        raise ModelError("candidate must define build_model(data) -> (model, extractor)")
    try:
        built = builder(data)
    except Exception as error:  # noqa: BLE001
        raise ModelError(f"build_model failed: {type(error).__name__}: {error}") from None
    if not (isinstance(built, tuple) and len(built) == 2):
        raise ModelError("build_model must return a (model, extractor) tuple")
    model, extractor = built
    if not isinstance(model, cp.Model) or not callable(extractor):
        raise ModelError("build_model must return (cpmpy.Model, callable extractor)")
    if model.objective_ is not None:
        raise ModelError("models must be satisfaction models (no objective), like the baseline")
    try:
        solver = solver_class(model)
        user_vars = list(solver.user_vars)
        solver.solver_vars(user_vars)
        index = [(var, solver.solver_var(var).Index(), isinstance(var, boolean_class)) for var in user_vars]
    except Exception as error:  # noqa: BLE001
        raise ModelError(f"CPMpy could not transform the model for OR-Tools: {type(error).__name__}: {error}") from None
    return solver, extractor, index


def build_and_extract(source, filename, data, oracle):
    """Entry point (see the module docstring); returns nothing, everything travels through ``oracle``."""
    _stub_pandas()
    import cpmpy as cp
    from cpmpy.expressions.variables import _BoolVarImpl
    from cpmpy.solvers.ortools import CPM_ortools
    _warm_up(cp, CPM_ortools)
    oracle({"phase": "ready"})
    solver, extractor, index = _build(source, filename, data, cp, CPM_ortools, _BoolVarImpl)
    text = str(solver.ort_model.Proto())
    message = {"phase": "model", "text": text}
    if len(text) > COMPRESS_ABOVE:
        message = {"phase": "model", "zlib": base64.b64encode(zlib.compress(text.encode(), 1)).decode("ascii")}
    del text
    reply = oracle(message)
    if not isinstance(reply, dict) or "solution" not in reply:
        return None
    solution = reply["solution"]
    for var, position, boolean in index:
        var._value = bool(solution[position]) if boolean else int(solution[position])
    try:
        answer = plain(extractor())
    except ModelError:
        raise
    except Exception as error:  # noqa: BLE001
        raise ModelError(f"extractor failed: {type(error).__name__}: {error}") from None
    oracle({"phase": "answer", "answer": answer})
    return None
