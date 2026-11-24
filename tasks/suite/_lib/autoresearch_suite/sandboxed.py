"""Entry point for evaluators whose candidate runs in a sandboxed child process.

``runtime.run`` imports the candidate file, so its module-level code runs in the evaluator's
process, where the prepared data and the scoring functions are. An evaluator that then calls
the candidate in a sandbox is still open to a candidate that reads the answers or replaces a
scoring function while it is imported. ``run`` here hands the task function the candidate
*unexecuted*: a module object that carries the file's path and nothing else. The sandbox
modules read the source from that path and execute it in the child only; ``defines`` tells
them which names the source binds.
"""
from __future__ import annotations

import ast
import time
import traceback
import types
from pathlib import Path

from . import runtime
from .runtime import CandidateError


def source_program(program_path, name="candidate_program"):
    """The candidate as a module that has not been executed: it carries the file's path only."""
    path = Path(program_path).resolve()
    if not path.is_file():
        raise CandidateError(f"candidate file not found: {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__candidate_source__ = True
    return module


def defines(module, name):
    """Whether the candidate provides ``name``.

    An imported module is asked for a callable attribute. An unexecuted one (``source_program``) is
    parsed: the answer is whether its source binds the name by a definition, an assignment or an
    import; the call itself happens in the child process.
    """
    if not getattr(module, "__candidate_source__", False):
        return callable(getattr(module, name, None))
    try:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    except (SyntaxError, ValueError) as error:
        raise CandidateError(f"candidate does not parse: {error}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store):
            return True
        if isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == name:
            return True
    return False


def run(program_path, task, *, timeout=None):
    """``runtime.run`` without importing the candidate: ``task`` receives the unexecuted module; never raises."""
    started = time.time()
    try:
        with runtime.Deadline(timeout):
            result = task(source_program(program_path))
    except CandidateError as error:
        result = runtime.failure(error, "maximize")
    except TimeoutError as error:
        result = runtime.failure(error, "maximize")
    except Exception as error:  # noqa: BLE001 - any crash of candidate or evaluator is a failed evaluation
        result = runtime.failure(f"{type(error).__name__}: {error}", "maximize", traceback=traceback.format_exc()[-1500:])
    result.setdefault("elapsed_seconds", round(time.time() - started, 6))
    return result
