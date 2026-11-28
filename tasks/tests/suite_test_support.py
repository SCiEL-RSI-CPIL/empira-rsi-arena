"""Shared by the tests of the added task collections: package files, prepared data and the slow-test gate.

The source repository's tests read seeds, parameters and family registrations from its generator
modules, which are not part of this library. The same values are read here from the generated
packages (suite/<id>/) and from the catalog, so the tests check the files that are shipped.
"""
from __future__ import annotations

import ast
import atexit
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import textwrap
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "suite/_lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "suite/_lib"))

# A test that needs more than about 20 seconds runs only on request.
SLOW = os.environ.get("ARENA_SLOW_TESTS") == "1"
slow = unittest.skipUnless(SLOW, "slow test; set ARENA_SLOW_TESTS=1 to run it")
BLOCK_START, BLOCK_END = "# EVOLVE-BLOCK-START\n", "# EVOLVE-BLOCK-END"


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def seed_source(task_id):
    """The seed of a package: the text between the markers of its initial_program.py."""
    text = (ROOT / "suite" / task_id / "initial_program.py").read_text(encoding="utf-8")
    return text[text.index(BLOCK_START) + len(BLOCK_START):text.index(BLOCK_END)]


def seed_module(task_id):
    """A package's initial_program.py as a module, executed from source so that no bytecode is written."""
    path = ROOT / "suite" / task_id / "initial_program.py"
    module = types.ModuleType("seed_" + task_id.replace("-", "_"))
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)  # noqa: S102 - the task's own seed
    return module


def evaluator(task_id):
    """DIRECTION, TIMEOUT, P and the body of task(module) of a package's evaluator.py, read without importing it."""
    text = (ROOT / "suite" / task_id / "evaluator.py").read_text(encoding="utf-8")
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("DIRECTION", "TIMEOUT", "P"):
            found[node.targets[0].id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef) and node.name == "task":
            lines = text.splitlines()[node.body[0].lineno - 1:node.end_lineno]
            found["body"] = textwrap.dedent("\n".join(lines))
    return found


def package(task_id):
    """A package in the terms of the family registration it was generated from."""
    problem, constants = load(f"suite/{task_id}/problem.json"), evaluator(task_id)
    return {"id": task_id, "family": problem["family"], "protocol": problem["protocol"], "contract": problem["contract"],
            "direction": constants["DIRECTION"], "timeout": constants["TIMEOUT"], "parameters": constants["P"],
            "body": constants["body"], "seed": seed_source(task_id)}


def records(collection):
    """The catalog records of a collection, in the order of catalog/science/<collection>.json."""
    return load(f"catalog/science/{collection}.json")


def registered(collection):
    """{family: package}: one package for each family of a collection, the first in catalog order.

    The family name is the one the package states, so a test that looks up a catalog record's family
    compares the catalog with the package.
    """
    found = {}
    for record in records(collection):
        spec = package(record["id"])
        found.setdefault(spec["family"], spec)
    return found


def specifications(collection):
    """A collection's packages with the fields of the generator's specification list.

    key is the family name without its collection prefix, function the verifier the evaluator calls,
    contribution and sota the reviewed label and reference value of the catalog record.
    """
    specs = []
    for record in records(collection):
        spec = package(record["id"])
        spec["key"] = spec["family"].removeprefix(f"science_{collection}_")
        spec["function"] = re.search(rf"\bscience_{collection}\.(\w+)\(", spec["body"]).group(1)
        spec["contribution"] = record["labels"]["scientific_contribution"]
        spec["sota"] = record["state_of_the_art"]
        specs.append(spec)
    return specs


def evidence(collection):
    """{task id: science_relevance} of a collection's tasks in catalog/tasks.json."""
    return {task["id"]: task["science_relevance"] for task in load("catalog/tasks.json")
            if f"science-{collection}" in task["paper_ids"] and "science_relevance" in task}


def have_data(*names):
    """True when the prepared finance tables exist (python3 scripts/prepare_finance.py)."""
    from autoresearch_suite import finance_data
    return all(finance_data.path(name).is_file() for name in names)


def module_from(source):
    """A candidate module object backed by a temporary file, as the sandboxed evaluators read it."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    handle.write(source)
    handle.close()
    atexit.register(lambda: Path(handle.name).unlink(missing_ok=True))
    namespace = {}
    exec(compile(source, handle.name, "exec"), namespace)  # noqa: S102 - test candidates and the tasks' own seeds
    return types.SimpleNamespace(__file__=handle.name, **{k: v for k, v in namespace.items() if not k.startswith("__")})


def source_file(test, relative):
    """A file the source repository's test reads; the test is skipped when this library does not carry it."""
    path = ROOT / relative
    if not path.is_file():
        test.skipTest(f"{relative} is a file of the source repository that is not part of this library")
    return path


def needs(package):
    """Skip a test or a test class when an optional package is not installed."""
    return unittest.skipUnless(importlib.util.find_spec(package), f"{package} is not installed")
