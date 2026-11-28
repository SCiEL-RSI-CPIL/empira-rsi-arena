"""Regression test for the CSPLib stored-answer defect: a model that returns answers stored for the published
seeds scored a speedup of hundreds on the fixed instance set; the seeded CSPLib tasks now score held-out draws."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "suite" / "_lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import or_csp_csplib as C  # noqa: E402
from autoresearch_suite import sandboxed  # noqa: E402

HAVE_CPMPY = all(importlib.util.find_spec(m) for m in ("cpmpy", "ortools"))


def wrapper(task):
    spec = importlib.util.spec_from_file_location(f"evaluator_{task.replace('-', '_')}", ROOT / "suite" / task / "evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TemporaryFiles(unittest.TestCase):
    def file(self, name, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / name
        path.write_text(text)
        return path


@unittest.skipUnless(HAVE_CPMPY, "cpmpy and ortools are required")
class CsplibHeldOutTests(TemporaryFiles):
    """A model that returns answers stored for the published seeds no longer scores a speedup."""

    TASK = "pdr-tail-assignment-csplib-115"

    def test_stored_answers_for_the_published_seeds_gain_nothing_on_held_out_instances(self):
        params = dict(wrapper(self.TASK).P)
        params["instances"] = params["instances"][:2]
        generate, check = C.PROBLEMS[params["problem"]]
        seed_source = (ROOT / "suite" / self.TASK / "initial_program.py").read_text()
        namespace = {}
        exec(compile(seed_source, "<seed>", "exec"), namespace)  # noqa: S102 - the package's own seed
        stored = {}
        for spec in params["instances"]:
            data = generate(**spec)
            model, extractor = namespace["build_model"](data)
            self.assertTrue(model.solve(solver="ortools"))
            check(data, extractor())
            stored[repr(sorted(data.items()))] = extractor()
        exploit = seed_source.replace("def build_model(", "def honest_model(") + f"""
STORED = {stored!r}


def build_model(data):
    key = repr(sorted(data.items()))
    if key in STORED:
        flag = cp.boolvar(name="flag")
        return cp.Model(flag == 1), (lambda: STORED[key])
    return honest_model(data)
"""
        program = sandboxed.source_program(self.file("candidate.py", exploit))
        published = dict(params)
        published.pop("held_out")
        value, _ = C.score(program, published)
        self.assertGreater(value, 20.0)  # the fixed instance set: a stored answer is a speedup of hundreds
        value, details = C.score(program, params)
        self.assertLess(value, 3.0)
        drawn = [row["instance"]["seed"] for row in details["per_instance"]]
        self.assertFalse(set(drawn) & {spec["seed"] for spec in params["instances"]})
        _, again = C.score(program, params)
        self.assertNotEqual(drawn, [row["instance"]["seed"] for row in again["per_instance"]])


if __name__ == "__main__":
    unittest.main()
