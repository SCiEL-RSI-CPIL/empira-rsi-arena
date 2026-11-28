"""Constrained covering arrays: parsing, exact feasibility (cross-checked with Z3), certification and the seed."""
from __future__ import annotations

import importlib.util
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite/_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import software_cit as cit  # noqa: E402
from autoresearch_suite import software_data  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import seed_module, seed_source  # noqa: E402

PREPARED = all((software_data.data_dir() / f"cit/{name}.json").is_file() for name in cit.INSTANCES)
HAVE_Z3 = importlib.util.find_spec("z3") is not None


def tiny():
    # Three parameters (2, 2, 3 levels); 1-based CASA literals: level 1 of p0 is index 1, p1 levels are 3-4, p2 levels 5-7.
    model = "2\n3\n2 2 3\n"
    constraints = "3\n2\n- 1 - 3\n2\n- 1 - 4\n2\n- 2 - 7\n"
    return cit.parse_casa(model, constraints)


class ParsingAndFeasibility(unittest.TestCase):
    def test_one_based_literals_become_parameter_levels(self):
        instance = tiny()
        self.assertEqual(instance["values"], [2, 2, 3])
        self.assertEqual(instance["forbidden"], [[[0, 0], [1, 0]], [[0, 0], [1, 1]], [[0, 1], [2, 2]]])

    def test_propagation_finds_an_infeasible_single_level(self):
        instance = tiny()
        feasibility = cit.Feasibility(instance["values"], instance["forbidden"])
        self.assertFalse(feasibility.satisfiable({0: 0}))  # p0=0 forbids both levels of p1
        self.assertTrue(feasibility.satisfiable({0: 1, 2: 0}))
        minimal = feasibility.minimal_infeasible(2)
        self.assertIn([[0, 0]], minimal)
        self.assertIn([[2, 2]], minimal)  # p0 is forced to level 1, which forbids level 2 of p2
        self.assertNotIn([[0, 0], [1, 0]], minimal)  # not minimal: contains the infeasible single level
        self.assertEqual(len(minimal), 2)

    def test_certification_accepts_complete_arrays_and_rejects_defects(self):
        instance = tiny()
        instance["minimal_infeasible"] = cit.Feasibility(instance["values"], instance["forbidden"]).minimal_infeasible(2)
        rows = [[1, a, b] for a in range(2) for b in range(2)]
        self.assertEqual(cit.check_array(instance, rows)[0], 4)
        with self.assertRaisesRegex(CandidateError, "not covered"):
            cit.check_array(instance, rows[:-1])
        with self.assertRaisesRegex(CandidateError, "forbidden"):
            cit.check_array(instance, rows + [[0, 0, 0]])
        for bad in ([[1, 0, True]], [[1, 0, 0.0]], [[1, 0]], [[1, 0, 3]], [[1, 0, 10 ** 30]], [], "rows"):
            with self.subTest(bad=bad), self.assertRaises(CandidateError):
                cit.check_array(instance, bad)


@unittest.skipUnless(PREPARED, "run scripts/prepare_software.py --only cit")
class PreparedInstances(unittest.TestCase):
    def test_counting_lower_bounds_and_sizes(self):
        expected = {"apache-2way": (172, 30), "gcc-2way": (199, 9), "spinv-2way": (55, 16), "spins-3way": (18, 64), "healthcare4-2way": (35, 42)}
        for name, (k, bound) in expected.items():
            instance, masks = cit.load_instance(name)
            self.assertEqual(len(instance["values"]), k)
            self.assertEqual(max(int(m.sum()) for m in masks.values()), bound, name)

    @unittest.skipUnless(HAVE_Z3, "z3-solver not installed")
    def test_required_combinations_agree_with_z3(self):
        import z3
        for name in cit.INSTANCES:
            instance, masks = cit.load_instance(name)
            values, t = instance["values"], instance["strength"]
            x = [[z3.Bool(f"x{p}_{v}") for v in range(n)] for p, n in enumerate(values)]
            solver = z3.Solver()
            for row in x:
                solver.add(z3.PbEq([(b, 1) for b in row], 1))
            for combination in instance["forbidden"]:
                solver.add(z3.Not(z3.And([x[p][v] for p, v in combination])))
            constrained = set(p for f in instance["forbidden"] for p, _ in f)
            checked = 0
            for params, mask in masks.items():
                if not constrained.intersection(params):
                    self.assertTrue(mask.all(), (name, params))  # untouched by any constraint
                    continue
                for levels in itertools.product(*(range(values[p]) for p in params)):
                    expected = solver.check([x[p][v] for p, v in zip(params, levels)]) == z3.sat
                    self.assertEqual(bool(mask[levels]), expected, (name, params, levels))
                    checked += 1
            self.assertGreater(checked, 0)

    def test_seed_returns_certified_arrays(self):
        for name in cit.INSTANCES:
            instance, masks = cit.load_instance(name)
            rows = seed_module(f"science-software-cit-{name}").generate(cit.public_model(instance))  # the instance's own package
            size, details = cit.check_array(instance, rows, masks)
            self.assertGreaterEqual(size, max(int(m.sum()) for m in masks.values()))

    def test_scoring_runs_the_candidate_in_the_sandbox(self):
        program = Path(tempfile.mkdtemp()) / "candidate.py"
        program.write_text(seed_source("science-software-cit-healthcare4-2way"))
        spec = importlib.util.spec_from_file_location("cit_candidate", program)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        score, details = cit.cit_score(module, "healthcare4-2way", cpu_seconds=30, reference=46, lower_bound=42)
        self.assertEqual(score, details["rows"])
        program.write_text("def generate(model):\n    return [[0] * len(model['values'])]\n")
        spec.loader.exec_module(module)
        with self.assertRaisesRegex(CandidateError, "not covered|forbidden"):
            cit.cit_score(module, "healthcare4-2way", cpu_seconds=30)


if __name__ == "__main__":
    unittest.main()
