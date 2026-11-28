"""PDR / CSPLib reformulation tasks: generators, checkers and the CP-SAT harness (package or_csp)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "suite" / "_lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import or_csp_csplib as C  # noqa: E402
from autoresearch_suite import or_csp_cpsat as H  # noqa: E402
from autoresearch_suite.or_csp_baselines import BASELINES  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

HAVE_CPMPY = all(importlib.util.find_spec(m) for m in ("cpmpy", "ortools"))
# OR-Tools 9.14 and highspy 1.15 ship different builds of libhighs.1.dylib and cannot share one process: whichever
# loads second binds to the other's library (an import error or a segfault). When another test has already imported
# highspy (through cvxpy), the harness tests run in a fresh interpreter instead (HarnessInFreshProcess).
UPGRADES = json.loads((ROOT / "catalog" / "upgrades" / "or_csp.json").read_text())
# sha256 prefixes of the JSON of each task's generated instance set; any generator change must update these on purpose
PINNED = {
    "pdr-car-sequencing-csplib-1": "66f9c8a806e4bce8",
    "pdr-vessel-loading-csplib-8": "2ac8a057de66c313",
    "pdr-balanced-academic-curriculum-csplib-30": "587c9e1becaf32a4",
    "pdr-tail-assignment-csplib-115": "51a9215389322b7d",
}


def candidate_module(source):
    """Write a candidate file and return a module-like object exposing build_model and __file__."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    handle.write(textwrap.dedent(source))
    handle.close()
    namespace = {"__name__": "candidate"}
    exec(compile(Path(handle.name).read_text(), handle.name, "exec"), namespace)
    module = types.SimpleNamespace(**namespace)
    module.__file__ = handle.name
    return module


class GeneratorTests(unittest.TestCase):
    def test_rng_stream_is_fixed(self):
        rng = H.Rng(1)
        self.assertEqual([rng.next64() for _ in range(3)], [3047811371739217279, 18215591310788271736, 10655362870870243545])

    def test_instance_sets_are_pinned(self):
        tasks = {tid: spec for tid, spec in UPGRADES["tasks"].items() if spec["family"].startswith("csplib_reformulation_")}
        self.assertEqual(set(tasks), set(PINNED))
        for tid, spec in tasks.items():
            with self.subTest(task=tid):
                params = spec["params"]
                generate = C.PROBLEMS[params["problem"]][0]
                digest = hashlib.sha256()
                for instance in params["instances"]:
                    digest.update(json.dumps(generate(**instance), sort_keys=True).encode())
                self.assertEqual(digest.hexdigest()[:16], PINNED[tid])
                self.assertEqual(len(params["instances"]), 8)
                self.assertEqual(spec["family"], "csplib_reformulation_" + params["problem"])

    def test_generated_data_never_contains_the_plant(self):
        for problem, spec in (("car_sequencing", {"seed": 1, "n_cars": 30, "n_pool": 6, "bias": 2.0}),
                              ("bacp", {"seed": 1, "n_courses": 20, "n_periods": 4, "density": 0.1, "load_slack": 0, "count_slack": 0}),
                              ("tail_assignment", {"seed": 1, "n_planes": 3, "min_legs": 2, "max_legs": 4, "n_airports": 3, "restrict": 0.3})):
            with self.subTest(problem=problem):
                data = C.PROBLEMS[problem][0](**spec)
                self.assertFalse({"sequence", "period", "routes", "solution", "plant"} & set(data))


class CheckerTests(unittest.TestCase):
    def test_covering_array(self):
        data = C.generate_covering_array(2, 3, 2, 4)
        good = {"matrix": [[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]]}
        C.check_covering_array(data, good)
        with self.assertRaises(CandidateError):
            C.check_covering_array(data, {"matrix": [[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0]]})
        with self.assertRaises(CandidateError):
            C.check_covering_array(data, {"matrix": [[0, 0, 1, 2], [0, 1, 0, 1], [0, 1, 1, 0]]})
        with self.assertRaises(CandidateError):
            C.check_covering_array(data, {"matrix": good["matrix"], "extra": 1})

    def test_bibd_fano_plane(self):
        blocks = [(0, 1, 2), (0, 3, 4), (0, 5, 6), (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)]
        matrix = [[int(o in block) for block in blocks] for o in range(7)]
        data = C.generate_bibd(7, 7, 3, 3, 1)
        C.check_bibd(data, {"matrix": matrix})
        matrix[0][0], matrix[0][1] = 0, 1
        with self.assertRaises(CandidateError):
            C.check_bibd(data, {"matrix": matrix})
        with self.assertRaises(ValueError):
            C.generate_bibd(7, 7, 3, 3, 2)

    def test_social_golfers(self):
        data = C.generate_social_golfers(2, 3, 3)
        C.check_social_golfers(data, {"schedule": [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]]})
        with self.assertRaises(CandidateError):  # golfers 0 and 1 meet twice
            C.check_social_golfers(data, {"schedule": [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 1, 1, 2, 2, 0, 1, 2]]})

    def test_car_sequencing_windows(self):
        data = {"n_cars": 4, "n_classes": 2, "n_options": 1, "quantity": [2, 2], "maxcars": [1], "blksize": [2], "usage": [[1], [0]]}
        C.check_car_sequencing(data, {"sequence": [0, 1, 0, 1]})
        with self.assertRaises(CandidateError):
            C.check_car_sequencing(data, {"sequence": [0, 0, 1, 1]})
        with self.assertRaises(CandidateError):
            C.check_car_sequencing(data, {"sequence": [0, 1, 1, 1]})

    def test_vessel_loading_separation(self):
        data = {"deck_width": 4, "deck_length": 2, "n_containers": 2, "n_classes": 2, "width": [1, 1], "length": [2, 2],
                "container_class": [0, 1], "separation": [[0, 2], [2, 0]]}
        C.check_vessel_loading(data, {"west": [0, 3], "east": [1, 4], "north": [0, 0], "south": [2, 2]})
        with self.assertRaises(CandidateError):  # gap 1 < separation 2
            C.check_vessel_loading(data, {"west": [0, 2], "east": [1, 3], "north": [0, 0], "south": [2, 2]})
        with self.assertRaises(CandidateError):  # wrong extent
            C.check_vessel_loading(data, {"west": [0, 2], "east": [2, 3], "north": [0, 0], "south": [2, 2]})

    def test_bacp_codes_and_tail(self):
        bacp = {"n_courses": 3, "n_periods": 2, "course_load": [2, 2, 3], "prerequisites": [[0, 2]], "load_per_period_lb": 2,
                "load_per_period_ub": 4, "courses_per_period_lb": 1, "courses_per_period_ub": 2}
        C.check_bacp(bacp, {"period": [0, 0, 1]})
        with self.assertRaises(CandidateError):
            C.check_bacp(bacp, {"period": [1, 0, 0]})
        codes = C.generate_error_correcting_codes(4, 2, 2, 3, "lee")
        C.check_error_correcting_codes(codes, {"codewords": [[0, 0], [2, 1]]})
        with self.assertRaises(CandidateError):
            C.check_error_correcting_codes(codes, {"codewords": [[0, 0], [1, 1]]})
        tail = {"n_flights": 3, "n_planes": 2, "maint_freq": 1, "network": [[0, 1]], "can_fly": [[0, 1, 2], [2]],
                "start": [[0], [2]], "maint": [1, 2]}
        C.check_tail_assignment(tail, {"routes": [[0, 1], [2]]})
        for bad in ({"routes": [[0], [2]]}, {"routes": [[1, 0], [2]]}, {"routes": [[0, 1], []]}, {"routes": [[0, 1, 2], [2]]}):
            with self.assertRaises(CandidateError):
                C.check_tail_assignment(tail, bad)


@unittest.skipUnless(HAVE_CPMPY, "cpmpy and ortools are required")
class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "highspy" in sys.modules:
            raise unittest.SkipTest("highspy is loaded in this process; HarnessInFreshProcess runs these tests in a fresh interpreter")

    TINY = {
        "car_sequencing": {"seed": 3, "n_cars": 20, "n_pool": 6, "bias": 2.0},
        "vessel_loading": {"seed": 3, "deck_width": 8, "deck_length": 8, "n_containers": 5, "n_classes": 2, "fill": 0.8, "sep_max": 1},
        "social_golfers": {"weeks": 3, "groups": 3, "group_size": 3},
        "bibd": {"v": 7, "b": 7, "r": 3, "k": 3, "lam": 1},
        "bacp": {"seed": 3, "n_courses": 16, "n_periods": 4, "density": 0.1, "load_slack": 0, "count_slack": 0},
        "error_correcting_codes": {"alphabet_size": 4, "length": 4, "num_codewords": 4, "min_dist": 3, "distance": "lee"},
        "covering_array": {"t": 2, "k": 4, "g": 2, "b": 6},
        "tail_assignment": {"seed": 3, "n_planes": 2, "min_legs": 2, "max_legs": 3, "n_airports": 3, "restrict": 0.3},
    }

    def params(self, problem):
        return {"problem": problem, "instances": [self.TINY[problem]], "cap_factor": 3.0, "cap_floor": 5.0, "baseline_cap": 30.0}

    def test_baseline_as_candidate_is_valid_and_near_one(self):
        for problem in C.PROBLEMS:
            with self.subTest(problem=problem):
                module = candidate_module(BASELINES[problem])
                value, details = C.score(module, self.params(problem))
                self.assertEqual(details["instances_used"], 1)
                self.assertEqual(details["timeouts"], 0)
                self.assertGreater(value, 0.2)
                self.assertLess(value, 5.0)

    def test_wrong_answers_and_contract_breaches_are_invalid(self):
        bad = {
            "wrong answer": """
                import cpmpy as cp
                def build_model(data):
                    x = cp.intvar(0, 1, shape=2, name="x")
                    return cp.Model(x[0] <= x[1]), lambda: {"matrix": [[0] * data["b"] for _ in range(data["k"])]}
            """,
            "infeasible claim": """
                import cpmpy as cp
                def build_model(data):
                    x = cp.intvar(0, 1, shape=2, name="x")
                    return cp.Model(x[0] + x[1] >= 3), lambda: {}
            """,
            "objective": """
                import cpmpy as cp
                def build_model(data):
                    x = cp.intvar(0, 3, shape=2, name="x")
                    model = cp.Model(x[0] <= x[1])
                    model.maximize(x[0])
                    return model, lambda: {}
            """,
            "forbidden import": """
                import os
                import cpmpy as cp
                def build_model(data):
                    return cp.Model(), lambda: {}
            """,
            "dunder access": """
                import cpmpy as cp
                def build_model(data):
                    return cp.Model(), (lambda: {}).__globals__
            """,
            "bad return": """
                def build_model(data):
                    return None
            """,
            "non-json answer": """
                import cpmpy as cp
                def build_model(data):
                    x = cp.intvar(0, 1, shape=2, name="x")
                    return cp.Model(x[0] <= x[1]), lambda: {"matrix": object()}
            """,
        }
        for label, source in bad.items():
            with self.subTest(case=label):
                with self.assertRaises(CandidateError):
                    C.score(candidate_module(source), self.params("covering_array"))

    def test_a_timeout_counts_twice_the_cap(self):
        params = self.params("covering_array")
        params["cap_floor"], params["cap_factor"] = 1e-4, 1e-4  # no model can be built within 0.1 ms
        value, details = C.score(candidate_module(BASELINES["covering_array"]), params)
        row = details["per_instance"][0]
        self.assertEqual(row["candidate_status"], "TIMEOUT")
        self.assertEqual(details["timeouts"], 1)
        self.assertAlmostEqual(row["candidate_seconds"], 2 * row["cap_seconds"], places=6)
        self.assertAlmostEqual(value, row["baseline_seconds"] / row["candidate_seconds"], places=3)


@unittest.skipUnless(HAVE_CPMPY, "cpmpy and ortools are required")
class HarnessInFreshProcess(unittest.TestCase):
    def test_harness_tests_pass_in_a_fresh_interpreter(self):
        if "highspy" not in sys.modules:
            self.skipTest("HarnessTests run in this process")
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "tests.test_or_csp_csplib.HarnessTests"], cwd=ROOT,
                                capture_output=True, text=True, timeout=1800)  # -B: nothing is written into the library
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        self.assertNotIn("skipped", result.stderr.splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
