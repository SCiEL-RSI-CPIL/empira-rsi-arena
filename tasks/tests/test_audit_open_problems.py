"""Open-problem instance packages (suite/_lib/arnew): verifiers against independent implementations, and the audit's fixes."""
from __future__ import annotations

import ast
import contextlib
import gzip
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/new_open_problems"
sys.path[:0] = [str(ROOT / "suite/_lib"), str(SCRIPTS)]
from arnew import common, cvrp, dnacode, epfl, jssp, maxcut, miplib, qap, snake, sortnet, sphere, thomson
from arnew.common import CandidateError
import registry_cvrp
import registry_miplib
import registry_misc


def rejected(function, *arguments):
    try:
        function(*arguments)
    except CandidateError:
        return True
    return False


def evaluator_parameters(task_id):
    """The constant P of a package's evaluator.py."""
    tree = ast.parse((ROOT / "suite" / task_id / "evaluator.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "P":
            return ast.literal_eval(node.value)
    raise AssertionError(f"{task_id}: evaluator.py has no P")


def packages(prefix):
    return sorted(path.name for path in (ROOT / "suite").glob(prefix + "*") if (path / "evaluator.py").is_file())


class PointsOnSpheres(unittest.TestCase):
    """Coordinates above about 1e154 overflow in the norm; the rows used to become zero vectors."""

    def test_huge_rows_are_scaled_not_zeroed(self):
        rng = np.random.default_rng(1)
        directions = rng.normal(size=(40, 4))
        expected = sphere.check(directions, 4, 40)[0]
        self.assertLess(expected, 60.0)
        with np.errstate(over="ignore"):
            self.assertAlmostEqual(sphere.check(directions * 1e200, 4, 40)[0], expected, places=9)
            # 40 copies of one direction: the minimum angle is 0, it was reported as 90 degrees
            self.assertEqual(sphere.check(np.full((40, 4), 1e200), 4, 40)[0], 0.0)
            self.assertEqual(sphere.check(np.full((40, 4), 1e308), 4, 40)[0], 0.0)

    def test_thomson_point_with_huge_coordinates_is_on_the_sphere(self):
        rng = np.random.default_rng(2)
        points = rng.normal(size=(10, 3))
        expected = thomson.check(points, 10)[0]
        scaled = points.copy()
        scaled[0] *= 1e200
        with np.errstate(over="ignore"):
            self.assertAlmostEqual(thomson.check(scaled, 10)[0], expected, places=9)

    def test_ordinary_rows_are_unchanged(self):
        rng = np.random.default_rng(3)
        points = rng.normal(size=(30, 5)) * rng.uniform(1e-3, 1e3, size=(30, 1))
        self.assertTrue(np.array_equal(common.unit_rows(points), points / np.linalg.norm(points, axis=1, keepdims=True)))
        tetrahedron = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
        self.assertAlmostEqual(sphere.check(tetrahedron * 3.0, 3, 4)[0], math.degrees(math.acos(-1 / 3)), places=10)
        self.assertAlmostEqual(thomson.check([[0, 0, 1], [0, 0, -2.0]], 2)[0], 0.5, places=12)

    def test_invalid_points(self):
        nan, inf = float("nan"), float("inf")
        for bad in ([[nan, 0, 0], [0, 1, 0]], [[inf, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 1, 0]], [[1, 0], [0, 1]], None):
            self.assertTrue(rejected(sphere.check, bad, 3, 2))
            self.assertTrue(rejected(thomson.check, bad, 2))
        with np.errstate(divide="ignore"):
            self.assertTrue(rejected(thomson.check, [[1, 0, 0], [2, 0, 0]], 2))  # coincident after scaling

    def test_thomson_energy_against_plain_sum(self):
        rng = np.random.default_rng(4)
        points = rng.normal(size=(12, 3))
        unit = [[x / math.sqrt(sum(v * v for v in row)) for x in row] for row in points.tolist()]
        plain = math.fsum(1.0 / math.dist(unit[i], unit[j]) for i in range(12) for j in range(i))
        self.assertAlmostEqual(thomson.check(points, 12)[0], plain, places=10)


class ExactIntegers(unittest.TestCase):
    """3.7 was read as 3 by the route, snake and comparator verifiers."""

    INSTANCE = {"n": 3, "capacity": 10, "coords": np.array([[0, 0], [3, 4], [6, 8], [0, 5]], dtype=float),
                "demand": np.array([0, 4, 5, 6], dtype=np.int64), "name": "tiny"}

    def test_cvrp(self):
        self.assertEqual(cvrp.route_cost([[1, 2], [3]], self.INSTANCE)[0], 30)
        self.assertEqual(cvrp.route_cost([[1.0, 2.0], np.array([3])], self.INSTANCE)[0], 30)
        for bad in ([[1.9, 2.2], [3.7]], [["1", "2"], ["3"]], [[1, float("inf")], [3]], [[1, float("nan")], [3]]):
            self.assertTrue(rejected(cvrp.route_cost, bad, self.INSTANCE), bad)
        for bad in ([[0, 1, 2], [3]], [[1, 2, 3]], [[1, 2], [3], [1]], [[1, 2]], [[1, 2], [4]], None, [1, 2, 3]):
            self.assertTrue(rejected(cvrp.route_cost, bad, self.INSTANCE), bad)

    def test_cvrp_distances_are_rounded_per_leg(self):
        # depot (0, 0); legs 1.414 -> 1, 2.236 -> 2, 3.606 -> 4, as TSPLIB's nint for EUC_2D
        instance = {"n": 3, "capacity": 9, "coords": np.array([[0, 0], [1, 1], [2, 3], [0, 0]], dtype=float),
                    "demand": np.array([0, 1, 1, 1], dtype=np.int64), "name": "legs"}
        self.assertEqual(cvrp.route_cost([[1, 2]], dict(instance, n=2, coords=instance["coords"][:3], demand=instance["demand"][:3]))[0], 1 + 2 + 4)

    def test_snake(self):
        self.assertEqual(snake.check([0, 1, 3, 7, 6], 3)[0], 4)
        self.assertEqual(snake.check([0.0, 1.0, 3.0], 3)[0], 2)
        for bad in ([0, 1.5, 3.2, 7.9, 6.1], ["0", "1", "3"], [0, float("inf")], [0, 1, 3, 2], [0, 1, 0], [0, 8], [0]):
            self.assertTrue(rejected(snake.check, bad, 3), bad)

    def test_sorting_network(self):
        network = [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]
        self.assertEqual(sortnet.check(network, 4)[0], 5)
        self.assertEqual(sortnet.check(np.array(network, dtype=float), 4)[0], 5)
        for bad in ([(a + 0.2, b + 0.9) for a, b in network], [("0", "1")], network[:-1], [(1, 0)] + network, []):
            self.assertTrue(rejected(sortnet.check, bad, 4), bad)

    def test_exact_int(self):
        self.assertEqual([common.exact_int(x) for x in (3, 3.0, np.int64(3), True)], [3, 3, 3, 1])
        for bad in (3.9, "3", float("nan"), float("inf")):
            with self.assertRaises((ValueError, TypeError)):
                common.exact_int(bad)


class IndependentVerifiers(unittest.TestCase):
    """Small exhaustive comparisons with implementations written from the definitions."""

    def test_sorting_networks_against_all_zero_one_inputs(self):
        import random
        rnd = random.Random(5)

        def sorts(comparators, n):
            for bits in itertools.product((0, 1), repeat=n):
                wires = list(bits)
                for i, j in comparators:
                    if wires[i] > wires[j]:
                        wires[i], wires[j] = wires[j], wires[i]
                if any(wires[k] > wires[k + 1] for k in range(n - 1)):
                    return False
            return True

        saved = sortnet.BLOCK_BITS
        try:
            for block_bits in (saved, 3, 6):  # small blocks run the path of the wires above the block (n = 25..32 in the tasks)
                sortnet.BLOCK_BITS = block_bits
                for n in range(2, 9):
                    for _ in range(12):
                        comparators = sortnet.batcher(n)
                        if rnd.random() < 0.6:
                            del comparators[rnd.randrange(len(comparators))]
                        if rnd.random() < 0.3:
                            rnd.shuffle(comparators)
                        self.assertEqual(not rejected(sortnet.check, comparators, n), sorts(comparators, n), (block_bits, n, comparators))
        finally:
            sortnet.BLOCK_BITS = saved

    def test_snake_against_pairwise_definition(self):
        import random
        rnd = random.Random(6)
        for _ in range(400):
            n = rnd.randrange(2, 6)
            path = [rnd.randrange(2 ** n)]
            for _ in range(rnd.randrange(1, 10)):
                path.append(path[-1] ^ (1 << rnd.randrange(n)))
            valid = len(set(path)) == len(path) and all(
                bin(path[a] ^ path[b]).count("1") >= 2 for a in range(len(path)) for b in range(a + 2, len(path)))
            self.assertEqual(not rejected(snake.check, path, n), valid, path)

    def test_dna_codes(self):
        complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

        def distance(a, b):
            return sum(x != y for x, y in zip(a, b))

        words = ["".join(w) for w in itertools.product("ACGT", repeat=4) if sum(c in "CG" for c in w) == 2]
        code = []
        for word in words:  # greedy code with distance 3 and the reverse-complement rule, the word itself included
            reverse = "".join(complement[c] for c in reversed(word))
            if distance(word, reverse) >= 3 and all(distance(word, v) >= 3 and distance(reverse, v) >= 3 for v in code):
                code.append(word)
        self.assertGreaterEqual(len(code), 2)
        self.assertEqual(dnacode.check(code, 4, 3, True)[0], len(code))
        self.assertEqual(dnacode.check(["CCAA", "GGTT", "AACC", "TTGG"], 4, 4, False)[0], 4)
        self.assertEqual(dnacode.check(["CAAA", "ACAA"], 4, 2, False)[1]["gc_content"], 1)  # any constant GC-content
        for bad, d, rc in ((["ACGT"], 3, True), (["CCAA", "TTGG"], 4, True), (["CCAA", "CCAT"], 4, False), (["CCAA", "CAAA"], 1, False),
                           (["CCAA", "CCAA"], 1, False), (["ccaa"], 1, False), ([b"CCAA"], 1, False), ([], 1, False)):
            self.assertTrue(rejected(dnacode.check, bad, 4, d, rc), bad)

    def test_job_shop(self):
        instance = {"jobs": [[(0, 3), (1, 2)], [(1, 2), (0, 4)]], "n_machines": 2, "n_jobs": 2, "name": "tiny"}
        self.assertEqual(jssp.makespan([[0, 3], [0, 3]], instance)[0], 7)
        self.assertEqual(jssp.makespan([[0.0, 3.0], [0.0, 5.0]], instance)[0], 9)
        for bad in ([[0, 3], [3, 5]], [[0, 2], [0, 7]], [[-1, 3], [0, 3]], [[0, 3.9], [0, 3]], [[0, 3]], [[0, 3, 9], [0, 3]], None):
            self.assertTrue(rejected(jssp.makespan, bad, instance), bad)

    def test_max_cut_and_qap(self):
        edges = np.array([[0, 1, 1], [1, 2, -1], [0, 2, 3]], dtype=np.int64)
        for sides in itertools.product((0, 1), repeat=3):
            expected = sum(w for i, j, w in edges.tolist() if sides[i] != sides[j])
            self.assertEqual(maxcut.cut_value(list(sides), 3, edges)[0], expected)
        with np.errstate(invalid="ignore"):
            for bad in ([0, 0.5, 1], [0, 2, 1], [0, 1], [0, float("nan"), 1], None):
                self.assertTrue(rejected(maxcut.cut_value, bad, 3, edges), bad)
        a = np.array([[0, 2, 1], [3, 0, 4], [5, 6, 0]], dtype=np.int64)
        b = np.array([[0, 5, 2], [7, 0, 1], [3, 8, 0]], dtype=np.int64)
        for p in itertools.permutations(range(3)):
            expected = sum(int(a[i, j]) * int(b[p[i], p[j]]) for i in range(3) for j in range(3))
            self.assertEqual(qap.objective(list(p), a, b)[0], expected)
        self.assertTrue(rejected(qap.objective, [0, 0, 1], a, b))

    def test_lut_count_convention(self):
        # two LUTs in series, an output buffer and a constant: 2 LUTs, 2 levels, widest node 2 (the EPFL table's counting)
        text = ".model t\n.inputs a b c\n.outputs o k\n.names a b x\n11 1\n.names x c y\n1- 1\n-1 1\n.names y o\n1 1\n.names k\n1\n.end\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "t.blif"
            path.write_text(text)
            self.assertEqual(epfl.blif_stats(path), (2, 2, 2))
            path.write_text(text.replace(".names x c y", ".gate nand2 a=x b=c O=y"))
            self.assertTrue(rejected(epfl.blif_stats, path))


class MiplibReader(unittest.TestCase):
    MPS = ("NAME tiny\nROWS\n N obj\n G c1\nCOLUMNS\n    MARKER 'MARKER' 'INTORG'\n    y obj -1 c1 1\n    w obj -1 c1 1\n"
           "    MARKER 'MARKER' 'INTEND'\n    z obj 1 c1 {coef}\nRHS\n    rhs c1 0\nBOUNDS\n UP bnd w 5\nENDATA\n")

    def test_feasibility_rule(self):
        model = {"num_cols": 2, "col_lower": np.array([0.0, 0.0]), "col_upper": np.array([1.0, 4.0]), "integrality": np.array([True, False]),
                 "A": np.array([[1.0, 1.0]]), "row_lower": np.array([2.0]), "row_upper": np.array([np.inf]), "col_cost": np.array([3.0, 1.0]),
                 "offset": 0.5, "maximise": False, "indicators": [(0, 0, 1.0)]}
        self.assertEqual(miplib.check([1, 1], model)[0], 4.5)
        self.assertEqual(miplib.check([0, 0], model)[0], 0.5)  # the indicator row binds only when column 0 equals 1
        self.assertAlmostEqual(miplib.check([1, 1 - 5e-7], model)[0], 4.5 - 5e-7, places=12)  # inside the tolerance of 1e-6, not rounded
        for bad in ([1, 0.9], [0.5, 2], [1, 4.1], [-0.1, 3], [1, float("nan")], [1], None):
            self.assertTrue(rejected(miplib.check, bad, model), bad)

    def test_integer_column_without_bounds_is_binary_in_the_plain_reader(self):
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("scipy is not installed")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.mps"
            path.write_text(self.MPS.format(coef="1e20"))
            model = miplib._parse_mps(str(path))
        self.assertEqual(model["col_names"], ["y", "w", "z"])
        self.assertEqual(model["col_upper"].tolist(), [1.0, 5.0, np.inf])  # y: [0, 1], as in MIPLIB's solution checker and HiGHS
        self.assertEqual(model["integrality"].tolist(), [True, True, False])

    def test_highs_start_leaves_stdout_clean(self):
        if importlib.util.find_spec("highspy") is None:
            self.skipTest("highspy is not installed")
        code = ("import sys; sys.path.insert(0, %r)\nfrom arnew import miplib\nh = miplib._highs()\nprint('RESULT')\n" % str(ROOT / "suite/_lib"))
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(run.stdout, "RESULT\n", run.stdout[:200])  # the banner of the first Highs() goes to stderr

    def test_both_readers_agree_on_default_bounds(self):
        if importlib.util.find_spec("highspy") is None or importlib.util.find_spec("scipy") is None:
            self.skipTest("highspy or scipy is not installed")
        saved = os.environ.get("ARNEW_DATA")
        models = []
        try:
            for coef in ("1", "1e20"):  # HiGHS reads the first file and rejects the second
                with tempfile.TemporaryDirectory() as folder:
                    os.environ["ARNEW_DATA"] = folder
                    (Path(folder) / "miplib").mkdir()
                    with gzip.open(Path(folder) / "miplib" / "tiny.mps.gz", "wt") as stream:
                        stream.write(self.MPS.format(coef=coef))
                    models.append(miplib.load("tiny", folder))
        finally:
            if saved is None:
                os.environ.pop("ARNEW_DATA", None)
            else:
                os.environ["ARNEW_DATA"] = saved
        for model in models:
            self.assertEqual(model["col_upper"].tolist(), [1.0, 5.0, np.inf])
            self.assertTrue(rejected(miplib.check, [2, 5, 0], model))
            self.assertEqual(miplib.check([1, 5, 0], model)[0], -6.0)


class RegistryAndPackages(unittest.TestCase):
    """The registry files next to prepare_new_data.py and the constants of the evaluators."""

    def test_cvrp(self):
        records = dict(registry_cvrp.X_INSTANCES + registry_cvrp.XL_INSTANCES)
        seen = {}
        for task in packages("cvrplib-"):
            parameters = evaluator_parameters(task)
            seen[parameters["instance"]] = parameters["best_known"]
        self.assertEqual(seen, records)
        self.assertEqual((records["X-n766-k71"], records["XL-n8389-k2028"]), (114413, 3352570))  # CVRPLIB, 2026-11-24

    def test_job_shop_qap_and_sorting_networks(self):
        for task in packages("jssp-"):
            parameters = evaluator_parameters(task)
            jobs, machines, lower, upper = registry_misc.JSSP[parameters["instance"]]
            self.assertEqual((parameters["lower_bound"], parameters["best_known"]), (lower, upper), task)
            self.assertLess(lower, upper, task)
        for task in packages("qaplib-"):
            parameters = evaluator_parameters(task)
            if parameters["instance"] in registry_misc.QAP:  # the three older QAPLIB packages are not in this registry
                self.assertEqual((parameters["n"], parameters["best_known"], parameters["lower_bound"]), registry_misc.QAP[parameters["instance"]], task)
        bound = 39  # 12 inputs, proven; Van Voorhis: one more input costs at least ceil(log2 n) comparators
        for n in range(13, 33):
            bound += math.ceil(math.log2(n))
            if n in registry_misc.SORTNET:
                parameters = evaluator_parameters(f"sorting-network-{n}")
                self.assertEqual((parameters["best_known"], parameters["lower_bound"]), (registry_misc.SORTNET[n][0], bound), n)
                self.assertEqual(registry_misc.SORTNET[n][2], bound, n)

    def test_miplib_epfl_and_admet(self):
        for task in packages("miplib-"):
            parameters = evaluator_parameters(task)
            self.assertEqual(parameters["best_known"], registry_miplib.OPEN[parameters["instance"]], task)
        by_stem = {stem: (size, depth) for _, stem, size, depth in registry_misc.EPFL.values()}
        for task in packages("epfl-"):
            parameters = evaluator_parameters(task)
            self.assertEqual(parameters["best_known"], by_stem[parameters["circuit"]][parameters["metric"] == "depth"], task)
        for task in packages("tdc-admet-"):
            parameters = evaluator_parameters(task)
            metric, best, _ = registry_misc.ADMET[parameters["name"]]
            self.assertEqual((parameters["metric"], parameters["best_known"]), (metric, best), task)

    def test_seeds_that_matched_the_record_were_replaced(self):
        for task in ("epfl-alu-control-unit-depth", "epfl-barrel-shifter-size", "epfl-barrel-shifter-depth", "epfl-decoder-depth"):
            spec = importlib.util.spec_from_file_location("seed_" + task.replace("-", "_"), ROOT / "suite" / task / "initial_program.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            script = module.script()
            epfl.check_script(script)
            self.assertNotIn("dc2", script, task)  # `strash; dc2; if -K 6` already gives the best known value on these four
        self.assertIn("mip_max_improving_sols", (ROOT / "suite/miplib-neos-3009394-lami/initial_program.py").read_text())


class MatbenchPreparation(unittest.TestCase):
    def test_single_task_can_be_derived(self):
        from autoresearch_suite import chem_sources
        spec = importlib.util.spec_from_file_location("prepare_new_data_under_test", SCRIPTS / "prepare_new_data.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        saved = dict(chem_sources.MATBENCH), set(chem_sources.CLASSIFICATION)
        rows = [["Al", False], ["Cu", True], ["Zr", True], ["Fe", False], ["Ni", True]] * 2
        identifiers = [f"mb-glass-{k + 1:02d}" for k in range(10)]
        folds = {f"fold_{k}": {"test": identifiers[2 * k:2 * k + 2], "train": identifiers[:2 * k] + identifiers[2 * k + 2:]} for k in range(5)}
        try:
            with tempfile.TemporaryDirectory() as folder:
                raw = Path(folder) / "raw"
                raw.mkdir()
                (raw / "validation.json").write_text(json.dumps({"splits": {"matbench_glass": folds}}))
                with gzip.open(raw / "matbench_glass.json.gz", "wt") as stream:
                    json.dump({"index": list(range(10)), "columns": ["composition", "gfa"], "data": rows}, stream)
                with contextlib.redirect_stdout(io.StringIO()):
                    made = module.derive_matbench(Path(folder))  # the four other raw files are absent: this raised FileNotFoundError
                self.assertEqual(made, ["glass.json"])
                task = json.loads((Path(folder) / "glass.json").read_text())
                self.assertEqual(task["test_folds"], [[2 * k, 2 * k + 1] for k in range(5)])
                self.assertEqual(task["task_type"], "classification")
        finally:
            chem_sources.MATBENCH.clear()
            chem_sources.MATBENCH.update(saved[0])
            chem_sources.CLASSIFICATION.clear()
            chem_sources.CLASSIFICATION.update(saved[1])


if __name__ == "__main__":
    unittest.main()
