"""Tests of the BIG-bench scaling-law evaluator (metaopt_lasr) on synthetic rows; no prepared data needed."""
import importlib.util
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import metaopt_lasr as lasr  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402


def synthetic_rows(n=600, seed=1):
    rng = np.random.default_rng(seed)
    steps = rng.choice([1e5, 3e5, 1e6], n)
    X = {"training_steps": steps, "batch_size": rng.choice([2.6e5, 5e5], n), "total_params": rng.choice([1e7, 1e8, 1e9], n),
         "non_embedding_params": rng.choice([2e6, 5e7, 8e8], n), "flop_matched_params": rng.choice([2e6, 5e7], n),
         "shots": rng.choice([0.0, 1.0, 2.0, 3.0], n), "temperature": rng.choice([0.0, 1.0], n)}
    y = 0.2 + 0.05 * np.log10(steps) + 0.01 * X["shots"] + rng.normal(0, 0.02, n)
    groups = [f"{a}|{b}" for a, b in zip(steps, X["shots"])]
    return X, y, groups, ["task"] * n


def candidate(source):
    directory = tempfile.mkdtemp()
    path = Path(directory) / "candidate.py"
    path.write_text(textwrap.dedent(source))
    module = types.ModuleType("candidate")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


GOOD = """
import numpy as np
N_PARAMS = 3
def initial_params():
    return [0.2, 0.05, 0.0]
def scaling_law(X, params):
    a, b, c = params
    return a + b * np.log10(X["training_steps"]) + c * X["shots"]
"""


class LasrTests(unittest.TestCase):
    def setUp(self):
        rows = synthetic_rows()
        patcher = mock.patch.object(lasr, "load_rows", return_value=rows)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.y = rows[1]

    def test_split_sizes_match_the_paper_counts(self):
        train, test = lasr.split_indices(53812, seed=0)
        self.assertEqual((len(train), len(test)), (43049, 10763))
        self.assertEqual(len(set(train) | set(test)), 53812)
        again, _ = lasr.split_indices(53812, seed=0)
        self.assertTrue(np.array_equal(train, again))

    def test_true_law_recovers_noise_level(self):
        mse, details = lasr.score(candidate(GOOD), {})
        self.assertLess(mse, 0.02 ** 2 * 1.5)
        self.assertAlmostEqual(details["params"][1], 0.05, delta=0.01)
        self.assertLess(mse, details["reference_constant_mse"])
        self.assertEqual(details["n_train"] + details["n_test"], len(self.y))

    def test_constant_law_matches_constant_reference(self):
        module = candidate("""
        import numpy as np
        N_PARAMS = 1
        def scaling_law(X, params):
            return np.full(len(X["shots"]), params[0])
        """)
        mse, details = lasr.score(module, {})
        self.assertAlmostEqual(mse, details["reference_constant_mse"], places=8)

    def test_rejects_too_many_parameters(self):
        module = candidate(GOOD.replace("N_PARAMS = 3", "N_PARAMS = 6"))
        with self.assertRaises(CandidateError):
            lasr.score(module, {})

    def test_rejects_hidden_constant_tables(self):
        table = ", ".join(f"{0.1 + i / 1000:.4f}" for i in range(12))
        module = candidate(GOOD + f"\nTABLE = [{table}]\n")
        with self.assertRaises(CandidateError):
            lasr.score(module, {})

    def test_rejects_small_integer_and_string_tables(self):
        ints = ", ".join(str(i % 7) for i in range(100))
        with self.assertRaises(CandidateError):
            lasr.score(candidate(GOOD + f"\nTABLE = [{ints}]\n"), {})
        with self.assertRaises(CandidateError):
            lasr.score(candidate(GOOD + "\nBLOB = '" + "ab" * 600 + "'\n"), {})
        documented = candidate(GOOD.replace("def scaling_law(X, params):", 'def scaling_law(X, params):\n    """' + "x" * 1500 + '"""'))
        self.assertLess(lasr.score(documented, {})[0], 0.02 ** 2 * 1.5)  # docstrings do not count

    def test_rejects_wrong_shape_and_file_access(self):
        wrong = candidate("""
        import numpy as np
        N_PARAMS = 1
        def scaling_law(X, params):
            return np.zeros(3)
        """)
        with self.assertRaises(CandidateError):
            lasr.score(wrong, {})
        snooping = candidate("""
        import numpy as np
        N_PARAMS = 1
        def scaling_law(X, params):
            open("/etc/hosts").read()
            return np.full(len(X["shots"]), params[0])
        """)
        with self.assertRaises(CandidateError):
            lasr.score(snooping, {})

    def test_rejects_nondeterministic_law(self):
        module = candidate("""
        import numpy as np
        N_PARAMS = 1
        def scaling_law(X, params):
            return params[0] + np.random.default_rng().normal(0, 0.1, len(X["shots"]))
        """)
        with self.assertRaises(CandidateError):
            lasr.score(module, {})

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is not installed (the reference of the Levenberg-Marquardt check)")
    def test_levenberg_marquardt_matches_scipy(self):
        from scipy.optimize import least_squares
        X, y, _, _ = synthetic_rows(n=800, seed=3)
        y = 0.35 - 0.03 / (X["training_steps"] / 2e5) ** (0.5 * X["shots"] + 0.5) + 0.01 * np.sin(np.arange(len(y)))
        law = """
import numpy as np
N_PARAMS = 3
def initial_params():
    return [0.3, -0.01, 1e5]
def scaling_law(X, params):
    E, A, B = params
    return E + A / (X["training_steps"] / B) ** (0.5 * X["shots"] + 0.5)
"""
        namespace = {}
        exec(compile(textwrap.dedent(law) + lasr.HARNESS, "<law>", "exec"), namespace)
        cols = {k: v.tolist() for k, v in X.items()}
        fit = namespace["_artask_fit"](cols, y.tolist(), cols, 3, 8, 0, 400)
        residual = lambda p: namespace["scaling_law"](X, p) - y  # noqa: E731
        reference = min(float(np.mean(least_squares(residual, x0, method="lm").fun ** 2))
                        for x0 in ([0.3, -0.01, 1e5], [0.4, -0.05, 3e5], [0.2, -0.02, 5e4]))
        self.assertLessEqual(fit["train_mse"], reference * (1 + 1e-6) + 1e-12)

    def test_literal_count(self):
        self.assertEqual(lasr.literal_count("x = 1 + 2.5 - 1e6 + 10 + 11"), 3)

    def test_group_mean_ceiling(self):
        X, y, groups, _ = synthetic_rows()
        train, test = lasr.split_indices(len(y))
        ceiling = lasr.group_mean_mse(groups, y, train, test)
        self.assertLess(ceiling, float(np.mean((y[train].mean() - y[test]) ** 2)))


if __name__ == "__main__":
    unittest.main()
