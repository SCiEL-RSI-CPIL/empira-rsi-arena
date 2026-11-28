"""Tests of the EoB-style benchmark-synthesis evaluator (metaopt_eob, metaopt_ela and the optimiser pool)."""
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import metaopt_ela as ela  # noqa: E402
from autoresearch_suite import metaopt_eob as eob  # noqa: E402
from autoresearch_suite.blackbox import BBOBFunction  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402


def sphere(X):
    return np.sum((X - 1.0) ** 2, axis=1)


class PoolTests(unittest.TestCase):
    def test_every_optimiser_respects_the_budget_and_solves_a_sphere(self):
        for name, algorithm in eob.POOL.items():
            calls = []

            def counted(X):
                calls.append(len(X))
                return sphere(X)

            budget = eob.Budget(counted, 10, -5.0, 5.0, 3000)
            algorithm(budget, np.random.default_rng(0))
            self.assertEqual(sum(calls), 3000, name)
            self.assertLess(budget.best, 20.0, name)  # a random point scores about 100

    def test_runs_are_reproducible(self):
        a, b = (eob.Budget(sphere, 10, -5.0, 5.0, 1000) for _ in range(2))
        eob.POOL["jade"](a, np.random.default_rng([0, 2, 1]))
        eob.POOL["jade"](b, np.random.default_rng([0, 2, 1]))
        self.assertEqual(a.best, b.best)

    def test_budget_clips_points_to_the_box(self):
        seen = []
        budget = eob.Budget(lambda X: seen.append(X.copy()) or np.zeros(len(X)), 2, -5.0, 5.0, 10)
        budget(np.array([[9.0, -9.0]]))
        self.assertTrue(np.array_equal(seen[0], [[5.0, -5.0]]))


class FeatureTests(unittest.TestCase):
    def test_features_are_invariant_to_bias_and_positive_scale(self):
        X = ela.sample(10, -5, 5, 300, 1)
        y = sphere(X) + np.sin(X[:, 0])
        a, b = ela.features(X, y, -5, 5), ela.features(X, 7.0 * y + 1e3, -5, 5)
        self.assertTrue(np.allclose(a, b, atol=1e-6))
        self.assertEqual(len(a), len(ela.NAMES))

    def test_held_out_bbob_instance_is_more_similar_than_noise(self):
        ref = eob.anchors()
        X = ref["X"]
        noise = eob.lsi(ela.features(X, np.sin(37.0 * X @ np.arange(1, 11)) * np.cos(11.0 * X[:, 0]), -5, 5), ref)
        for fid in (1, 24):  # instance 9 is not among the anchors' instances 1-5
            f = BBOBFunction(fid, 10, instance=9)
            bbob = eob.lsi(ela.features(X, np.array([f(x) for x in X]), -5, 5), ref)
            self.assertEqual(bbob[1], fid)
            self.assertGreater(bbob[0], noise[0])
        centroid = eob.lsi(ref["Z"][2] * ref["scale"], ref)
        self.assertEqual(centroid[1], 3)
        self.assertAlmostEqual(centroid[0], 1.0)
        copy = eob.lsi(ref["raw"][2, 0], ref)  # an exact copy of an anchor instance is not an exact match
        self.assertLess(copy[0], 0.9)


class ScoreTests(unittest.TestCase):
    def test_seed_like_function_scores(self):
        module = types.SimpleNamespace(objective=lambda X: np.sum((X - 1.5) ** 2, 1) + 10 * np.sum(1 - np.cos(2 * np.pi * (X - 1.5)), 1))
        value, details = eob.score(module, {"runs": 2})
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertEqual(len(details["pool_mean_best"]), 10)

    def test_rejects_invalid_functions(self):
        state = {"calls": 0}

        def stateful(X):
            state["calls"] += 1
            return sphere(X) + state["calls"]

        bad = {
            "nan": lambda X: np.where(X[:, 0] > 4.9, np.nan, sphere(X)),
            "huge": lambda X: 1e300 * sphere(X),
            "shape": lambda X: sphere(X)[:, None],
            "constant": lambda X: np.ones(len(X)),
            "batch": lambda X: sphere(X) + len(X),
            "stateful": stateful,
        }
        for name, function in bad.items():
            with self.assertRaises(CandidateError, msg=name):
                eob.score(types.SimpleNamespace(objective=function), {"runs": 1})

    def test_special_casing_the_public_sample_is_caught(self):
        X = eob.anchors()["X"]
        bbob = BBOBFunction(9, 10, instance=9)
        table = {tuple(row): bbob(row) for row in X}

        def cheat(Z):  # a BBOB instance on the public sample points, a product of sines everywhere else
            return np.array([table[tuple(z)] if tuple(z) in table else float(np.prod(np.sin(z))) for z in Z])

        _, honest = eob.score(types.SimpleNamespace(objective=sphere), {"runs": 1})
        self.assertLess(abs(honest["lsi_fixed_sample"] - honest["lsi_perturbed_sample"]), 0.01)
        _, faked = eob.score(types.SimpleNamespace(objective=cheat), {"runs": 1})
        self.assertGreater(faked["lsi_fixed_sample"] - faked["lsi_perturbed_sample"], 0.2)
        self.assertEqual(faked["lsi"], min(faked["lsi_fixed_sample"], faked["lsi_perturbed_sample"]))

    def test_mutating_input_is_harmless(self):
        def mutating(X):
            X += 100.0
            return sphere(X - 100.0)
        value, _ = eob.score(types.SimpleNamespace(objective=mutating), {"runs": 1})
        expected, _ = eob.score(types.SimpleNamespace(objective=sphere), {"runs": 1})
        self.assertAlmostEqual(value, expected, places=3)  # X + 100 - 100 is not exact


if __name__ == "__main__":
    unittest.main()
