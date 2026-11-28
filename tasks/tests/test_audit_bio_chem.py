"""Biology and chemistry evaluators: regressions of the audit fixes and the conventions checked against the official scorers."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "suite/_lib")]
from autoresearch_suite import bio_align, bio_flu, bio_hp, bio_phylo, chem_adme, chem_matbench, chem_yields, runtime
from autoresearch_suite.runtime import CandidateError

# A diverged model: every prediction is finite, as the contracts require, but squares overflow a double.
HUGE = '''
def predict(train_smiles, train_values, test_smiles, info):
    return [1e200] * len(test_smiles)

def fit_predict(train_inputs, train_targets, test_inputs, info):
    return [1e200] * len(test_inputs)

def predict_yields(train_reactions, train_yields, test_reactions, info):
    return [1e200] * len(test_reactions)
'''


class HPCoordinateTests(unittest.TestCase):
    SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]  # HHHH folded into a square: one H-H contact

    def test_a_translated_walk_is_the_same_conformation(self):
        # The contract asks for integer lattice points, not for points near the origin.
        for shift in (0, 41, 10 ** 6, 2 ** 40 - 1):
            walk = [[x + shift, y - shift] for x, y in self.SQUARE]
            self.assertEqual(bio_hp.energy("HHHH", walk, 2), -1, shift)

    def test_coordinates_that_are_not_exact_in_float64_are_rejected(self):
        for shift in (2 ** 41, 2 ** 60, 10 ** 30):
            with self.assertRaises(CandidateError):
                bio_hp.energy("HHHH", [[x + shift, y] for x, y in self.SQUARE], 2)


class PredictionOverflowTests(unittest.TestCase):
    """Finite but huge predictions must give a result that runtime.main can print, not a crash of the evaluator."""

    def result(self, score_function, direction):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "candidate.py"
        path.write_text(HUGE)

        def task(module):
            score, details = score_function(module)
            return runtime.success(score, direction, **details)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # numpy reports the overflow this test provokes
            result = runtime.run(path, task, timeout=60)
        json.dumps(result, allow_nan=False, default=str)  # what runtime.main prints; raises on inf or NaN
        return result

    def test_reportable_replaces_only_nonfinite_floats(self):
        details = {"mae": 1.5, "rmse": float("inf"), "n": 3, "folds": [{"r2": float("-inf"), "name": "a"}, {"r2": float("nan")}]}
        self.assertEqual(chem_adme.reportable(details),
                         {"mae": 1.5, "rmse": None, "n": 3, "folds": [{"r2": None, "name": "a"}, {"r2": None}]})

    def test_adme(self):
        values = np.arange(6.0)
        with mock.patch.object(chem_adme, "load", return_value=(["C"] * 6, values, ["C"] * 4, values[:4])):
            result = self.result(lambda module: chem_adme.adme_score(module, "solubility", wall_seconds=30), "minimize")
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["score"] / 1e200, 1.0)  # the mean absolute error itself does not overflow
        self.assertIsNone(result["rmse"])
        self.assertIsNone(result["r2"])

    def test_matbench_regression(self):
        data = {"task": "matbench_test", "task_type": "regression", "input_type": "composition", "target": "t", "unit": "u",
                "inputs": [{"formula": "Fe", "composition": {"Fe": 1.0}}] * 6, "targets": [float(i) for i in range(6)],
                "test_folds": [[0, 1, 2], [3, 4, 5]]}
        with mock.patch.object(chem_matbench, "load", return_value=data):
            result = self.result(lambda module: chem_matbench.matbench_score(module, "steels", wall_seconds_per_fold=30),
                                 "minimize")
        self.assertTrue(result["valid"])
        self.assertIsNone(result["mean_rmse"])
        self.assertEqual([fold["r2"] for fold in result["folds"]], [None, None])

    def test_yields_fail_as_an_ordinary_evaluation(self):
        data = {"train_size": 4, "reactions": ["CC>>CC"] * 6, "yields_fraction": [0.1, 0.5, 0.2, 0.9, 0.4, 0.7],
                "split_orders": [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]]}
        with mock.patch.object(chem_yields, "load_json", return_value=data):
            result = self.result(lambda module: chem_yields.suzuki_miyaura_score(module, wall_seconds_per_split=30),
                                 "maximize")
        self.assertFalse(result["valid"])  # R^2 itself overflowed, so there is no score
        self.assertIn("nonfinite", result["error"])


class ConventionTests(unittest.TestCase):
    """The definitions the audit compared with the official implementations, pinned on small independent computations."""

    def test_weighted_interval_score_is_twice_the_mean_pinball_loss(self):
        # Bracher et al. (2021): with the median and K central intervals, WIS equals the mean over the 2K + 1
        # quantile levels of twice the pinball loss; the FluSight hub scores its 23 levels this way.
        generator = random.Random(3)
        for _ in range(200):
            quantiles = sorted(generator.uniform(0, 500) for _ in bio_flu.QUANTILES)
            observation = generator.choice([generator.uniform(0, 600), quantiles[generator.randrange(23)]])
            pinball = sum(2 * ((observation <= q) - level) * (q - observation) for level, q in zip(bio_flu.QUANTILES, quantiles))
            self.assertAlmostEqual(bio_flu.weighted_interval_score(observation, np.array(quantiles)), pinball / 23, places=9)

    def test_robinson_foulds_is_normalised_by_the_splits_of_both_trees(self):
        true = bio_phylo.parse_newick("((t0,t1),(t2,t3),(t4,t5));", 6)
        self.assertEqual(len(true), 3)  # n - 3 internal edges of a binary tree
        # Binary answer: the usual RF / (2 (n - 3)). One of three splits differs on each side.
        other = bio_phylo.parse_newick("((t0,t1),(t2,t4),(t3,t5));", 6)
        self.assertAlmostEqual(bio_phylo.robinson_foulds(true, other), 4 / 6)
        # Unresolved answers are not rewarded: a star tree scores 1, not the 0.5 that 2 (n - 3) would give.
        self.assertEqual(bio_phylo.robinson_foulds(true, bio_phylo.parse_newick("(t0,t1,t2,t3,t4,t5);", 6)), 1.0)
        partial = bio_phylo.parse_newick("((t0,t1),t2,t3,t4,t5);", 6)
        self.assertAlmostEqual(bio_phylo.robinson_foulds(true, partial), 2 / 4)
        # Rooting and child order do not matter.
        self.assertEqual(bio_phylo.robinson_foulds(true, bio_phylo.parse_newick("(t5,(t4,((t3,t2),(t1,t0))));", 6)), 0.0)

    def test_alignment_scores_follow_bali_score_on_core_blocks(self):
        # bali_score 3.01 with an XML reference: SP counts residue pairs of core columns that share a test column,
        # over n (n - 1) / 2 pairs per core column; TC counts core columns reproduced by all sequences.
        case = {"sequences": ["ACDE", "ACE", "CDE"], "reference": ["ACDE", "AC-E", "-CDE"], "core": [0, 1, 0, 1]}
        self.assertEqual(bio_align.case_scores(case, case["reference"]), (1.0, 1.0))
        # Shifting the gap of the second row misaligns its C: of the six core pairs, C-C (rows 0-1) and C-C (rows 1-2)
        # are lost, and only the last core column is still complete.
        sp, tc = bio_align.case_scores(case, ["ACDE", "A-CE", "-CDE"])
        self.assertAlmostEqual(sp, 4 / 6)
        self.assertAlmostEqual(tc, 1 / 2)

    def test_binary_roc_auc_of_labels_is_the_balanced_accuracy(self):
        # The Matbench leaderboard's "mean balanced accuracy" column; its "probability ROC AUC" needs probabilities.
        truth, labels = np.array([1, 1, 1, 0, 0, 0, 0, 1]), np.array([1, 0, 1, 0, 1, 0, 0, 1])
        metrics = chem_matbench.classification_metrics(truth, labels)
        self.assertAlmostEqual(metrics["balanced_accuracy"], (3 / 4 + 3 / 4) / 2)
        self.assertAlmostEqual(metrics["rocauc"], metrics["balanced_accuracy"])


if __name__ == "__main__":
    unittest.main()
