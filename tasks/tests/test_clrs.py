"""EvoPrompting CLRS-30 tasks (package clrs): specs, metrics, source guard, upgrade records (numpy only).

The source repository's test reads the seed program, the label rule and the family registration from
its generator module (scripts/families_clrs.py), which is not part of this library; the same values
are read here from the generated packages (suite/evoprompting-*/).
"""
from __future__ import annotations

import base64
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "suite" / "_lib", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from autoresearch_suite import clrs_data as D  # noqa: E402
from autoresearch_suite import clrs_eval as E  # noqa: E402
from autoresearch_suite import clrs_metrics as M  # noqa: E402
from autoresearch_suite import clrs_sandbox as SB  # noqa: E402
from autoresearch_suite import clrs_specs as S  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

import suite_test_support as support  # noqa: E402

UPGRADES = json.loads((ROOT / "catalog" / "upgrades" / "clrs.json").read_text())
LOCK = json.loads((ROOT / "catalog" / "clrs-data-lock.json").read_text())
HAVE_DATA = all((D.data_dir() / f"{a}.npz").is_file() for a in S.ALGORITHMS)
TASKS = sorted(UPGRADES["tasks"])
# Label rule of the family: (scientific_contribution, artifact, domain, data, verification).
LABELS = ("engineering", "model", "machine_learning", "public_benchmark", "simulation")


def reference_mask_f1(pred, truth):
    """dm-clrs evaluation._mask_fn, transcribed."""
    mask = (truth != -1).astype(np.float32)
    tp = np.sum((((pred > 0.5) * (truth > 0.5)) * 1.0) * mask)
    fp = np.sum((((pred > 0.5) * (truth < 0.5)) * 1.0) * mask)
    fn = np.sum((((pred < 0.5) * (truth > 0.5)) * 1.0) * mask)
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


class SpecTests(unittest.TestCase):
    def test_thirty_algorithms_and_task_ids(self):
        self.assertEqual(len(S.ALGORITHMS), 30)
        self.assertEqual(set(S.ALGORITHMS), set(S.SPECS))
        self.assertEqual(S.task_algorithm("evoprompting-knuth-morris-pratt"), "kmp_matcher")
        self.assertEqual(S.task_algorithm("evoprompting-find-maximum-subarray-kadane"), "find_maximum_subarray_kadane")
        with self.assertRaises(KeyError):
            S.task_algorithm("evoprompting-mnist1d")

    def test_model_spec_conversions(self):
        spec = S.model_spec("insertion_sort")
        self.assertEqual(spec["pred"], (S.OUTPUT, S.NODE, S.PERMUTATION_POINTER))
        self.assertEqual(spec["pred_mask"], (S.OUTPUT, S.NODE, S.MASK_ONE))
        self.assertEqual(S.model_spec("minimum")["pred"], (S.INPUT, S.NODE, S.POINTER))
        self.assertNotIn("pred_h", S.model_spec("minimum"))
        for algorithm in S.ALGORITHMS:
            self.assertTrue(M.outputs(algorithm), algorithm)

    def test_cyclic_predecessor_round_trip(self):
        rng = np.random.RandomState(0)
        order = np.stack([rng.permutation(6) for _ in range(4)])
        pred = np.zeros((4, 6), dtype=np.int64)
        for b in range(4):
            pred[b, order[b, 0]] = order[b, 0]
            for i in range(1, 6):
                pred[b, order[b, i]] = order[b, i - 1]
        perm, first = D.cyclic_predecessor_and_first(pred)
        self.assertTrue(np.allclose(perm.sum(-1), 1) and np.allclose(perm.sum(-2), 1))  # a permutation matrix
        fused = M.fuse_permutation(perm.argmax(-1), first.argmax(-1))
        np.testing.assert_array_equal(fused, pred)


class MetricTests(unittest.TestCase):
    def test_mask_f1_matches_reference(self):
        rng = np.random.RandomState(1)
        for _ in range(50):
            truth = rng.choice([-1, 0, 1], size=(5, 7), p=[0.1, 0.5, 0.4]).astype(np.int8)
            pred = rng.randint(0, 2, size=(5, 7)).astype(np.int8)
            self.assertAlmostEqual(M.mask_f1(pred, truth), reference_mask_f1(pred, truth), places=12)
        self.assertEqual(M.mask_f1(np.zeros(3), np.zeros(3)), 1.0)  # no positives anywhere: CLRS returns 1

    def test_one_hot_accuracy_excludes_masked_rows(self):
        truth = np.array([[0, 1, 0], [1, 0, 0], [-1, -1, -1], [0, 0, 1]], dtype=np.int8)
        self.assertAlmostEqual(M.one_hot_accuracy(np.array([1, 2, 0, 2]), truth), 2 / 3)

    def test_score_is_mean_over_outputs(self):
        truths = {"topo": np.array([[0, 0, 1]]), "topo_head": np.array([[1, 0, 0]], dtype=np.int8)}
        preds = {"topo": np.array([[0, 1, 1]]), "topo_head": np.array([0])}
        value, per_output = M.score("topological_sort", preds, truths)
        self.assertAlmostEqual(per_output["topo"], 2 / 3)
        self.assertEqual(per_output["topo_head"], 1.0)
        self.assertAlmostEqual(value, (2 / 3 + 1) / 2)

    def test_check_predictions_rejects_bad_payloads(self):
        truths = {"pi": np.zeros((2, 5), dtype=np.int8)}
        good = {"pi": np.zeros((2, 5), dtype=np.int64)}
        self.assertEqual(M.check_predictions("bfs", good, truths, 5)["pi"].dtype, np.int64)
        for bad in ({}, {"pi": good["pi"], "extra": good["pi"]}, {"pi": np.zeros((2, 4), dtype=np.int64)},
                    {"pi": np.zeros((2, 5))}, {"pi": np.full((2, 5), 5)}, {"pi": np.full((2, 5), -1)}, [good["pi"]]):
            with self.assertRaises(CandidateError):
                M.check_predictions("bfs", bad, truths, 5)
        onehot = {"min": np.eye(4, dtype=np.int8)[[1, 2]]}
        M.check_predictions("minimum", {"min": np.array([3, 0])}, onehot, 4)
        with self.assertRaises(CandidateError):
            M.check_predictions("minimum", {"min": np.array([4, 0])}, onehot, 4)


class GuardTests(unittest.TestCase):
    def test_seed_passes_the_guard(self):
        for task_id in TASKS:
            SB.check_source(support.seed_source(task_id))

    def test_guard_rejections(self):
        for source in ("import os\n", "import sys\n", "from autoresearch_suite import clrs_eval\n", "from . import x\n",
                       "x = open('f')\n", "y = eval('1')\n", "z = getattr(int, 'real')\n", "w = (1).__class__\n",
                       "import torch\nq = torch.__dict__\n", "import importlib\n", "def f(:\n"):
            with self.assertRaises(CandidateError, msg=source):
                SB.check_source(source)
        SB.check_source("import torch\nfrom torch import nn\nimport math, functools\n\nclass A(nn.Module):\n"
                        "    def __init__(self):\n        super().__init__()\n")

    def test_decode_arrays(self):
        array = np.arange(6, dtype=np.int64).reshape(2, 3)
        np.testing.assert_array_equal(SB.decode_arrays(SB.encode_arrays({"a": array}))["a"], array)
        raw = base64.b64encode(array.tobytes()).decode()
        for bad in ({"a": {"dtype": "float64", "shape": [2, 3], "data": raw}},
                    {"a": {"dtype": "int64", "shape": [2, 2], "data": raw}},
                    {"a": {"dtype": "int64", "shape": [2, 3], "data": "!!"}}, {"a": [1, 2]}, [1]):
            with self.assertRaises(CandidateError):
                SB.decode_arrays(bad)


class UpgradeRecordTests(unittest.TestCase):
    def test_upgraded_tasks_use_locked_data(self):
        tasks = UPGRADES["tasks"]
        self.assertEqual(UPGRADES["schema"], "artask.upgrades/1")
        self.assertGreater(len(tasks), 0)
        digests = {entry["path"][:-4]: entry["content_sha256"] for entry in LOCK["files"]}
        seen = set()
        for task_id, spec in tasks.items():
            algorithm = S.task_algorithm(task_id)
            seen.add(algorithm)
            P = spec["params"]
            self.assertEqual(spec["family"], "clrs_processor")
            self.assertEqual(P["algorithm"], algorithm)
            self.assertEqual(P["data_sha256"], digests[algorithm])
            self.assertIn(f"file:data/clrs/{algorithm}.npz", spec["requires"])
            self.assertTrue(set(P) <= set(E.BUDGET_KEYS) | {"algorithm", "data_sha256", "val_samples", "test_samples",
                                                            "train_seeds", "wall_seconds", "threads", "memory_mb"}, task_id)
            self.assertLessEqual(P["wall_seconds"], 1750, task_id)  # safety net inside the 1800 s task timeout
            self.assertGreaterEqual(len(spec["notes"]), 5, task_id)
            self.assertTrue(all(isinstance(note, str) and note for note in spec["notes"]))
        self.assertLessEqual(seen, set(S.ALGORITHMS))

    def test_every_task_is_upgraded_or_withdrawn(self):
        checks_path = ROOT / "validation" / "clrs-checks.json"
        if not checks_path.is_file():
            self.skipTest("validation/clrs-checks.json not written yet")
        checks = json.loads(checks_path.read_text())
        upgraded, withdrawn = set(UPGRADES["tasks"]), set(checks["withdraw"])
        everything = {"evoprompting-" + ("knuth-morris-pratt" if a == "kmp_matcher" else a.replace("_", "-")) for a in S.ALGORITHMS}
        self.assertFalse(upgraded & withdrawn)
        self.assertEqual(upgraded | withdrawn, everything)
        self.assertEqual(set(checks["tasks"]), upgraded)
        for task_id, reason in checks["withdraw"].items():
            self.assertTrue(isinstance(reason, dict) and reason.get("reason"), task_id)

    def test_family_registration(self):
        for task_id in TASKS:
            family = support.package(task_id)
            self.assertEqual(family["family"], "clrs_processor", task_id)
            self.assertIn("clrs_eval.score(module, P)", family["body"], task_id)
            self.assertEqual(family["direction"], "maximize", task_id)
            self.assertLessEqual(family["timeout"], 1800, task_id)
            labels = support.load(f"suite/{task_id}/problem.json")["labels"]
            self.assertEqual((labels["scientific_contribution"], labels["artifact"], labels["domain"], labels["data"],
                              labels["verification"]), LABELS, task_id)


@unittest.skipUnless(HAVE_DATA, "CLRS-30 files not prepared (scripts/prepare_clrs.py)")
class PreparedDataTests(unittest.TestCase):
    def test_locked_digests_and_splits(self):
        for entry in LOCK["files"]:
            algorithm = entry["path"][:-4]
            data = D.load(algorithm, expected_digest=entry["content_sha256"])
            for split in ("train", "val", "test"):
                self.assertEqual(len(data[split]["lengths"]), entry["samples"][split])
                self.assertEqual(data[split]["input_pos"].shape[1], entry["nodes"][split])

    def test_child_never_receives_test_outputs(self):
        data = D.load("dfs")
        splits = E.visible_splits(data, np.arange(4), np.arange(3))
        self.assertFalse([key for key in splits["test"] if key.startswith("output_")])
        self.assertTrue([key for key in splits["val"] if key.startswith("output_")])
        self.assertEqual(len(splits["test"]["lengths"]), 3)
        batch = D.make_batch(splits["test"], "dfs", [0, 1], with_outputs=False)
        self.assertEqual(batch.outputs, [])
        self.assertEqual(batch.nb_nodes, 64)

    def test_batches_for_every_algorithm(self):
        rng = np.random.RandomState(0)
        for algorithm in S.ALGORITHMS:
            data = D.load(algorithm)
            batch = D.make_batch(data["train"], algorithm, [0, 1, 2], rng=rng)
            names = {p.name for p in batch.inputs + batch.hints + batch.outputs}
            self.assertEqual(names, set(S.model_spec(algorithm)), algorithm)
            for probe in batch.hints:
                self.assertEqual(probe.data.shape[0], int(batch.lengths.max()), algorithm)
            truth = M.truth_arrays(algorithm, data["test"], [0, 1])
            self.assertEqual(sorted(truth), sorted(name for name, _, _ in M.outputs(algorithm)))


if __name__ == "__main__":
    unittest.main()
