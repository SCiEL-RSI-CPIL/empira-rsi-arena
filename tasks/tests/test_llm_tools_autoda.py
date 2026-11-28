"""Offline tests of the L-AutoDA decision-attack evaluator (a stand-in classifier replaces the ResNet-18)."""
from __future__ import annotations

import base64
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import llm_tools_autoda as autoda  # noqa: E402
from autoresearch_suite import llm_tools_common as common  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

AUTODA_SEED = seed_source("l-autoda-cifar-10-decision-based-adversarial-attack")

NEEDS_TORCH = unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed (the attack's distance check)")


def brightness_classifier(model, images, batch=256):
    """Class 1 for images whose mean intensity exceeds 0.5, else class 0."""
    return (np.asarray(images, dtype=np.float64).reshape(len(images), -1).mean(axis=1) > 0.5).astype(np.int64)


def run_attack(source, steps=40, n=2):
    originals = np.full((n,) + autoda.SHAPE, 0.2, dtype=np.float32)
    starts = np.full((n,) + autoda.SHAPE, 0.9, dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    attack = autoda.Attack(None, originals, labels, starts, steps)
    encode = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode("ascii")  # noqa: E731
    payload = {"n": n, "shape": list(autoda.SHAPE), "originals": encode(originals), "starts": encode(starts), "noise_seed": 0,
               "s0": attack.s.tolist(), "steps": steps}
    with mock.patch.object(autoda, "classify", brightness_classifier):
        ok, _, failure = common.call_candidate(source, "draw_proposals", [], handlers={"query": attack.query}, harness=autoda.HARNESS,
                                               harness_entry="_artask_autoda_entry_", payload=payload, cpu_seconds=60)
    return attack, ok, failure


class AutodaTests(unittest.TestCase):
    def test_step_factor_matches_the_paper(self):
        self.assertEqual(autoda.step_factor(np.array([0.0, 0.25, 1.0])).tolist(), [0.5, 1.0, 1.5])
        self.assertAlmostEqual(float(autoda.step_factor(np.array([0.1]))[0]), 0.5 + 2 * 0.1)
        self.assertAlmostEqual(float(autoda.step_factor(np.array([0.7]))[0]), 5 / 6 + 2 * 0.7 / 3)

    @NEEDS_TORCH
    def test_seed_walks_towards_the_original_and_stays_adversarial(self):
        attack, ok, failure = run_attack(AUTODA_SEED)
        self.assertTrue(ok, failure)
        self.assertEqual(attack.step, 40)
        initial = np.linalg.norm(np.full(autoda.SHAPE, 0.7))
        final = attack.distances(attack.best)
        self.assertTrue((final < initial).all())
        self.assertTrue((brightness_classifier(None, attack.best) == 1).all())
        self.assertGreater(attack.accepted, 0)
        again, _, _ = run_attack(AUTODA_SEED)
        self.assertTrue(np.array_equal(again.best, attack.best))  # seeded noise: identical runs

    @NEEDS_TORCH
    def test_step_size_follows_acceptance(self):
        stay = "def draw_proposals(org, best, noise, s):\n    return best\n"
        attack, ok, _ = run_attack(stay, steps=10)
        self.assertTrue(ok)
        self.assertEqual(attack.accepted, 0)  # never closer, so never accepted
        self.assertTrue(np.allclose(attack.s, 0.05 * 0.5 ** (0.1 * 10)))

    def test_invalid_proposals(self):
        for source in ("import numpy as np\ndef draw_proposals(org, best, noise, s):\n    return np.zeros((3, 4, 4))\n",
                       "def draw_proposals(org, best, noise, s):\n    return best * float('nan')\n"):
            attack, ok, failure = (None, False, None)
            try:
                attack, ok, failure = run_attack(source, steps=3)
            except CandidateError as error:
                failure = {"message": str(error)}
            self.assertFalse(ok)
            self.assertTrue("shape" in failure["message"] or "non-finite" in failure["message"], failure)

    def test_protocol_is_enforced(self):
        attack = autoda.Attack(None, np.zeros((1,) + autoda.SHAPE, np.float32), np.zeros(1, np.int64), np.ones((1,) + autoda.SHAPE, np.float32), 2)
        with self.assertRaises(CandidateError):
            attack.query(1, "")  # out-of-order step
        with self.assertRaises(CandidateError):
            attack.query(0, "not base64!")

    @NEEDS_TORCH
    def test_resnet_layout_matches_timm_keys(self):
        keys = set(autoda.build_resnet18().state_dict())
        for key in ("conv1.weight", "bn1.running_var", "layer1.0.conv2.weight", "layer2.0.downsample.0.weight", "layer4.1.bn2.bias", "fc.weight"):
            self.assertIn(key, keys)
        self.assertEqual(len([k for k in keys if k.endswith("num_batches_tracked")]), 20)


if __name__ == "__main__":
    unittest.main()
