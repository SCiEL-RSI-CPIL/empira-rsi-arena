"""Tests of the Push-T heuristic-controller evaluator (sim_control_pusht, gym-pusht server, sandboxed policies).

Ported from the source repository; the seed is read from the generated package because the generator module
(scripts/families_sim_control.py) is not part of this library.
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from suite_test_support import seed_source  # noqa: E402
from autoresearch_suite import sim_control_pusht as SP  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

HAVE_RUNTIME = SP.env_python().is_file()


class FakeServer:
    def __init__(self, success_at=None):
        self.calls, self.success_at = [], success_at

    def call(self, cmd, **payload):
        self.calls.append((cmd, payload))
        steps = len(self.calls)
        return {"obs": [1.0, 2.0, 3.0, 4.0, 0.5], "coverage": 0.5, "success": steps == self.success_at,
                "terminated": steps == self.success_at, "truncated": False}


def module_for(source, tmp):
    path = Path(tmp) / "candidate.py"
    path.write_text(source)
    return types.SimpleNamespace(Policy=object, __file__=str(path))


class EpisodeOracleTests(unittest.TestCase):
    def test_actions_are_validated_and_clipped(self):
        server = FakeServer()
        episode = SP.Episode(server, horizon=3)
        reply = episode({"op": "step", "action": [-5.0, 900.0]})
        self.assertEqual(server.calls[0], ("step", {"action": [0.0, 512.0]}))
        self.assertFalse(reply["done"])
        for bad in ([1.0], [1.0, float("nan")], [True, 1.0], "1,2", None):
            with self.assertRaises(CandidateError):
                episode({"op": "step", "action": bad})
        with self.assertRaises(CandidateError):
            episode({"op": "reset"})

    def test_episode_ends_on_success_or_horizon(self):
        episode = SP.Episode(FakeServer(success_at=2), horizon=300)
        episode({"op": "step", "action": [1.0, 1.0]})
        self.assertTrue(episode({"op": "step", "action": [1.0, 1.0]})["done"])
        self.assertTrue(episode.success)
        with self.assertRaises(CandidateError):
            episode({"op": "step", "action": [1.0, 1.0]})
        capped = SP.Episode(FakeServer(), horizon=2)
        capped({"op": "step", "action": [1.0, 1.0]})
        self.assertTrue(capped({"op": "step", "action": [1.0, 1.0]})["done"])
        self.assertFalse(capped.success)


@unittest.skipUnless(HAVE_RUNTIME, "Push-T runtime missing (python3 scripts/prepare_sim_control.py --setup-environment)")
class PushTEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.seed = seed_source("enpire-gym-pusht-simulation")

    def score(self, source, episodes=3):
        with tempfile.TemporaryDirectory() as tmp:
            return SP.score(module_for(source, tmp), {"episodes": episodes, "first_seed": 0})

    def test_seed_is_deterministic(self):
        a, details = self.score(self.seed)
        b, _ = self.score(self.seed)
        self.assertEqual(a, b)
        self.assertEqual(details["episodes"], 3)
        self.assertEqual(details["simulator"]["gym-pusht"], "0.1.6")
        self.assertTrue(details["simulator"]["pymunk"].startswith("6."))
        self.assertTrue(all(1 <= r["steps"] <= 300 for r in details["per_episode"]))

    def test_success_needs_the_goal_pose(self):
        idle = "class Policy:\n    def act(self, observation):\n        return observation[0], observation[1]\n"
        value, details = self.score(idle)
        self.assertEqual(value, 0.0)
        self.assertTrue(all(r["steps"] == 300 for r in details["per_episode"]))

    def test_invalid_policies_are_rejected(self):
        bad = {
            "nan": "class Policy:\n    def act(self, observation):\n        return float('nan'), 1.0\n",
            "three": "class Policy:\n    def act(self, observation):\n        return 1.0, 2.0, 3.0\n",
            "text": "class Policy:\n    def act(self, observation):\n        return 'left'\n",
            "raises": "class Policy:\n    def act(self, observation):\n        raise ValueError('stuck')\n",
            "file": "class Policy:\n    def act(self, observation):\n        open('/etc/hosts').read()\n        return 1.0, 1.0\n",
            "import_env": "import gym_pusht\n\nclass Policy:\n    def act(self, observation):\n        return 1.0, 1.0\n",
        }
        for name, source in bad.items():
            with self.subTest(name=name), self.assertRaises(CandidateError):
                self.score(source, episodes=1)

    def test_policy_state_does_not_leak_between_episodes(self):
        counter = ("COUNT = [0]\n\nclass Policy:\n    def __init__(self):\n        COUNT[0] += 1\n"
                   "        assert COUNT[0] == 1, 'module state leaked'\n"
                   "    def act(self, observation):\n        return observation[0], observation[1]\n")
        value, _ = self.score(counter, episodes=2)
        self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
