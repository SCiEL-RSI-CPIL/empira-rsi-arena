"""Tests of the Eureka BallBalance/Quadcopter analogues (sim_control_eureka*, sandboxed reward training)."""
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

try:
    import mujoco  # noqa: F401
    HAVE_MUJOCO = True
except ImportError:  # pragma: no cover - environment without MuJoCo
    HAVE_MUJOCO = False

from autoresearch_suite.runtime import CandidateError  # noqa: E402

TINY = {"iterations": 2, "population": 4, "elite": 2, "episodes": 1, "eval_episodes": 2}


def module_for(source, tmp):
    path = Path(tmp) / "candidate.py"
    path.write_text(source)
    namespace = {}
    exec(compile(source, str(path), "exec"), namespace)
    return types.SimpleNamespace(__file__=str(path), **{k: v for k, v in namespace.items() if k == "reward"})


@unittest.skipUnless(HAVE_MUJOCO, "mujoco is not installed")
class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        from autoresearch_suite import sim_control_eureka_envs as envs
        from autoresearch_suite import sim_control_eureka_views as views
        self.envs, self.views = envs, views

    def test_views_and_simulator_share_constants(self):
        self.assertAlmostEqual(self.views.TRAY_HEIGHT, self.envs.TRAY_HEIGHT, places=12)
        self.assertAlmostEqual(self.envs.TRAY_HEIGHT, 0.5591, places=4)  # Isaac: leg_length * sqrt(2) + 2 leg_radius + thickness / 2
        self.assertEqual(self.views.BALL_RADIUS, self.envs.BALL_RADIUS)
        for task, env_class in self.envs.ENVS.items():
            env = env_class()
            state = env.reset(np.random.default_rng(0))
            self.assertEqual(len(state), self.views.STATE_DIMS[task])
            self.assertEqual(len(env.observe(state)), env.obs_dim)
            self.assertEqual(env.act_dim, self.views.ACTION_DIMS[task])

    def test_resets_follow_the_isaac_distributions_and_are_deterministic(self):
        tray, quad = self.envs.TrayEnv(), self.envs.QuadEnv()
        for seed in range(20):
            a = tray.reset(np.random.default_rng(seed))
            self.assertTrue(np.array_equal(a, tray.reset(np.random.default_rng(seed))))
            ball, velocity = a[4:7], a[7:10]
            self.assertLessEqual(math.hypot(ball[0], ball[1]), 0.5 + 1e-9)
            self.assertTrue(1.0 <= ball[2] <= 2.0)
            self.assertAlmostEqual(velocity[2], -5.0)
            self.assertLessEqual(math.hypot(velocity[0], velocity[1]), 5.0 + 1e-9)
            q = quad.reset(np.random.default_rng(seed))
            self.assertTrue(abs(q[0]) <= 1.5 and abs(q[1]) <= 1.5 and 0.8 <= q[2] <= 2.5)
            self.assertTrue(np.all(np.abs(q[13:21]) <= 0.2) and np.all(q[21:25] == 0.0))

    def test_quadcopter_ignores_the_first_action_and_integrates_thrust(self):
        quad = self.envs.QuadEnv()
        quad.reset(np.random.default_rng(1))
        action = np.zeros(12)
        action[8:12] = 1.0
        quad.step(action)
        self.assertTrue(np.all(quad.thrusts == 0.0))  # Isaac zeroes thrust on the step after a reset
        quad.step(action)
        self.assertTrue(np.allclose(quad.thrusts, 2.0))  # 0.01 s * 200 * 1 = 2 N, the clamp
        self.assertAlmostEqual(sum(quad.model.body_mass), 0.2515, places=3)

    def test_episodes_end_when_the_ball_falls(self):
        tray = self.envs.TrayEnv()
        tray.reset(np.random.default_rng(3))
        done, failed = False, False
        while not done:
            state, failed, done = tray.step(np.array([1.0, 1.0]))  # tilt fully: the ball rolls off
        self.assertTrue(failed)
        self.assertLess(state[6], 0.15)
        self.assertLess(tray.progress, 500)

    def test_instability_guard(self):
        guard = self.envs._Guard.__new__(self.envs._Guard)
        guard.index, guard.seen, guard.events = 0, 0, 0
        guard.data = types.SimpleNamespace(warning=[types.SimpleNamespace(number=0)])
        self.assertFalse(guard.tripped(np.zeros(3)))
        guard.data.warning[0].number = 1
        self.assertTrue(guard.tripped(np.zeros(3)))
        self.assertTrue(guard.tripped(np.array([np.nan])))
        self.assertEqual(guard.events, 2)

    def test_reference_rewards_match_the_isaac_formulas(self):
        from autoresearch_suite import sim_control_eureka as SE
        for task, name in (("ballbalance", "ballbalance_human"), ("quadcopter", "quadcopter_human")):
            namespace = {}
            exec(SE.REFERENCE_REWARDS[name], namespace)
            env = self.envs.ENVS[task]()
            observe, info = self.views.VIEWS[task]
            state = env.reset(np.random.default_rng(5))
            for _ in range(30):
                state, _, done = env.step(np.full(env.act_dim, 0.3))
                ours = namespace["reward"](observe(state), np.zeros(env.act_dim), observe(state), info(state, 1, False))
                self.assertAlmostEqual(ours, env.human_reward(state), places=12)
                if done:
                    break


@unittest.skipUnless(HAVE_MUJOCO, "mujoco is not installed")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        from autoresearch_suite import sim_control_eureka as SE
        self.SE = SE

    def test_sandboxed_training_equals_in_process_training(self):
        for task, name in (("ballbalance", "ballbalance_human"), ("quadcopter", "quadcopter_human")):
            P = self.SE.parameters({"task": task, **TINY, "episodes": 2})
            namespace = {}
            exec(self.SE.REFERENCE_REWARDS[name], namespace)
            local, _ = self.SE.train_local(namespace["reward"], P)
            sandboxed, oracle = self.SE.train(self.SE.REFERENCE_REWARDS[name], P)
            self.assertTrue(np.array_equal(local, sandboxed))
            self.assertEqual(len(oracle.history), P["iterations"])

    def test_scores_are_deterministic(self):
        source = self.SE.REFERENCE_REWARDS["ballbalance_human"]
        a, details = self.SE.score_source(source, {"task": "ballbalance", **TINY})
        b, _ = self.SE.score_source(source, {"task": "ballbalance", **TINY})
        self.assertEqual(a, b)
        self.assertTrue(1 <= a <= 500)
        self.assertEqual(details["eval_episodes"], 2)

    def test_invalid_rewards_are_rejected(self):
        bad = {
            "nan": "def reward(o, a, n, i):\n    return float('nan')\n",
            "string": "def reward(o, a, n, i):\n    return 'high'\n",
            "vector": "import numpy as np\n\ndef reward(o, a, n, i):\n    return np.ones(3)\n",
            "raises": "def reward(o, a, n, i):\n    raise KeyError('missing')\n",
            "reads_files": "def reward(o, a, n, i):\n    return float(len(open('/etc/hosts').read()))\n",
            "huge": "def reward(o, a, n, i):\n    return 1e300\n",
        }
        for name, source in bad.items():
            with self.subTest(name=name), self.assertRaises(CandidateError):
                self.SE.score_source(source, {"task": "ballbalance", **TINY})

    def test_reserved_names_and_missing_reward(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            module = module_for("def reward(o, a, n, i):\n    return 0.0\n# __sim_control_reward_driver__\n", tmp)
            with self.assertRaises(CandidateError):
                self.SE.score(module, {"task": "ballbalance", **TINY})
            with self.assertRaises(CandidateError):
                self.SE.score(types.SimpleNamespace(__file__=module.__file__), {"task": "ballbalance", **TINY})

    def test_candidate_cannot_reach_the_simulator(self):
        # The reward process only receives transitions; module globals of the evaluator are not reachable.
        probe = ("import sys\n\ndef reward(o, a, n, i):\n"
                 "    assert 'autoresearch_suite' not in sys.modules and 'mujoco' not in sys.modules\n    return 0.0\n")
        value, _ = self.SE.score_source(probe, {"task": "ballbalance", **TINY})
        self.assertGreater(value, 0)


if __name__ == "__main__":
    unittest.main()
