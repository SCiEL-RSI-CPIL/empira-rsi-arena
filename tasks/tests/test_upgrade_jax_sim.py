"""Regression tests of the jax_station and sim_control evaluators (LIMEN, ZAPBench, Sodarace, Push-T, BallBalance).

Each test pins a defect found when the packages were brought into this library, or a property the audit relied on:
the wrappers hand the candidate over unexecuted; LIMEN's static interface policy cannot be sidestepped through module
aliases, frame objects, class patterns, dunder names or system modules; the ZAPBench scorer feeds one test window per
call (a batch of overlapping windows leaks held-out targets) and reports a missing package as a defect of the
environment; booleans and strings are not rewards, positions or coordinates; and the scoring arithmetic of each family
agrees with an independent computation. Tests that need an optional package, prepared data or more than about twenty
seconds are skipped with the reason.
"""
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from suite_test_support import needs, seed_source, slow  # noqa: E402
from autoresearch_suite import jax_station_limen as limen  # noqa: E402
from autoresearch_suite import jax_station_policy as policy  # noqa: E402
from autoresearch_suite import jax_station_zap as zap  # noqa: E402
from autoresearch_suite import jax_station_zap_score as zap_score  # noqa: E402
from autoresearch_suite import jax_station_zapdata as zd  # noqa: E402
from autoresearch_suite import runtime, sandboxed  # noqa: E402
from autoresearch_suite import sim_control_eureka_views as views  # noqa: E402
from autoresearch_suite import sim_control_pusht as pusht  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

TASKS = ("limen-easy-pickup", "limen-medium-place-near", "limen-hard-rule-chain",
         "the-station-zapbench-neural-activity-prediction", "elm-sodarace-walker-program-evolution",
         "enpire-gym-pusht-simulation", "eureka-ballbalance")
SODARACE, PUSHT = TASKS[4], TASKS[5]
HAVE_LIMEN = all(importlib.util.find_spec(n) for n in ("jax", "flax", "distrax", "xminigrid")) and \
    (limen.limen_dir() / "limen" / "evaluator.py").is_file()
HAVE_TEST_SPLIT = bool(importlib.util.find_spec("jax")) and all((zd.zap_dir() / name).is_file() for name in zd.FILES["test"])
HAVE_ZAP = all((zd.zap_dir() / name).is_file() for pair in zd.FILES.values() for name in pair) and \
    all(importlib.util.find_spec(n) for n in ("jax", "flax", "optax", "flatbuffers"))
HAVE_PUSHT = pusht.env_python().is_file()
PUBLISHED_MEAN_BASELINE = 0.031859533156597224  # ZAPBench combined.json, method "mean", context 4, 9 conditions x 32 steps
# What a hostile candidate does while it is imported: leave a mark and replace the scoring function. It then stops, so
# that wherever the file is executed the evaluation ends at once.
PROBE = '''
import pathlib
pathlib.Path({marker!r}).write_text("ran")
try:
    from autoresearch_suite import {module}, runtime
    runtime.TAMPERED = True
    {module}.score = lambda candidate, parameters: (1e9, {{}})
except Exception:
    pass
raise RuntimeError("the probe ran")
'''
MEAN_OF_CONTEXT = '''import jax.numpy as jnp

BATCH_SIZE = 8


class MeanOfContext:
    def init(self, rng_key, dummy_input):
        return {"w": jnp.zeros((1,), jnp.float32)}

    def apply(self, params, x, training=False, mutable=None, rngs=None):
        return jnp.repeat(jnp.mean(x, axis=1, keepdims=True), 32, axis=1) + 0.0 * params["w"]


def create_network(hparams):
    return MeanOfContext()
'''
# Replaces the training child's predictor by one that copies targets from the neighbouring windows of a batch.
POOLING = '''

def _pooled(network, treedef):
    import jax

    def predictor(leaves, x):
        params = jax.tree_util.tree_unflatten(treedef, leaves)
        out = jax.vmap(lambda window: network.apply(params, window[None], training=False)[0])(x)
        for k in range(1, x.shape[0]):
            out = out.at[:x.shape[0] - k, k - 1].set(x[k:, 3])
        return out.astype(jnp.float32)

    return predictor


import __main__
__main__.make_predictor = _pooled
'''


def write(source):
    path = Path(tempfile.mkdtemp()) / "candidate.py"
    path.write_text(source)
    return path


def wrapper(test, task):
    """The package's evaluator.py as a module; the test is skipped when its library needs a package that is absent."""
    spec = importlib.util.spec_from_file_location(f"evaluator_{task.replace('-', '_')}", ROOT / "suite" / task / "evaluator.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        test.skipTest(f"{task}: {error}")
    return module


class WrapperTests(unittest.TestCase):
    def test_wrappers_load_the_candidate_unexecuted(self):
        for task in TASKS:
            text = (ROOT / "suite" / task / "evaluator.py").read_text()
            self.assertIn("sandboxed.run(program_path, task", text, task)
            self.assertNotIn("runtime.run(", text, task)

    def test_module_level_code_never_runs_in_the_evaluator(self):
        self.addCleanup(lambda: runtime.__dict__.pop("TAMPERED", None))
        for task in TASKS:
            with self.subTest(task=task):
                marker = Path(tempfile.mkdtemp()) / "marker"
                library = re_module((ROOT / "suite" / task / "evaluator.py").read_text())
                candidate = write(seed_source(task) + PROBE.format(marker=str(marker), module=library))
                result = wrapper(self, task).evaluate(str(candidate))
                self.assertFalse(marker.exists(), "the candidate was executed in the evaluator's process")
                self.assertFalse(hasattr(runtime, "TAMPERED"))
                self.assertIsNone(result["score"])
                self.assertNotEqual(result.get("combined_score"), 1e9)

    def test_the_probe_would_run_if_the_candidate_were_imported(self):
        marker = Path(tempfile.mkdtemp()) / "marker"
        self.addCleanup(lambda: (runtime.__dict__.pop("TAMPERED", None), policy.__dict__.pop("score", None)))
        candidate = write(seed_source(SODARACE) + PROBE.format(marker=str(marker), module="jax_station_policy"))
        with self.assertRaisesRegex(RuntimeError, "the probe ran"):
            runtime.load_program(candidate)
        self.assertTrue(marker.exists() and getattr(runtime, "TAMPERED", False))

    @needs("Box2D")
    def test_a_candidate_that_tampers_at_import_gains_nothing(self):
        tamper = ("try:\n    from autoresearch_suite import sim_control_sodarace\n"
                  "    sim_control_sodarace.score_walker = lambda walker, P: (1e9, {})\nexcept Exception:\n    pass\n")
        evaluate = wrapper(self, SODARACE).evaluate
        honest, hostile = evaluate(str(write(seed_source(SODARACE)))), evaluate(str(write(tamper + seed_source(SODARACE))))
        self.assertEqual((honest["valid"], hostile["valid"]), (True, True))
        self.assertEqual(hostile["score"], honest["score"])
        self.assertLess(hostile["score"], 1.0)


def re_module(text):
    """The family module a wrapper calls (the last ``from autoresearch_suite import`` line)."""
    return [line.split()[3] for line in text.splitlines() if line.startswith("from autoresearch_suite import ")][-1]


class LimenPolicyTests(unittest.TestCase):
    BYPASSES = {
        "alias": "import jax\nj = jax\nj.config.update('jax_enable_x64', True)\n",
        "argument": "import jax\ndef f(m):\n    return m.config\nf(jax)\n",
        "container": "import jax\nfor m in [jax]:\n    m.experimental\n",
        "default": "import jax\ndef f(m=jax):\n    return m.extend\n",
        "walrus": "import jax\n(j := jax).config\n",
        "submodule alias": "from jax import numpy as jnp\nn = jnp\nn.lib\n",
        "generator frame": "g = (i for i in [1])\nb = g.gi_frame.f_builtins\n",
        "frame chain": "def f(frame):\n    return frame.f_back.f_globals\n",
        "class pattern": "def f():\n    pass\nmatch f:\n    case object(__globals__=g):\n        pass\n",
        "loader": "x = __loader__\n",
        "spec": "y = __spec__\n",
        "system module": "from jax import collect_profile\nq = collect_profile.os\n",
    }
    HONEST = {
        "list.extend": "import jax.numpy as jnp\ndef f(x):\n    out = []\n    out.extend([x])\n    return jnp.stack(out)\n",
        "private helper": "import jax.numpy as jnp\n_C = jnp.array([1])\ndef _h(x):\n    return jnp.sum(x) + _C\n",
        "function as argument": "import jax\nimport jax.numpy as jnp\ny = jax.tree_util.tree_map(jnp.asarray, [1])\n",
        "imported functions": "from jax.numpy import where\nfrom jax import lax\ndef f(x):\n    return lax.select(x > 0, x, where(x > 0, x, -x))\n",
        "module name": "import math\nx = math.pi if __name__ else 0.0\n",
    }
    FORGED = ("import jax\nimport jax.numpy as jnp\n\n\ndef get_observation(state):\n    return jnp.zeros(4)\n\n\n"
              "def compute_reward(state, action, next_state):\n    return jnp.float32(0.0)\n\n\ndef _reach():\n"
              "    table = (i for i in [1]).gi_frame.f_builtins\n    modules = table['__import__']('sys').modules\n"
              "    adapter = modules['limen.adapters.xminigrid'].XMinigridAdapter\n"
              "    table['setattr'](adapter, 'compute_success', lambda self, stats, params: jnp.ones_like(stats.length, dtype=jnp.float32))\n"
              "\n\n_reach()\n")

    def test_routes_to_builtins_and_module_internals_are_rejected(self):
        for name, source in self.BYPASSES.items():
            with self.subTest(name=name):
                self.assertTrue(policy.check_interface_source(source), source)

    def test_honest_interfaces_are_accepted(self):
        for name, source in self.HONEST.items():
            with self.subTest(name=name):
                self.assertEqual(policy.check_interface_source(source), [])
        for task in TASKS[:3]:
            self.assertEqual(policy.check_interface_source(seed_source(task)), [], task)

    def test_published_interfaces_are_accepted(self):
        examples = sorted((limen.limen_dir() / "examples" / "evolved_interfaces").glob("*.py"))
        if not examples:
            self.skipTest("LIMEN is not prepared (scripts/prepare_jax_station.py --limen)")
        for path in examples:
            self.assertEqual(policy.check_interface_source(path.read_text()), [], path.name)

    def test_a_forged_success_measure_is_stopped_before_any_process_starts(self):
        original = limen.run_limen
        self.addCleanup(lambda: setattr(limen, "run_limen", original))
        limen.run_limen = lambda source, P: self.fail("the candidate reached LIMEN")
        with self.assertRaisesRegex(CandidateError, "interface policy.*gi_frame"):
            limen.score(sandboxed.source_program(write(self.FORGED)), {"task": "easy_pickup", "wall_seconds": 10})

    def test_the_source_of_an_unexecuted_candidate_is_checked(self):
        good = "import jax.numpy as jnp\ndef get_observation(s):\n    return jnp.zeros(3)\ndef compute_reward(s, a, n):\n    return 0.0\n"
        self.assertEqual(limen.candidate_source(sandboxed.source_program(write(good))), good)
        with self.assertRaisesRegex(CandidateError, "must define compute_reward"):
            limen.candidate_source(sandboxed.source_program(write("def get_observation(s):\n    return s\n")))
        with self.assertRaisesRegex(CandidateError, "does not parse"):
            limen.candidate_source(sandboxed.source_program(write("def get_observation(:\n")))

    @slow
    @unittest.skipUnless(HAVE_LIMEN, "needs jax, flax, distrax, xminigrid and scripts/prepare_jax_station.py --limen")
    def test_short_run_from_candidate_to_score(self):
        # One PPO update of 256 environments instead of LIMEN's budget: the path, not the task's score.
        overrides = {"training": {"num_envs": 256, "total_timesteps": 4096, "total_timesteps_full": 4096, "num_seeds": 1, "eval_episodes": 8}}
        candidate = sandboxed.source_program(ROOT / "suite" / TASKS[0] / "initial_program.py")
        value, details = limen.score(candidate, {"task": "easy_pickup", "wall_seconds": 90, "overrides": overrides})
        self.assertTrue(0.0 <= value <= 1.0)
        self.assertEqual((details["stage"], details["fitness_key"], details["obs_dim"]), ("full_train", "success_rate", 171))
        self.assertEqual([run["total_timesteps"] for run in details["runs"]], [4096, 4096])
        self.assertEqual(value, details["runs"][-1]["success_rate"])


class ZapScoringTests(unittest.TestCase):
    def test_only_single_window_predictors_are_accepted(self):
        self.assertEqual(zap_score.check_shapes((1, 4, 71721), (1, 32, 71721)), 71721)
        for shapes in (((8, 4, 71721), (8, 32, 71721)), ((1, 4, 71721), (1, 31, 71721)), ((1, 5, 9), (1, 32, 9)),
                       ((1, 4, 9), (1, 32, 8)), ((4, 9), (32, 9)), ((1, 4, 9), (32, 9))):
            with self.subTest(shapes=shapes), self.assertRaises(zap_score.ContractError):
                zap_score.check_shapes(*shapes)

    def test_metric_against_an_independent_computation(self):
        rng = np.random.default_rng(7)
        labels = np.array([0] * 50 + [5] * 40 + [3] * 45 + [8] * 30, dtype=np.int8)  # condition 8 is too short for a window
        data = rng.normal(size=(len(labels), 6)).astype(np.float32)
        calls = []

        def predict(x):
            calls.append(x.shape)
            return np.repeat(x[:, -1:] * 0.5, 32, axis=1) + np.arange(32, dtype=np.float32)[None, :, None] * 0.01

        result = zap_score.evaluate(predict, data, labels)
        errors, pooled = {}, []
        for condition in (0, 5, 3):
            rows = data[labels == condition]
            windows = [rows[i:i + 36] for i in range(len(rows) - 35)]
            absolute = [np.abs(predict(w[None, :4])[0] - w[4:]).astype(np.float64) for w in windows]  # float32 errors, summed exactly
            errors[condition] = np.mean([a.mean(axis=1) for a in absolute], axis=0)
            pooled += [a.mean() for a in absolute]
        self.assertEqual(result["test_windows"], 15 + 5 + 10)
        self.assertEqual(set(calls), {(1, 4, 6)})
        self.assertEqual(result["condition_windows"], {"0": 15, "5": 5, "3": 10})
        self.assertAlmostEqual(result["test_mae"], float(np.mean([errors[c].mean() for c in errors])), places=9)
        self.assertAlmostEqual(result["pooled_mae"], float(np.mean(pooled)), places=9)
        self.assertNotAlmostEqual(result["test_mae"], result["pooled_mae"], places=4)  # conditions weigh equally, windows do not
        self.assertAlmostEqual(result["holdout_mae"], float(errors[3].mean()), places=9)
        self.assertAlmostEqual(result["train_conditions_mae"], float(np.mean([errors[0].mean(), errors[5].mean()])), places=9)
        for step in (1, 4, 8, 16, 32):
            self.assertAlmostEqual(result["step_mae"][str(step)], float(np.mean([errors[c][step - 1] for c in errors])), places=9)

    def test_invalid_predictions_are_the_candidates_failure(self):
        labels, data = np.zeros(40, dtype=np.int8), np.ones((40, 3), dtype=np.float32)
        for name, bad in (("nan", np.full((1, 32, 3), np.nan, np.float32)), ("inf", np.full((1, 32, 3), np.inf, np.float32)),
                          ("shape", np.zeros((1, 32, 2), np.float32)), ("batch", np.zeros((2, 32, 3), np.float32))):
            with self.subTest(name=name), self.assertRaises(zap_score.ContractError):
                zap_score.evaluate(lambda x: bad, data, labels)

    def test_a_missing_package_is_not_the_candidates_failure(self):
        self.assertIsNone(policy.missing_modules(("json", "ast")))
        self.assertIn("no_such_package_for_this_test", policy.missing_modules(("json", "no_such_package_for_this_test")))
        self.assertIn("flatbuffers", zap_score.REQUIRED_MODULES)
        with self.assertRaisesRegex(RuntimeError, "cannot run.*flatbuffers"):
            zap._checked({"ok": False, "environment": True, "error": "the evaluator's interpreter lacks flatbuffers"}, "training")
        with self.assertRaisesRegex(CandidateError, "training failed: ValueError"):
            zap._checked({"ok": False, "contract": False, "error": "ValueError: bad shape"}, "training")
        zap._checked({"ok": True}, "training")

    @slow
    @unittest.skipUnless(HAVE_TEST_SPLIT, "needs jax and the prepared ZAPBench test split (scripts/prepare_jax_station.py --zapbench)")
    def test_published_mean_baseline_one_window_per_call(self):
        import jax
        import jax.numpy as jnp
        from jax import export

        def mean_of_context(leaves, x):
            return jax.vmap(lambda w: jnp.repeat(jnp.mean(w, axis=0, keepdims=True), 32, axis=0) + 0.0 * leaves[0])(x)

        specs = [jax.ShapeDtypeStruct((1,), jnp.float32)]
        exported = export.export(jax.jit(mean_of_context), platforms=("cpu",))(specs, jax.ShapeDtypeStruct((1, 4, zd.NUM_NEURONS), jnp.float32))
        self.assertEqual(zap_score.check_shapes(exported.in_avals[-1].shape, exported.out_avals[0].shape), zd.NUM_NEURONS)
        chunked = export.export(jax.jit(mean_of_context), platforms=("cpu",))(specs, jax.ShapeDtypeStruct((8, 4, zd.NUM_NEURONS), jnp.float32))
        with self.assertRaises(zap_score.ContractError):  # eight overlapping windows per call could copy each other's targets
            zap_score.check_shapes(chunked.in_avals[-1].shape, chunked.out_avals[0].shape)
        data, labels = zd.load_split("test")
        call, leaves = jax.jit(exported.call), [np.zeros(1, np.float32)]
        result = zap_score.evaluate(lambda x: call(leaves, x), data, labels)
        self.assertAlmostEqual(result["test_mae"], PUBLISHED_MEAN_BASELINE, delta=1e-7)
        self.assertEqual((result["test_windows"], len(result["condition_mae"])), (1557, 9))

    @slow
    @unittest.skipUnless(HAVE_ZAP, "needs jax, flax, optax, flatbuffers and scripts/prepare_jax_station.py --zapbench")
    def test_a_predictor_that_pools_a_batch_gains_nothing(self):
        budget = {"max_epochs": 1, "train_wall_seconds": 600}
        honest, _ = zap.score(sandboxed.source_program(write(MEAN_OF_CONTEXT)), budget)
        hostile, _ = zap.score(sandboxed.source_program(write(MEAN_OF_CONTEXT + POOLING)), budget)
        self.assertAlmostEqual(honest, PUBLISHED_MEAN_BASELINE, delta=1e-7)
        self.assertAlmostEqual(hostile, honest, delta=1e-9)


class OutputTypeTests(unittest.TestCase):
    """Booleans, strings (also numeric ones) and nested values are not numbers a candidate may return."""

    def test_rewards_are_real_scalars(self):
        for value in (1, 2.5, np.float32(0.5), np.int64(3), np.array(1.5)):
            self.assertEqual(views.real_number(value), float(value))
        for value in (True, np.True_, "1.5", b"1", 1 + 0j, [1.0], np.ones(1), np.ones(3), None):
            with self.subTest(value=value), self.assertRaises(views.RewardContractError):
                views.real_number(value)

    @needs("mujoco")
    def test_wrong_type_rewards_are_rejected(self):
        from autoresearch_suite import sim_control_eureka as SE
        tiny = {"task": "ballbalance", "iterations": 2, "population": 4, "elite": 2, "episodes": 1, "eval_episodes": 2}
        for name, value in (("bool", "True"), ("numeric string", "'1.5'"), ("list", "[1.0]")):
            with self.subTest(name=name), self.assertRaisesRegex(CandidateError, "real number"):
                SE.score_source(f"def reward(o, a, n, i):\n    return {value}\n", tiny)

    @unittest.skipUnless(HAVE_PUSHT, "Push-T runtime missing (python3 scripts/prepare_sim_control.py --setup-environment)")
    def test_wrong_type_actions_are_rejected(self):
        for name, value in (("booleans", "True, False"), ("numeric strings", "'256', '256'"), ("nested", "[[256.0], [256.0]]"),
                            ("matrix", "__import__('numpy').zeros((2, 1))"), ("scalar", "256.0")):
            source = f"class Policy:\n    def act(self, observation):\n        return {value}\n"
            with self.subTest(name=name), self.assertRaisesRegex(CandidateError, "two numbers"):
                pusht.score(sandboxed.source_program(write(source)), {"episodes": 1, "first_seed": 0, "horizon": 5})
        array = "import numpy as np\n\nclass Policy:\n    def act(self, observation):\n        return np.array([256, 256])\n"
        value, _ = pusht.score(sandboxed.source_program(write(array)), {"episodes": 1, "first_seed": 0, "horizon": 5})
        self.assertEqual(value, 0.0)

    @needs("Box2D")
    def test_wrong_type_walkers_are_rejected(self):
        from autoresearch_suite import sim_control_sodarace as S
        bad = {"string coordinates": "Walker([('0', '0'), ('1.0', '0')], [[0, 1, {'amplitude': 1.0, 'phase': 0.5}]])",
               "boolean coordinates": "Walker([(False, False), (True, False)], [[0, 1, {'amplitude': 1.0, 'phase': 0.5}]])",
               "boolean amplitude": "Walker([(0.0, 0.0), (1.0, 0.0)], [[0, 1, {'amplitude': True, 'phase': 0.5}]])",
               "string phase": "Walker([(0.0, 0.0), (1.0, 0.0)], [[0, 1, {'amplitude': 1.0, 'phase': '0.5'}]])",
               "boolean index": "Walker([(0.0, 0.0), (1.0, 0.0)], [[False, True, {'amplitude': 1.0, 'phase': 0.5}]])",
               "float index": "Walker([(0.0, 0.0), (1.0, 0.0)], [[0.0, 1.0, {'amplitude': 1.0, 'phase': 0.5}]])"}
        for name, walker in bad.items():
            with self.subTest(name=name), self.assertRaisesRegex(CandidateError, "must be"):
                S.score(sandboxed.source_program(write(f"def make_walker():\n    return {walker}\n")), {"updates": 2})


class ScoringArithmeticTests(unittest.TestCase):
    @needs("Box2D")
    def test_sodarace_score_is_the_displacement_of_the_centre_of_mass(self):
        from Box2D import Box2D as b2
        from autoresearch_suite import sim_control_sodarace as S
        from autoresearch_suite import sim_control_sodarace_sim as sim
        walker = S.build_walker(seed_source(SODARACE))
        value, details = S.score_walker(walker, {})
        simulator = sim.SodaraceSimulator(S.validate(walker))

        def centre():
            nodes = [body for body in simulator.world.world.bodies if body.type == b2.b2_dynamicBody]
            return sum(body.mass * body.position[0] for body in nodes) / sum(body.mass for body in nodes)

        start = centre()
        steps = sum(simulator.world.update_world(350) for _ in range(80))
        self.assertEqual((steps, details["physics_steps"]), (1680, 1680))  # 80 updates of 21 steps of 1/60 s = 28 s
        self.assertAlmostEqual(value, abs(centre() - start), places=9)
        self.assertAlmostEqual(details["signed_displacement"], centre() - start, places=5)
        self.assertEqual(S.score_walker(walker, {})[0], value)

    @unittest.skipUnless(HAVE_PUSHT, "Push-T runtime missing (python3 scripts/prepare_sim_control.py --setup-environment)")
    def test_pusht_success_is_coverage_above_95_percent(self):
        value, details = pusht.score(sandboxed.source_program(ROOT / "suite" / PUSHT / "initial_program.py"), {"episodes": 5, "first_seed": 0})
        records = details["per_episode"]
        self.assertEqual([record["success"] for record in records], [True, True, False, False, True])
        for record in records:
            self.assertEqual(record["success"], record["max_coverage"] > 0.95)
            self.assertEqual(record["steps"] < 300, record["success"])  # an episode ends at its first success or after 300 steps
            if record["success"]:
                self.assertEqual(record["final_coverage"], record["max_coverage"])
        self.assertEqual(value, sum(record["success"] for record in records) / 5)

    @needs("mujoco")
    def test_ballbalance_fitness_is_the_mean_episode_length(self):
        from autoresearch_suite import sim_control_eureka as SE
        from autoresearch_suite import sim_control_eureka_envs as envs
        P = SE.parameters({"task": "ballbalance"})
        self.assertEqual((P["eval_seed"], P["eval_episodes"], P["train_seed"]), (2026, 50, 0))
        env = envs.TrayEnv()
        value, lengths = SE.evaluate_policy(env, np.zeros((env.obs_dim + 1) * env.act_dim), P)
        counted = []
        for episode in range(50):  # the policy that never tilts the tray, stepped by hand on the same 50 throws
            state, steps = env.reset(np.random.default_rng([2026, 2, episode])), 0
            while True:
                state, failed, done = env.step(np.zeros(2))
                steps += 1
                if done:
                    self.assertEqual(failed, state[6] < 0.15)
                    self.assertTrue(failed or steps == 500)
                    break
            counted.append(steps)
        self.assertEqual(lengths, counted)
        self.assertAlmostEqual(value, sum(counted) / 50, places=12)
        self.assertAlmostEqual(value, 227.76, places=2)  # the source repository's record for this policy
        self.assertTrue(all(1 <= steps <= 500 for steps in counted) and math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
