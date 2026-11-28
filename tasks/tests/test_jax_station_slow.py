"""Child-process tests of the jax_station evaluators; skipped without JAX/XLand-MiniGrid or the prepared data.

Each test takes 10-60 s (a child imports JAX), so the child tests run only with ARENA_SLOW_TESTS=1.
Full evaluations (minutes of PPO or ZAPBench training) run only with ARTASK_JAX_STATION_FULL=1; their
recorded results are in validation/jax_station-checks.json.

Ported from the source repository. Changes: prepared data are looked up in the task cache; the slow-test
gate; the tests that pass a serialised predictor between processes need flatbuffers (jax.export imports it)
and are skipped without it; the scoring child accepts single-window predictors only, so the published-baseline
test exports [1, 4, N] -> [1, 32, N] instead of a chunk of eight windows.
"""
import importlib.util
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from suite_test_support import slow  # noqa: E402

from autoresearch_suite import jax_station_common as common  # noqa: E402
from autoresearch_suite import jax_station_limen as limen  # noqa: E402
from autoresearch_suite import jax_station_zap as zap  # noqa: E402
from autoresearch_suite import jax_station_zapdata as zd  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

HAVE_JAX = all(importlib.util.find_spec(name) for name in ("jax", "flax", "optax"))
HAVE_LIMEN = HAVE_JAX and all(importlib.util.find_spec(n) for n in ("xminigrid", "distrax")) and \
    (limen.limen_dir() / "limen" / "evaluator.py").is_file()
HAVE_ZAP = HAVE_JAX and all((zd.zap_dir() / name).is_file() for pair in zd.FILES.values() for name in pair)
HAVE_FLATBUFFERS = importlib.util.find_spec("flatbuffers") is not None
NEEDS_FLATBUFFERS = unittest.skipUnless(HAVE_FLATBUFFERS, "flatbuffers is not installed (jax.export serialisation needs it)")
FULL = os.environ.get("ARTASK_JAX_STATION_FULL") == "1"
PUBLISHED_MEAN_BASELINE = 0.031859533156597224  # ZAPBench combined.json, method "mean", context 4, 9 conditions x 32 steps


def module_from(source):
    path = Path(tempfile.mkdtemp()) / "candidate.py"
    path.write_text(textwrap.dedent(source))
    module = types.ModuleType("candidate")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


@unittest.skipUnless(HAVE_LIMEN, "needs jax, flax, distrax, xminigrid and scripts/prepare_jax_station.py --limen")
class LimenChildTest(unittest.TestCase):
    @slow
    def test_crash_filter_rejects_a_two_dimensional_observation(self):
        module = module_from("""
            import jax.numpy as jnp
            def get_observation(state):
                return state.grid.astype(jnp.float32)
            def compute_reward(state, action, next_state):
                return jnp.float32(0.0)
        """)
        with self.assertRaisesRegex(CandidateError, "crash filter"):
            limen.score(module, {"task": "easy_pickup", "wall_seconds": 300})

    @unittest.skipUnless(FULL, "full PPO evaluation; set ARTASK_JAX_STATION_FULL=1")
    def test_published_easy_interface(self):
        source = (limen.limen_dir() / "examples" / "evolved_interfaces" / "easy_pickup.py").read_text()
        value, details = limen.score(module_from(source), {"task": "easy_pickup", "wall_seconds": 840})
        self.assertGreaterEqual(value, 0.9)
        self.assertEqual(details["stage"], "full_train")


@unittest.skipUnless(HAVE_ZAP, "needs jax, flax, optax and scripts/prepare_jax_station.py --zapbench")
class ZapChildTest(unittest.TestCase):
    @slow
    @NEEDS_FLATBUFFERS
    def test_scoring_child_reproduces_the_published_mean_baseline(self):
        import jax
        import jax.numpy as jnp
        import numpy as np

        def predictor(leaves, x):
            return jnp.repeat(jnp.mean(x, axis=1, keepdims=True), 32, axis=1) + 0.0 * leaves[0]

        out = Path(tempfile.mkdtemp())
        specs = [jax.ShapeDtypeStruct((1,), jnp.float32)]
        exported = jax.export.export(jax.jit(predictor), platforms=("cpu",))(specs, jax.ShapeDtypeStruct((1, 4, zd.NUM_NEURONS), jnp.float32))
        (out / "predictor.bin").write_bytes(exported.serialize())
        np.savez(out / "params.npz", np.zeros(1, np.float32))
        result, _ = common.run_child(zap.SCORE_WORKER, {"out_dir": str(out), "test_dir": str(zd.zap_dir() / "test"),
                                                        "cpu_seconds": 600}, wall_seconds=300, memory_mb=6000)
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["test_mae"], PUBLISHED_MEAN_BASELINE, delta=1e-7)
        self.assertEqual(result["test_windows"], 1557)

    def test_module_check_accepts_lapack_custom_calls(self):
        import jax
        import jax.numpy as jnp
        from autoresearch_suite import jax_station_zap_score as zap_score
        exported = jax.export.export(jax.jit(lambda x: jnp.linalg.inv(x @ x.T + jnp.eye(4)) @ x))(
            jax.ShapeDtypeStruct((4, 3), jnp.float32))
        text = exported.mlir_module()
        self.assertIn("custom_call @lapack_", text)
        self.assertEqual(zap_score.check_module(text), [])
        self.assertTrue(zap_score.check_module(text.replace("@lapack_sgetrf_ffi", "@my_target")))

    @slow
    @NEEDS_FLATBUFFERS
    def test_training_child_cannot_read_the_test_split(self):
        test_file = str(zd.zap_dir() / "test" / "test.npy")
        module = module_from(f"""
            import numpy as np
            LEAK = np.load({test_file!r}, mmap_mode="r")
        """)
        with self.assertRaisesRegex(CandidateError, "denied"):
            zap.score(module, {"max_epochs": 1, "train_wall_seconds": 300})

    @unittest.skipUnless(FULL, "ZAPBench training; set ARTASK_JAX_STATION_FULL=1")
    @NEEDS_FLATBUFFERS
    def test_seed_one_epoch(self):
        from autoresearch_suite import jax_station_zapseed
        value, details = zap.score(module_from(jax_station_zapseed.SEED), {"max_epochs": 1})
        self.assertLess(value, 0.04)
        self.assertEqual(details["epochs"], 1)


if __name__ == "__main__":
    unittest.main()
