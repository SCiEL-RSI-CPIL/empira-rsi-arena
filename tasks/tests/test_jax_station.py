"""Tests of the jax_station evaluators (LIMEN on XLand-MiniGrid, Station ZAPBench) that need no JAX and no data.

Integration runs (PPO training, ZAPBench training) take minutes and are recorded in
validation/jax_station-checks.json; tests/test_jax_station_slow.py exercises the
child processes when JAX and the prepared data are available.

Ported from the source repository. Its generator module (scripts/families_jax_station.py) is not part of
this library, so seeds, contracts, timeouts and wrapper bodies are read from the generated packages
(suite_test_support.package); prepared data live in the task cache (arena_task_paths), not under suite/_lib.
"""
import json
import os
import re
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

import arena_task_paths  # noqa: E402
import suite_test_support as support  # noqa: E402
from autoresearch_suite import jax_station_common as common  # noqa: E402
from autoresearch_suite import jax_station_limen as limen  # noqa: E402
from autoresearch_suite import jax_station_policy as policy  # noqa: E402
from autoresearch_suite import jax_station_zap as zap  # noqa: E402
from autoresearch_suite import jax_station_zap_score as zap_score  # noqa: E402
from autoresearch_suite import jax_station_zapdata as zd  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402


LIMEN_TASKS = {"easy_pickup": "limen-easy-pickup", "medium_place_near": "limen-medium-place-near",
               "hard_rule_chain": "limen-hard-rule-chain"}
ZAP_TASK = "the-station-zapbench-neural-activity-prediction"


def module_from(source):
    directory = tempfile.mkdtemp()
    path = Path(directory) / "candidate.py"
    path.write_text(textwrap.dedent(source))
    module = types.ModuleType("candidate")
    module.__file__ = str(path)
    return module


class PolicyTest(unittest.TestCase):
    def test_seeds_pass(self):
        for name, task_id in LIMEN_TASKS.items():
            self.assertEqual(policy.check_interface_source(support.seed_source(task_id)), [], name)

    def test_rejects_imports_assignments_and_io(self):
        source = textwrap.dedent("""
            import os
            import jax
            import jax.numpy as jnp
            from jax import config
            jnp.where = None
            def get_observation(state):
                jax.config.update("jax_enable_x64", True)
                return jnp.load(open("x"))
            def compute_reward(s, a, n):
                return jax.pure_callback(lambda: 0, None).__class__
        """)
        problems = " | ".join(policy.check_interface_source(source))
        for fragment in ("import of os", "jax.config", "attribute where", "attribute config", "name open",
                         "attribute load", "attribute pure_callback", "attribute __class__"):
            self.assertIn(fragment, problems)

    def test_list_extend_and_constants_are_allowed(self):
        source = "import jax.numpy as jnp\nD = jnp.array([1, 2])\ndef f(x):\n    out = []\n    out.extend([x])\n    return out\n"
        self.assertEqual(policy.check_interface_source(source), [])

    def test_global_statements_and_syntax_errors(self):
        self.assertTrue(policy.check_interface_source("x = 1\ndef f():\n    global x\n    x = 2\n"))
        self.assertTrue(policy.check_interface_source("def f(:\n"))


class LimenEvaluatorTest(unittest.TestCase):
    def test_missing_functions_and_policy_violations_are_invalid(self):
        module = module_from("def get_observation(state):\n    return state\n")
        module.get_observation = lambda state: state
        with self.assertRaises(CandidateError):
            limen.score(module, {"task": "easy_pickup", "wall_seconds": 10})
        module = module_from("import os\ndef get_observation(s):\n    return s\ndef compute_reward(s, a, n):\n    return 0.0\n")
        module.get_observation = module.compute_reward = lambda *a: 0.0
        with self.assertRaisesRegex(CandidateError, "interface policy"):
            limen.score(module, {"task": "easy_pickup", "wall_seconds": 10})

    def test_configs_and_paths(self):
        self.assertEqual(set(limen.CONFIGS), {"easy_pickup", "medium_place_near", "hard_rule_chain"})
        self.assertEqual(limen.limen_dir(), common.data_dir() / "limen-cf0b44da")
        self.assertEqual(common.data_dir(), arena_task_paths.data_root() / "jax_station")
        self.assertFalse(common.data_dir().is_relative_to(ROOT))  # prepared data never live inside the task library
        self.assertAlmostEqual(limen._std([0.16, 0.28, 0.2]), 0.0611010, places=6)


class FamiliesTest(unittest.TestCase):
    def test_registrations(self):
        registered = {spec["family"]: spec for spec in map(support.package, [*LIMEN_TASKS.values(), ZAP_TASK])}
        self.assertEqual(set(registered), {"limen_xland_easy_pickup", "limen_xland_medium_place_near",
                                           "limen_xland_hard_rule_chain", "station_zapbench_forecasting"})
        for name, spec in registered.items():
            self.assertLessEqual(spec["timeout"], 900, name)
            self.assertEqual(len(re.findall(r"\bjax_station_(?:limen|zap)\.score\(", spec["body"])), 1, name)
        for name in ("limen_xland_easy_pickup", "limen_xland_medium_place_near", "limen_xland_hard_rule_chain"):
            self.assertEqual(registered[name]["protocol"], "original")
            self.assertIn("CascadeEvaluator", registered[name]["contract"])
        zap_spec = registered["station_zapbench_forecasting"]
        self.assertEqual((zap_spec["direction"], zap_spec["protocol"]), ("minimize", "recreated"))
        self.assertIn("1,557", zap_spec["contract"])
        self.assertNotIn("<EPOCHS>", zap_spec["contract"])
        for name, task_id in LIMEN_TASKS.items():
            spec = support.package(task_id)
            self.assertIn(f"task={name!r}", spec["body"])
            self.assertLess(int(re.search(r"wall_seconds=(\d+)", spec["body"]).group(1)), spec["timeout"])


class ZapDataTest(unittest.TestCase):
    def test_split_sizes_match_the_station_and_zapbench(self):
        counts = zd.expected_counts()
        self.assertEqual(counts["train"], {"timesteps": 5052, "windows": 4772})
        self.assertEqual(counts["val"], {"timesteps": 749, "windows": 469})
        self.assertEqual(counts["test"], {"timesteps": 1872, "windows": 1557})
        self.assertEqual(zd.segments("test")[-1], (3, 3331, 3734))
        self.assertEqual(zd.segments("val")[0], (0, 451, 519))

    def test_windows_never_cross_conditions(self):
        labels = np.array([0] * 40 + [1] * 36 + [2] * 10)
        starts = zd.window_starts(labels)
        self.assertEqual(starts.tolist(), [0, 1, 2, 3, 4, 40])
        self.assertEqual(len(zd.window_starts(np.zeros(35))), 0)

    def test_zapbench_split_function(self):
        self.assertEqual(zd.split_bounds("train", 1, 648), (1, 455))
        self.assertEqual(zd.split_bounds("test_holdout", 3079, 3734), (3331, 3734))
        with self.assertRaises(ValueError):
            zd.split_bounds("bogus", 0, 10)


class ZapScoreModuleCheckTest(unittest.TestCase):
    def test_custom_call_policy(self):
        ok = 'func.func @main() { %0 = stablehlo.custom_call @lapack_sgetrf_ffi(%a) : x\n %1 = stablehlo.add %0, %0 }'
        self.assertEqual(zap_score.check_module(ok), [])
        bad = 'stablehlo.custom_call @xla_ffi_python_cpu_callback(%a)\n stablehlo.outfeed %b'
        problems = " ".join(zap_score.check_module(bad))
        self.assertIn("xla_ffi_python_cpu_callback", problems)
        self.assertIn("outfeed", problems)
        self.assertTrue(zap_score.check_module('stablehlo.custom_call @"my_target"(%x)'))

    def test_score_rejects_non_callable_hooks(self):
        module = module_from("create_network = 3\n")
        module.create_network = 3
        with self.assertRaisesRegex(CandidateError, "create_network"):
            zap.score(module, {})


class ChildRunnerTest(unittest.TestCase):
    def write_script(self, body):
        path = Path(tempfile.mkdtemp()) / "worker.py"
        path.write_text(textwrap.dedent(body))
        return path

    def test_round_trip_and_environment(self):
        script = self.write_script("""
            import json, os, sys
            job = json.loads(sys.stdin.read())
            keys = ("NPROC", "PJRT_NPROC", "OMP_NUM_THREADS", "JAX_PLATFORMS", "PYTHONHASHSEED")
            with os.fdopen(int(sys.argv[1]), "w") as out:
                out.write(json.dumps({"echo": job["x"], "env": {k: os.environ.get(k) for k in keys}, "cwd": os.getcwd()}))
        """)
        result, stats = common.run_child(script, {"x": 7}, wall_seconds=60, memory_mb=0, threads=4)
        self.assertEqual(result["echo"], 7)
        self.assertEqual(result["env"], {"NPROC": "4", "PJRT_NPROC": "4", "OMP_NUM_THREADS": "4", "JAX_PLATFORMS": "cpu",
                                         "PYTHONHASHSEED": "0"})
        self.assertFalse(os.path.exists(result["cwd"]))
        self.assertIn("cpu_seconds", stats)

    def test_timeout_and_crash(self):
        script = self.write_script("import time\ntime.sleep(30)\n")
        with self.assertRaises(common.ChildFailure) as caught:
            common.run_child(script, {}, wall_seconds=1, memory_mb=0)
        self.assertEqual(caught.exception.kind, "timeout")
        script = self.write_script("import sys\nsys.stderr.write('boom')\nraise SystemExit(3)\n")
        with self.assertRaises(common.ChildFailure) as caught:
            common.run_child(script, {}, wall_seconds=30, memory_mb=0)
        self.assertEqual(caught.exception.kind, "crash")
        self.assertIn("boom", caught.exception.stderr_tail)

    def test_audit_hook_confines_files_and_processes(self):
        allowed = tempfile.mkdtemp()
        secret = Path(tempfile.mkdtemp()) / "secret.txt"
        secret.write_text("answer")
        script = self.write_script(f"""
            import importlib.util, json, os, sys
            spec = importlib.util.spec_from_file_location("p", {str(Path(policy.__file__))!r})
            p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
            p.install_audit_hook(read_roots=[{allowed!r}], write_roots=[{allowed!r}])
            out = {{}}
            for name, action in (("read_secret", lambda: open({str(secret)!r}).read()),
                                 ("write_allowed", lambda: open(os.path.join({allowed!r}, "ok.txt"), "w").write("x")),
                                 ("write_elsewhere", lambda: open(os.path.join(os.path.dirname({str(secret)!r}), "x"), "w")),
                                 ("system", lambda: os.system("true"))):
                try:
                    action(); out[name] = "allowed"
                except (PermissionError, ImportError) as error:
                    out[name] = "blocked"
            with os.fdopen(int(sys.argv[1]), "w") as channel:
                channel.write(json.dumps(out))
        """)
        result, _ = common.run_child(script, {}, wall_seconds=60, memory_mb=0)
        self.assertEqual(result, {"read_secret": "blocked", "write_allowed": "allowed", "write_elsewhere": "blocked",
                                  "system": "blocked"})


class LockTest(unittest.TestCase):
    def test_lock_records_sources_and_licences(self):
        lock = json.loads((ROOT / "catalog" / "jax_station-data-lock.json").read_text())
        self.assertEqual(lock["limen"]["commit"], limen.LIMEN_COMMIT)
        self.assertEqual(lock["limen"]["license"], "MIT")
        self.assertIn("limen/evaluator.py", lock["limen"]["files"])
        self.assertEqual(lock["zapbench"]["license"], "CC-BY-4.0")
        self.assertEqual(lock["zapbench"]["chunks"]["count"], 16 * 141)
        self.assertEqual(set(lock["zapbench"]["files"]), {n for pair in zd.FILES.values() for n in pair})


if __name__ == "__main__":
    unittest.main()
