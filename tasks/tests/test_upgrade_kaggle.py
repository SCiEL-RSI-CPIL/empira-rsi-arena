"""Regression tests of the port of the kaggle package (MLE-bench recreations) to this library.

Each test reproduces a defect of the imported version on tiny synthetic data: label reads through
import-time code in the evaluator's process, native code, symbolic links and directory descriptors,
process creation below subprocess, a forged "ready" message, module-level work outside the budget,
booleans and text accepted as predictions, inexact decimals of jsonl inputs, a derivation that
depended on the pandas build, and preparation that wrote files before they were verified.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "scripts"))

from autoresearch_suite import kaggle_data, kaggle_sandbox, runtime, sandboxed  # noqa: E402

MISSING = [name for name in ("pandas", "scipy") if importlib.util.find_spec(name) is None]
needs_packages = unittest.skipIf(MISSING, f"not installed: {', '.join(MISSING)}")
if not MISSING:
    import pandas as pd  # noqa: E402
    from autoresearch_suite import kaggle_derive, kaggle_tasks, kaggle_worker  # noqa: E402
    import prepare_kaggle  # noqa: E402

TASK_IDS = ["eurekagent-jigsaw-toxic-comment", "loongflow-denoising-dirty-documents", "loongflow-histopathologic-cancer-detection",
            "loongflow-nomad2018-predict-transparent-conductors", "loongflow-stanford-covid-vaccine",
            "loongflow-us-patent-phrase-to-phrase-matching"]
# appended to hostile candidates: predictions made from the text of a label file, if they got it
SCORES = """

def _scores(text):
    rows = [line.split(",") for line in text.strip().splitlines()[1:]]
    return [float(r[1]) + 1e-9 * k for k, r in enumerate(rows)]
"""


@needs_packages
class SandboxRegressionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kaggle-upgrade-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.public, private = self.tmp / "patent" / "public", self.tmp / "patent" / "private"
        self.public.mkdir(parents=True)
        private.mkdir(parents=True)
        (self.public / "train.csv").write_text("id,anchor,target,context,score\na,abc,abd,H01,0.5\nb,xyz,xya,H01,0.25\n")
        (self.public / "test.csv").write_text("id,anchor,target,context\nd,bar,bars,H01\ne,bar,qux,H01\nf,bar,baz,A01\n")
        self.labels = private / "test_labels.csv"
        self.labels.write_text("id,score\nd,1.0\ne,0.0\nf,0.5\n")
        for name, value in (("data_dir", self.tmp), ("raw_cache_dir", self.tmp / "raw")):
            patcher = mock.patch.object(kaggle_data, name, lambda value=value: value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def candidate(self, source):
        path = self.tmp / f"candidate_{abs(hash(source))}.py"
        path.write_text(textwrap.dedent(source) + SCORES)
        return path

    def evaluate(self, source, budget=5):
        """What the package wrappers do: sandboxed.run with the task function of the evaluator."""
        def task(module):
            value, details = kaggle_tasks.patent(module, {"budget_s": budget, "memory_mb": 3500})
            return runtime.success(value, "maximize", **details)
        return sandboxed.run(self.candidate(source), task, timeout=60)

    def assertRejected(self, source, fragment, budget=5):
        result = self.evaluate(source, budget)
        self.assertFalse(result["valid"], result)
        self.assertIn(fragment, result["error"])
        return result

    def test_import_time_tampering_gains_nothing(self):
        marker = self.tmp / "ran_in_the_evaluator.txt"
        result = self.evaluate(f"""
            try:
                open({str(marker)!r}, "w").write("x")
                from autoresearch_suite import kaggle_tasks as _tasks
                _truth = [1.0, 0.0, 0.5]
                _tasks.run_fit_predict = lambda *a, **k: ([__import__("numpy").array(_truth)], {{"tampered": True}})
            except Exception:
                pass

            def fit_predict(train, test, budget):
                return [0.0, 1.0, 0.5]
            """)
        self.assertTrue(result["valid"], result)
        self.assertAlmostEqual(result["score"], -1.0)  # the honest predictions, not the labels
        self.assertNotIn("tampered", result)
        self.assertFalse(marker.exists())
        self.assertIs(kaggle_tasks.run_fit_predict, kaggle_sandbox.run_fit_predict)

    def test_symlink_and_directory_descriptor_escapes_are_denied(self):
        with mock.patch.object(kaggle_sandbox, "LINUX_NAMESPACES", False), mock.patch.object(kaggle_sandbox, "SANDBOX_EXEC", "/nonexistent"):
            self.assertRejected(f"""
                import os
                def fit_predict(train, test, budget):
                    os.makedirs("/".join(["d"] * 12)); os.symlink("/".join(["d"] * 12), "A")
                    os.makedirs("sub/A"); os.makedirs("S"); os.symlink("A" + "/.." * 12, "sub/S")
                    fd = os.open("S" + {str(self.labels)!r}, os.O_RDONLY, dir_fd=os.open("sub", os.O_RDONLY))
                    text = os.read(fd, 1 << 20).decode()
                    return _scores(text)
                """, "denied")
            self.assertRejected(f"""
                import os
                def fit_predict(train, test, budget):
                    fd = os.open("planted.txt", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=os.open({str(self.public)!r}, os.O_RDONLY))
                    return [1.0, 2.0, 3.0]
                """, "denied")
        self.assertFalse((self.public / "planted.txt").exists())

    def test_process_creation_below_subprocess_is_denied(self):
        self.assertRejected(f"""
            import _posixsubprocess, os
            def fit_predict(train, test, budget):
                r, w = os.pipe(); er, ew = os.pipe()
                _posixsubprocess.fork_exec([b"/bin/cat", {str(self.labels)!r}.encode()], [b"/bin/cat"], True, (), None, None,
                                           -1, -1, -1, w, -1, -1, er, ew, False, False, 0, None, None, None, -1, None, False)
                os.close(w)
                text = os.read(r, 1 << 20).decode()
                return _scores(text)
            """, "not allowed")
        self.assertRejected("""
            import multiprocessing
            def fit_predict(train, test, budget):
                with multiprocessing.get_context("spawn").Pool(1) as pool:
                    return pool.map(float, [1, 2, 3])
            """, "not allowed")

    def test_forged_ready_message_is_rejected(self):
        self.assertRejected("""
            import json, os, struct, sys
            def fit_predict(train, test, budget):
                payload = json.dumps({"type": "ready"}).encode()  # would restart the parent's budget clock
                os.write(int(sys.argv[2]), struct.pack(">Q", len(payload)) + payload)
                return [1.0, 2.0, 3.0]
            """, "unknown message")

    def test_module_level_code_counts_against_the_budget(self):
        with mock.patch.object(kaggle_sandbox, "GRACE_S", 1.0):
            self.assertRejected("""
                import time
                time.sleep(4)  # stands for training done while the module is executed
                def fit_predict(train, test, budget):
                    return [1.0, 2.0, 3.0]
                """, "exceeded the time budget", budget=1)

    def test_booleans_and_text_are_not_predictions(self):
        self.assertRejected("def fit_predict(train, test, budget):\n    return [True, False, True]\n", "real numbers")
        self.assertRejected("def fit_predict(train, test, budget):\n    return ['1.0', '0.0', '0.5']\n", "real numbers")
        self.assertRejected("def fit_predict(train, test, budget):\n    return {'a': 1}\n", "real numbers")

    def test_constant_check_is_exact(self):
        rows = range(32)  # enough rows for numpy's pairwise sum to produce inf - inf in np.std
        (self.public / "test.csv").write_text("id,anchor,target,context\n" + "".join(f"t{k},a{k},b{k},H01\n" for k in rows))
        self.labels.write_text("id,score\n" + "".join(f"t{k},{k % 5 / 4}\n" for k in rows))
        result = self.evaluate("def fit_predict(train, test, budget):\n    return [(-1) ** k * 1e308 for k in range(len(test))]\n")
        self.assertNotIn("constant", str(result["error"]))
        self.assertRejected("def fit_predict(train, test, budget):\n    return [0.25] * len(test)\n", "constant")

    def test_oversized_predictions_are_rejected_before_loading(self):
        with mock.patch.object(kaggle_sandbox, "MAX_OUTPUT_BYTES", 100):
            self.assertRejected("def fit_predict(train, test, budget):\n    return [1.0, 2.0, 3.0]\n", "bytes accepted")

    def test_unloadable_public_inputs_are_an_evaluator_error(self):
        (self.public / "test.csv").write_bytes(b"id,anchor\n\xff\xfe\xfa,\xc3\n")
        result = self.evaluate("def fit_predict(train, test, budget):\n    return [1.0, 2.0, 3.0]\n")
        self.assertFalse(result["valid"])
        self.assertIn("RuntimeError: the sandboxed child could not load the public inputs", result["error"])

    def test_native_code_cannot_reach_labels_or_write_outside(self):
        rules = kaggle_data.isolation("patent")
        with tempfile.TemporaryDirectory() as work:
            if kaggle_sandbox.os_sandbox(rules["deny_read"], rules["allow_read"], work) != "linux-namespaces":
                self.skipTest("Linux user namespaces are not available here (audit hook only)")
        outside = self.tmp / "native_write.txt"
        self.assertRejected(f"""
            import ctypes, os
            def fit_predict(train, test, budget):
                libc = ctypes.CDLL(None, use_errno=True)
                libc.open({str(outside)!r}.encode(), os.O_WRONLY | os.O_CREAT, 0o644)
                fd = libc.open({str(self.labels)!r}.encode(), 0)
                if fd < 0:
                    raise OSError(ctypes.get_errno(), "labels not reachable")
                text = os.read(fd, 1 << 20).decode()
                return _scores(text)
            """, "labels not reachable")
        self.assertFalse(outside.exists())
        result = self.evaluate("def fit_predict(train, test, budget):\n    return [3.0, 1.0, 2.0]\n")
        self.assertEqual(result["os_sandbox"], "linux-namespaces")


@needs_packages
class DataRegressionTest(unittest.TestCase):
    def test_jsonl_inputs_are_the_exact_decimals(self):
        values = np.random.RandomState(0).lognormal(size=2000)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "table.jsonl"
            path.write_bytes(kaggle_derive.jsonl_bytes([{"id": i, "x": float(v)} for i, v in enumerate(values)]))
            table = kaggle_worker.load_input({"kind": "jsonl", "path": str(path)})
        self.assertTrue((table["x"].to_numpy() == values).all())  # pandas' default decoder is off for about a fifth

    def test_openvaccine_signal_to_noise_is_float_of_the_text(self):
        texts = ["1.7269999999999999", "2.7439999999999998", "3.6310000000000002", "0.1"] * 600
        lists = json.dumps([0.1] * 68)
        rows = [{"index": i, "id": f"id_{i}", "sequence": "A" * 107, "structure": "." * 107, "predicted_loop_type": "E" * 107,
                 "signal_to_noise": texts[i], "SN_filter": 1, "seq_length": 107, "seq_scored": 68,
                 **{c: lists for c in kaggle_derive.OV_LIST_COLUMNS}} for i in range(2400)]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "train.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            files, _ = kaggle_derive.openvaccine({"openvaccine-trainset": path})
        records = [json.loads(line) for line in files["openvaccine/public/train.jsonl"].decode().splitlines()]
        self.assertTrue(all(r["signal_to_noise"] == float(texts[int(r["id"][3:])]) for r in records))

    def test_preparation_writes_nothing_unless_every_derived_file_matches_the_lock(self):
        derived = {"openvaccine/private/test_labels.npz": b"a", "openvaccine/public/test.jsonl": b"b", "openvaccine/public/train.jsonl": b"c"}
        recorded = {path: hashlib.sha256(payload).hexdigest() for path, payload in derived.items()}
        recorded["openvaccine/public/train.jsonl"] = "0" * 64  # the last file checked differs
        payload = b"placed by hand\n"
        with tempfile.TemporaryDirectory() as root:
            raw = Path(root) / "raw" / "openvaccine-trainset" / "train.csv"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(payload)
            source = {"id": "openvaccine-trainset", "task": "openvaccine", "url": "https://invalid.example/never-fetched",
                      "filename": "train.csv", "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
            lock = Path(root) / "lock.json"
            lock.write_text(json.dumps({"sources": [source], "files": [{"path": p, "sha256": h} for p, h in recorded.items()]}))
            with mock.patch.object(prepare_kaggle, "LOCK", lock), \
                    mock.patch.object(kaggle_data, "raw_cache_dir", lambda: Path(root) / "raw"), \
                    mock.patch.object(kaggle_data, "data_dir", lambda: Path(root) / "data"), \
                    mock.patch.dict(prepare_kaggle.DERIVE, {"openvaccine": lambda raw: (derived, {})}):
                with self.assertRaises(RuntimeError) as caught:
                    prepare_kaggle.prepare(["openvaccine"])  # the source placed by hand is verified, not fetched
                self.assertIn("differs from the lock", str(caught.exception))
                self.assertFalse((Path(root) / "data").exists())  # two files matched, but nothing is written
                with mock.patch.dict(prepare_kaggle.NEEDS, {"openvaccine": [("no_such_module_for_this_test", "nosuch")]}):
                    with self.assertRaises(RuntimeError) as caught:
                        prepare_kaggle.prepare(["openvaccine"])
                self.assertIn("nosuch", str(caught.exception))

    def test_requests_and_lock_name_no_project_path(self):
        self.assertNotIn("autoresearch", json.dumps(prepare_kaggle.USER_AGENT))
        lock = json.loads((ROOT / "catalog" / "kaggle-data-lock.json").read_text())
        for field in ("raw_cache", "data_directory"):
            self.assertFalse(lock[field].startswith((".cache", "suite/")), lock[field])


class WrapperTest(unittest.TestCase):
    def test_wrappers_never_import_the_candidate(self):
        for task_id in TASK_IDS:
            tree = ast.parse((ROOT / "suite" / task_id / "evaluator.py").read_text())
            calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
            self.assertIn("sandboxed.run", calls, task_id)
            self.assertNotIn("runtime.run", calls, task_id)


if __name__ == "__main__":
    unittest.main()
