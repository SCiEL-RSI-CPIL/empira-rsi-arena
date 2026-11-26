"""Isolation and limits of the kaggle package's sandboxed fit_predict runner (tiny synthetic data)."""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import kaggle_data, kaggle_sandbox, runtime, sandboxed  # noqa: E402

# The evaluators and the child process need pandas, the patent metric scipy (requirements of the packages).
MISSING = [name for name in ("pandas", "scipy") if importlib.util.find_spec(name) is None]
if not MISSING:
    from autoresearch_suite import kaggle_tasks  # noqa: E402


def write_patent(root):
    public, private = root / "patent" / "public", root / "patent" / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    (public / "train.csv").write_text("id,anchor,target,context,score\na,abc,abd,H01,0.5\nb,xyz,xya,H01,0.25\nc,null,NA,A01,0.75\n")
    (public / "test.csv").write_text("id,anchor,target,context\nd,bar,bars,H01\ne,bar,qux,H01\nf,bar,baz,A01\n")
    (private / "test_labels.csv").write_text("id,score\nd,1.0\ne,0.0\nf,0.5\n")
    return private / "test_labels.csv"


@unittest.skipIf(MISSING, f"not installed: {', '.join(MISSING)}")
class SandboxTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="kaggle-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.labels = write_patent(self.tmp)
        patcher = mock.patch.object(kaggle_data, "data_dir", lambda: self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def score(self, source, budget=5, memory_mb=3500):
        path = self.tmp / f"candidate_{abs(hash(source))}.py"
        path.write_text(textwrap.dedent(source))
        module = sandboxed.source_program(path, name=f"kaggle_candidate_{abs(hash(source))}")  # as the wrappers hand it over
        return kaggle_tasks.patent(module, {"budget_s": budget, "memory_mb": memory_mb})

    def assertRejected(self, source, fragment, budget=5, memory_mb=3500):
        with self.assertRaises(runtime.CandidateError) as caught:
            self.score(source, budget, memory_mb)
        self.assertIn(fragment, str(caught.exception))

    def test_good_candidate_scores_and_sees_text_as_strings(self):
        value, details = self.score("""
            import numpy as np
            def fit_predict(train, test, budget):
                assert list(train.columns) == ["id", "anchor", "target", "context", "score"]
                assert train["anchor"].tolist()[2] == "null" and train["target"].tolist()[2] == "NA"
                assert "score" not in test.columns and budget == 5.0
                return np.array([len(set(a) & set(t)) for a, t in zip(test.anchor, test.target)], dtype=float)
        """)
        self.assertGreater(value, 0.9)
        self.assertEqual(details["test_pairs"], 3)

    def test_reading_held_out_labels_is_denied(self):
        self.assertRejected(f"""
            import pandas as pd
            def fit_predict(train, test, budget):
                return pd.read_csv({str(self.labels)!r})["score"].to_numpy()
        """, "denied")

    def test_listing_the_data_directory_is_denied(self):
        self.assertRejected(f"""
            import os
            def fit_predict(train, test, budget):
                return [float(len(os.listdir({str(self.tmp)!r})))] * 3
        """, "denied")

    def test_writing_outside_the_working_directory_is_denied(self):
        self.assertRejected(f"""
            def fit_predict(train, test, budget):
                open({str(self.tmp / 'leak.txt')!r}, "w").write("x")
                return [1.0, 2.0, 3.0]
        """, "denied")
        self.assertFalse((self.tmp / "leak.txt").exists())

    def test_scratch_files_in_the_working_directory_are_allowed(self):
        value, _ = self.score("""
            import numpy as np, tempfile, os
            def fit_predict(train, test, budget):
                path = os.path.join(tempfile.gettempdir(), "cache.npy")
                np.save(path, np.array([3.0, 1.0, 2.0]))
                return np.load(path)
        """)
        self.assertAlmostEqual(value, 1.0)

    def test_network_and_processes_are_denied(self):
        self.assertRejected("""
            import urllib.request
            def fit_predict(train, test, budget):
                urllib.request.urlopen("https://example.com", timeout=5)
        """, "not allowed")
        self.assertRejected("""
            import subprocess
            def fit_predict(train, test, budget):
                subprocess.run(["/bin/echo", "x"])
        """, "not allowed")

    def test_budget_is_enforced_by_killing_the_child(self):
        with mock.patch.object(kaggle_sandbox, "GRACE_S", 1.0):
            self.assertRejected("""
                import time
                def fit_predict(train, test, budget):
                    time.sleep(budget + 30)
            """, "exceeded the time budget", budget=1)

    def test_invalid_predictions_are_rejected(self):
        self.assertRejected("def fit_predict(train, test, budget):\n    return [1.0, 2.0]\n", "shape")
        self.assertRejected("def fit_predict(train, test, budget):\n    return [1.0, float('nan'), 2.0]\n", "finite")
        self.assertRejected("def fit_predict(train, test, budget):\n    return [1.0, 1.0, 1.0]\n", "constant")
        self.assertRejected("def fit_predict(train, test, budget):\n    return ['a', 'b', 'c']\n", "fit_predict failed")
        self.assertRejected("def fit_predict(train, test, budget):\n    raise RuntimeError('boom')\n", "boom")
        self.assertRejected("x = 1\n", "must define fit_predict")

    def test_crashing_child_is_reported(self):
        self.assertRejected("""
            import os
            def fit_predict(train, test, budget):
                os._exit(3)
        """, "ended without a result")

    def test_memory_limit(self):
        try:
            import psutil  # noqa: F401
        except ImportError:
            self.skipTest("psutil unavailable")
        self.assertRejected("""
            import numpy as np, time
            def fit_predict(train, test, budget):
                block = np.ones((400, 1024, 1024), dtype=np.uint8)
                time.sleep(3)
                return [float(block[0, 0, 0])] * 3
        """, "memory limit", memory_mb=200)

    def test_audit_hook_alone_blocks_python_level_reads(self):
        with mock.patch.object(kaggle_sandbox, "SANDBOX_EXEC", "/nonexistent/sandbox-exec"), \
                mock.patch.object(kaggle_sandbox, "LINUX_NAMESPACES", False):
            self.assertRejected(f"""
                def fit_predict(train, test, budget):
                    return [float(len(open({str(self.labels)!r}).read()))] * 3
            """, "denied")
            value, details = self.score("def fit_predict(train, test, budget):\n    return [3.0, 1.0, 2.0]\n")
            self.assertEqual(details["os_sandbox"], "audit-hook-only")
            self.assertAlmostEqual(value, 1.0)

    @unittest.skipUnless(Path(kaggle_sandbox.SANDBOX_EXEC).exists() and sys.platform == "darwin", "macOS sandbox-exec only")
    def test_os_sandbox_blocks_native_readers(self):
        self.assertRejected(f"""
            import pyarrow.csv
            def fit_predict(train, test, budget):
                return pyarrow.csv.read_csv({str(self.labels)!r}).column("score").to_numpy()
        """, "fit_predict failed")


class LayoutTest(unittest.TestCase):
    def test_isolation_denies_data_and_raw_cache_and_allows_public_inputs(self):
        rules = kaggle_data.isolation("nomad")
        self.assertIn(str(kaggle_data.data_dir()), rules["deny_read"])
        self.assertIn(str(kaggle_data.raw_cache_dir()), rules["deny_read"])
        self.assertEqual(rules["allow_read"], [str(kaggle_data.public_dir("nomad"))])

    def test_profile_orders_deny_before_allow(self):
        with tempfile.TemporaryDirectory() as root:
            profile = kaggle_sandbox._profile([root], [str(Path(root) / "x" / "public")], root)
        self.assertLess(profile.index("(deny file-read*"), profile.index("(allow file-read*"))
        self.assertIn("(deny network*)", profile)
        self.assertLess(profile.index("(deny file-write*)"), profile.index("(allow file-write*"))

    def test_missing_prepared_data_is_an_installation_error(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(kaggle_data, "data_dir", lambda: Path(root)):
            with self.assertRaises(RuntimeError) as caught:
                kaggle_data.require_files("patent", public=("train.csv",))
        self.assertIn("prepare_kaggle.py", str(caught.exception))

    def test_source_files_stay_under_500_lines(self):
        files = list((ROOT / "suite" / "_lib" / "autoresearch_suite").glob("kaggle_*.py"))
        files += [ROOT / "scripts" / "prepare_kaggle.py", ROOT / "tests" / "test_upgrade_kaggle.py"]
        files += list((ROOT / "tests").glob("test_kaggle*.py"))
        for path in files:
            self.assertLessEqual(len(path.read_text().splitlines()), 500, path.name)


if __name__ == "__main__":
    unittest.main()
