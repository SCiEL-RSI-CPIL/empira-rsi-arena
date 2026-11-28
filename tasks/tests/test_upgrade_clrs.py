"""Regression tests of the port and audit of the EvoPrompting CLRS-30 packages (package clrs).

Each test pins one defect that was found in the copied library or wrappers and fixed here:
 - the wrappers imported the candidate in the evaluator's process (runtime.run), so module-level code could
   replace the scoring function; they now hand it over unexecuted (sandboxed.run);
 - the parameter cap was counted once, right after build_processor, so parameters created at the first
   forward call (by hand or with torch.nn.LazyLinear) were not counted;
 - the stated limit is resident memory, but on Linux the child's address space was capped at the same
   value, so a candidate well under the stated limit failed; the address-space cap now has a margin;
 - without psutil (not a requirement of the packages) the resident memory read 0 and was never limited;
 - malformed child output could raise TypeError, ValueError or OverflowError in the scorer instead of a
   candidate error;
 - finite outputs were checked on the probe graph only, so a processor that is non-finite on the larger
   test graphs was scored instead of rejected;
 - the child kept arena_task_paths (which locates the prepared data) loaded after the policy was installed;
 - the prepared file with the test outputs stayed readable in the child: a native reader (torch.from_file)
   plus a frame walk to the trainer's predict() scored 1.0 with a linear processor; the task cache is now
   hidden from the child by the kernel (kaggle_worker.isolate) where namespaces are available.
The tests that train need torch and the CLRS-30 files (scripts/prepare_clrs.py) and are skipped otherwise.
"""
from __future__ import annotations

import base64
import importlib.util
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "suite" / "_lib", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from autoresearch_suite import clrs_data as D  # noqa: E402
from autoresearch_suite import clrs_metrics as M  # noqa: E402
from autoresearch_suite import clrs_sandbox as SB  # noqa: E402
from autoresearch_suite import clrs_specs as S  # noqa: E402
from autoresearch_suite import kaggle_sandbox  # noqa: E402
from autoresearch_suite import runtime, sandboxed  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

import suite_test_support as support  # noqa: E402

UPGRADES = support.load("catalog/upgrades/clrs.json")
TASKS = sorted(UPGRADES["tasks"])
LOCK = support.load("catalog/clrs-data-lock.json")
DIGESTS = {entry["path"][:-4]: entry["content_sha256"] for entry in LOCK["files"]}
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_DATA = (D.data_dir() / "bfs.npz").is_file()
TRAINS = unittest.skipUnless(HAVE_TORCH and HAVE_DATA, "needs torch and the prepared CLRS-30 file bfs.npz")

LINEAR = """
import torch


class Processor(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.layer = torch.nn.Linear(2 * hidden, hidden)
{extra_init}
    def forward(self, node_fts, edge_fts, graph_fts, adj, hidden):
        out = torch.relu(self.layer(torch.cat([node_fts, hidden], dim=-1)))
{extra_forward}
        return out


def build_processor(hidden_dim):
    return Processor(hidden_dim)
"""


NATIVE_READ = """
import torch


def build_processor(hidden_dim):
    path = {path!r}
    try:
        data = torch.from_file(path, shared=False, size=torch.os.stat(path).st_size, dtype=torch.uint8)
        outcome = "read %d bytes" % data.numel()
    except Exception as error:  # noqa: BLE001
        outcome = "not readable (%r)" % (error,)
    raise RuntimeError("probe: " + outcome)
"""

# The audit's exploit (scratch-up-clrs/audit2/cand_exploit_test_labels.py) with the data path given: read the
# prepared file natively, walk the frames to train_and_predict and replace its predict() by a lookup of the
# true test outputs; it falls back to the honest linear processor when the file cannot be read.
EXPLOIT = """
import numpy
import torch


def install(path):
    holder = []

    def walker():
        yield holder[0].gi_frame.f_back

    generator = walker()
    holder.append(generator)
    frame = next(generator)
    while frame is not None and frame.f_code.co_name != "train_and_predict":
        frame = frame.f_back
    raw = bytes(torch.from_file(path, shared=False, size=torch.os.stat(path).st_size, dtype=torch.uint8).numpy().tobytes())
    archive = numpy.load(torch.sys.modules["io"].BytesIO(raw), allow_pickle=False)
    truth = {key[5:]: archive[key] for key in archive.files if key.startswith("test/")}
    trainer = frame.f_globals
    original = trainer["predict"]

    def predict(net, split, algorithm, batch_size, torch_module):
        if "output_pi" in split or split["input_pos"].shape[1] != truth["input_pos"].shape[1]:
            return original(net, split, algorithm, batch_size, torch_module)
        rows = [next(j for j in range(len(truth["lengths"]))
                     if all(numpy.array_equal(split[k][i], truth[k][j]) for k in split if k.startswith("input_")))
                for i in range(len(split["lengths"]))]
        return {"pi": truth["output_pi"][rows].astype(numpy.int64)}

    trainer["predict"] = predict
"""


def linear(extra_init="", extra_forward="", prefix=""):
    init = textwrap.indent(textwrap.dedent(extra_init), " " * 8) if extra_init else ""
    forward = textwrap.indent(textwrap.dedent(extra_forward), " " * 8) if extra_forward else ""
    return prefix + LINEAR.format(extra_init=init, extra_forward=forward)


def tiny(**extra):
    """The bfs package's parameters with a short budget (2 steps, 4 test samples)."""
    P = dict(support.evaluator("evoprompting-bfs")["P"])
    P.update(steps=2, eval_every=1, val_samples=4, test_samples=4, threads=2, wall_seconds=90, **extra)
    return P


class CandidateFile:
    def __init__(self, test, source):
        directory = tempfile.TemporaryDirectory()
        test.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "candidate.py"
        self.path.write_text(textwrap.dedent(source), encoding="utf-8")


def score(test, source, **extra):
    """clrs_eval.score of an unexecuted candidate, as the switched wrappers call it."""
    from autoresearch_suite import clrs_eval

    return clrs_eval.score(sandboxed.source_program(CandidateFile(test, source).path), tiny(**extra))


def kernel_sandbox():
    """The kernel mechanism that would hide the task cache from the training child here, or None."""
    with tempfile.TemporaryDirectory() as work:
        return SB.os_sandbox(os.path.realpath(work), [os.path.realpath(p) for p in D.isolation()["deny_read"]])


class PackageTests(unittest.TestCase):
    def test_wrappers_hand_over_the_candidate_unexecuted(self):
        for task_id in TASKS:
            text = (ROOT / "suite" / task_id / "evaluator.py").read_text(encoding="utf-8")
            self.assertIn("sandboxed.run(program_path, task, timeout=TIMEOUT)", text, task_id)
            self.assertNotIn("runtime.run(", text, task_id)

    def test_parameters_agree_with_statement_record_and_lock(self):
        for task_id in TASKS:
            P = support.evaluator(task_id)["P"]
            problem = support.load(f"suite/{task_id}/problem.json")
            self.assertEqual(P, problem["parameters"], task_id)
            self.assertEqual(P, UPGRADES["tasks"][task_id]["params"], task_id)
            self.assertEqual(P["data_sha256"], DIGESTS[P["algorithm"]], task_id)
            self.assertEqual(S.task_algorithm(task_id), P["algorithm"], task_id)
            self.assertIn(f"{P['steps']} steps", problem["description"], task_id)
            self.assertIn(f"wall_seconds ({P['wall_seconds']} s)", problem["contract"], task_id)

    def test_every_package_has_the_same_seed(self):
        seeds = {support.seed_source(task_id) for task_id in TASKS}
        self.assertEqual(len(seeds), 1)

    def test_lock_directory_is_not_inside_the_library(self):
        self.assertFalse(LOCK["directory"].startswith("suite/"))
        self.assertTrue(str(D.data_dir()).endswith(os.path.join("data", "clrs")))

    def test_child_is_denied_the_task_cache_but_not_what_it_runs(self):
        denied = [Path(p).resolve() for p in D.isolation()["deny_read"]]
        self.assertTrue(any(D.data_dir().resolve().is_relative_to(root) for root in denied))
        for needed in (SB.WORKER, SB.LIB, Path(sys.prefix), Path(sys.executable)):
            self.assertFalse(any(needed.resolve().is_relative_to(root) for root in denied), needed)
        with tempfile.TemporaryDirectory() as other, mock.patch.dict(os.environ, {"XDG_CACHE_HOME": other}):
            default = str(Path(other, "empira", "tasks").resolve())  # also hidden when another cache is configured
            self.assertIn(default, D.isolation()["deny_read"])


class HostileOutputTests(unittest.TestCase):
    def test_malformed_child_payloads_are_candidate_errors(self):
        raw = base64.b64encode(np.zeros(10, dtype=np.int64).tobytes()).decode()
        cases = {"dtype list": {"dtype": ["int64"], "shape": [2, 5], "data": raw},
                 "dimension string": {"dtype": "int64", "shape": ["2", 5], "data": raw},
                 "dimension word": {"dtype": "int64", "shape": ["x", 5], "data": raw},
                 "dimension float": {"dtype": "int64", "shape": [2.0, 5], "data": raw},
                 "dimension None": {"dtype": "int64", "shape": [None, 5], "data": raw},
                 "dimension list": {"dtype": "int64", "shape": [[2], 5], "data": raw},
                 "dimension bool": {"dtype": "int64", "shape": [True, 10], "data": raw},
                 "overflowing dimensions": {"dtype": "int64", "shape": [1 << 32, 1 << 32], "data": raw},
                 "huge dimension": {"dtype": "int64", "shape": [1 << 70], "data": raw},
                 "too many axes": {"dtype": "int64", "shape": [1, 1, 1, 2, 5], "data": raw},
                 "data missing": {"dtype": "int64", "shape": [2, 5]}}
        for label, item in cases.items():
            with self.subTest(label), self.assertRaises(CandidateError):
                SB.decode_arrays({"pi": item})
        good = SB.decode_arrays({"pi": {"dtype": "int64", "shape": [2, 5], "data": raw}})
        self.assertEqual(good["pi"].shape, (2, 5))

    def test_prediction_names_of_any_type_are_checked(self):
        truths = {"pi": np.zeros((2, 5), dtype=np.int8)}
        with self.assertRaises(CandidateError):
            M.check_predictions("bfs", {"pi": np.zeros((2, 5), dtype=np.int64), 3: np.zeros(2)}, truths, 5)


class MemoryReadingTests(unittest.TestCase):
    @unittest.skipUnless(Path("/proc/self/statm").is_file(), "needs /proc (Linux)")
    def test_resident_memory_is_read_without_psutil(self):
        own = SB._rss_mb_without_psutil(os.getpid())
        self.assertGreater(own, 1.0)
        self.assertLess(own, 1 << 20)
        self.assertEqual(SB._rss_mb_without_psutil(2 ** 22 + 12345), 0.0)  # no such process

    def test_address_space_cap_has_a_margin(self):
        self.assertGreaterEqual(SB.ADDRESS_SPACE_MARGIN_MB, 1024)


@TRAINS
class TrainingRegressionTests(unittest.TestCase):
    def test_parameters_created_at_the_first_call_are_counted(self):
        lazy = linear(extra_init="self.wide = None\n",
                      extra_forward="if self.wide is None:\n"
                                    "    self.wide = torch.nn.Linear(node_fts.shape[-1], 12000)\n"
                                    "out = out + self.wide(node_fts)[..., :out.shape[-1]]\n")
        lazy_linear = linear(extra_init="self.wide = torch.nn.LazyLinear(12000)\n",
                             extra_forward="out = out + self.wide(node_fts)[..., :out.shape[-1]]\n")
        for label, source in (("created in forward", lazy), ("torch.nn.LazyLinear", lazy_linear)):
            with self.subTest(label), self.assertRaisesRegex(CandidateError, "at most 1000000 are allowed"):
                score(self, source)

    def test_non_finite_outputs_on_test_graphs_are_rejected(self):
        source = linear(extra_forward="if node_fts.shape[1] > 16:\n    out = out * float('nan')\n")
        with self.assertRaisesRegex(CandidateError, "non-finite values on a 64-node graph"):
            score(self, source)

    def test_resident_memory_under_the_limit_is_accepted_and_over_it_rejected(self):
        # The child of a linear processor holds about 0.5-0.6 GB resident and about 0.7 GB more mapped.
        ballast = "self.ballast = torch.ones({} * 262144)\n"
        value, details = score(self, linear(extra_init=ballast.format(400)), memory_mb=1600)
        self.assertTrue(0.0 <= value <= 1.0)
        self.assertLess(details["runs"][0]["peak_rss_mb"], 1600)
        with self.assertRaisesRegex(CandidateError, "exceeded the 1600 MB memory limit"):
            score(self, linear(extra_init=ballast.format(1500)), memory_mb=1600)

    def test_child_unloads_the_evaluator_and_path_modules(self):
        probe = linear(extra_init="raise RuntimeError('loaded=' + str(sorted(m for m in torch.sys.modules "
                                  "if m.startswith(('autoresearch_suite', 'arena_task_paths')))))\n")
        with self.assertRaisesRegex(CandidateError, r"loaded=\[\]"):
            score(self, probe)

    def test_native_reader_cannot_reach_the_prepared_file(self):
        if kernel_sandbox() != "linux-namespaces":
            self.skipTest("Linux user namespaces are not available here (audit hook only)")
        with self.assertRaisesRegex(CandidateError, r"probe: not readable \(FileNotFoundError"):
            score(self, NATIVE_READ.format(path=str(D.data_dir() / "bfs.npz")))
        with tempfile.TemporaryDirectory() as other, mock.patch.dict(os.environ, {"XDG_CACHE_HOME": other}):
            copy = Path(other, "empira", "tasks", "data", "clrs", "bfs.npz")  # a prepared file in the default cache
            copy.parent.mkdir(parents=True)
            copy.write_bytes(b"test outputs")
            with self.assertRaisesRegex(CandidateError, r"probe: not readable \(FileNotFoundError"):
                score(self, NATIVE_READ.format(path=str(copy)))

    def test_reading_the_test_outputs_and_replacing_predict_scores_honestly(self):
        if kernel_sandbox() != "linux-namespaces":
            self.skipTest("Linux user namespaces are not available here (audit hook only)")
        exploit = EXPLOIT + linear().replace("    return Processor(hidden_dim)", (
            "    try:\n        install({path!r})\n    except Exception:  # noqa: BLE001\n        pass\n"
            "    return Processor(hidden_dim)").format(path=str(D.data_dir() / "bfs.npz")))
        honest, honest_details = score(self, linear())
        value, details = score(self, exploit)
        self.assertEqual((details["os_sandbox"], honest_details["os_sandbox"]), ("linux-namespaces", "linux-namespaces"))
        self.assertEqual(details["runs"][0]["os_sandbox"], "linux-namespaces")
        self.assertEqual(value, honest)
        self.assertLess(value, 1.0)

    @unittest.skipUnless(sys.platform.startswith("linux"), "the fallback is reported where namespaces are switched off (Linux)")
    def test_fallback_without_namespaces_is_reported_and_leaves_native_reads_open(self):
        self.addCleanup(setattr, kaggle_sandbox, "LINUX_NAMESPACES", kaggle_sandbox.LINUX_NAMESPACES)
        kaggle_sandbox.LINUX_NAMESPACES = False  # as where the kernel forbids user namespaces
        value, details = score(self, linear())
        self.assertEqual(details["os_sandbox"], "audit-hook-only")
        with self.assertRaisesRegex(CandidateError, r"probe: read \d+ bytes"):  # the documented residual risk
            score(self, NATIVE_READ.format(path=str(D.data_dir() / "bfs.npz")))

    def test_import_time_tampering_gains_nothing(self):
        tamper = ("import torch\n"
                  "_metrics = torch.sys.modules.get('autoresearch_suite.clrs_metrics')\n"
                  "if _metrics is not None:\n"
                  "    _original = _metrics.score\n"
                  "    def _perfect(algorithm, predictions, truths):\n"
                  "        value, per_output = _original(algorithm, predictions, truths)\n"
                  "        return 1.0, {name: 1.0 for name in per_output}\n"
                  "    _metrics.score = _perfect\n")
        from autoresearch_suite import clrs_eval, clrs_metrics

        original_score = clrs_metrics.score
        self.addCleanup(setattr, clrs_metrics, "score", original_score)
        P = tiny()

        def task(module):
            value, details = clrs_eval.score(module, P)
            return runtime.success(value, "maximize", **details)

        honest = sandboxed.run(str(CandidateFile(self, linear()).path), task, timeout=120)
        tampering = CandidateFile(self, linear(prefix=tamper)).path
        sandboxed_result = sandboxed.run(str(tampering), task, timeout=120)
        self.assertIs(clrs_metrics.score, original_score)
        self.assertTrue(honest["valid"], honest.get("error"))
        self.assertEqual(sandboxed_result["score"], honest["score"])
        self.assertLess(honest["score"], 1.0)
        imported_result = runtime.run(str(tampering), task, timeout=120)  # the wrappers' former path
        self.assertEqual(imported_result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
