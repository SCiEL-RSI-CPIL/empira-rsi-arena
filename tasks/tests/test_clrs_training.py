"""EvoPrompting CLRS-30 tasks (package clrs): PyTorch network, sandboxed training and scoring on tiny budgets.

Needs torch and the CLRS-30 files built by scripts/prepare_clrs.py; skipped otherwise.

The source repository's test reads the seed program from its generator module
(scripts/families_clrs.py), which is not part of this library; it is read here from a generated
package (the seed is the same in every CLRS package, see test_upgrade_clrs.py).
"""
from __future__ import annotations

import atexit
import importlib.util
import json
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "suite" / "_lib", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from autoresearch_suite import clrs_data as D  # noqa: E402
from autoresearch_suite import clrs_specs as S  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

import suite_test_support as support  # noqa: E402

SEED = support.seed_source("evoprompting-bfs")
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_DATA = all((D.data_dir() / f"{a}.npz").is_file() for a in S.ALGORITHMS)
LOCK = json.loads((ROOT / "catalog" / "clrs-data-lock.json").read_text())
DIGESTS = {entry["path"][:-4]: entry["content_sha256"] for entry in LOCK["files"]}


def program(source):
    """Write a candidate program to a temporary file and import it as a module (as the suite template does)."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    handle.write(textwrap.dedent(source))
    handle.close()
    atexit.register(lambda: Path(handle.name).unlink(missing_ok=True))
    module = types.ModuleType("clrs_test_candidate")
    module.__file__ = handle.name
    exec(compile(Path(handle.name).read_text(), handle.name, "exec"), module.__dict__)
    return module


def tiny(algorithm, **extra):
    return {"algorithm": algorithm, "data_sha256": DIGESTS[algorithm], "steps": 2, "eval_every": 1, "val_samples": 2,
            "test_samples": 2, "threads": 2, "wall_seconds": 300, **extra}


# EvoPrompting (arXiv 2302.14838) Appendix A.6 Table 3, model size of the baseline Triplet-GMPNN per algorithm.
TABLE3_BASELINE_SIZES = {
    "activity_selector": 262204, "articulation_points": 531913, "bellman_ford": 524604, "bfs": 523963,
    "binary_search": 262204, "bridges": 532556, "bubble_sort": 524477, "dag_shortest_paths": 793287, "dfs": 661190,
    "dijkstra": 525886, "find_maximum_subarray_kadane": 264514, "floyd_warshall": 625089, "graham_scan": 398409,
    "heapsort": 659654, "insertion_sort": 524477, "jarvis_march": 264898, "kmp_matcher": 398021, "lcs_length": 270419,
    "matrix_chain_order": 624448, "minimum": 261307, "mst_kruskal": 399691, "mst_prim": 525886,
    "naive_string_matcher": 262588, "optimal_bst": 625987, "quickselect": 395714, "quicksort": 525759,
    "segments_intersect": 263359, "strongly_connected_components": 663243, "task_scheduling": 262333,
    "topological_sort": 660164,
}

LINEAR = """
import torch


class Processor(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.layer = torch.nn.Linear(2 * hidden, hidden)

    def forward(self, node_fts, edge_fts, graph_fts, adj, hidden):
        return torch.relu(self.layer(torch.cat([node_fts, hidden], dim=-1)))


def build_processor(hidden_dim):
    return Processor(hidden_dim)
"""


@unittest.skipUnless(HAVE_TORCH and HAVE_DATA, "needs torch and the prepared CLRS-30 files")
class NetworkTests(unittest.TestCase):
    def test_seed_equals_reference_processor(self):
        import torch

        from autoresearch_suite import clrs_processors

        seed = program(SEED)
        torch.manual_seed(7)
        mine = seed.build_processor(16)
        torch.manual_seed(7)
        reference = clrs_processors.triplet_gmpnn(16)
        generator = torch.Generator().manual_seed(3)
        args = (torch.randn(2, 5, 16, generator=generator), torch.randn(2, 5, 5, 16, generator=generator),
                torch.randn(2, 16, generator=generator), torch.ones(2, 5, 5), torch.randn(2, 5, 16, generator=generator))
        for a, b in zip(mine(*args), reference(*args)):
            torch.testing.assert_close(a, b)
        self.assertEqual(sum(p.numel() for p in seed.build_processor(128).parameters()), 258872)

    def test_triplet_messages_equal_the_dense_clrs_expression(self):
        import torch

        from autoresearch_suite import clrs_processors

        torch.manual_seed(0)
        proc = clrs_processors.triplet_gmpnn(8)
        z, edges, graph = torch.randn(2, 4, 16), torch.randn(2, 4, 4, 8), torch.randn(2, 8)
        tri = (proc.t_1(z)[:, :, None, None] + proc.t_2(z)[:, None, :, None] + proc.t_3(z)[:, None, None, :]
               + proc.t_e_1(edges)[:, :, :, None] + proc.t_e_2(edges)[:, :, None, :] + proc.t_e_3(edges)[:, None, :, :]
               + proc.t_g(graph)[:, None, None, None])  # (B, N, N, N, T) as in clrs get_triplet_msgs
        dense = torch.relu(proc.o3(tri.amax(dim=1)))
        torch.testing.assert_close(proc.triplet_messages(z, edges, graph), dense)

    def test_model_sizes_equal_evoprompting_baseline(self):
        """With the seed processor, the whole network has EvoPrompting's Triplet-GMPNN model size (Table 3)."""
        import torch

        from autoresearch_suite import clrs_train

        seed = program(SEED)
        for algorithm, size in TABLE3_BASELINE_SIZES.items():
            data = D.load(algorithm)
            net, processor_parameters, edge = clrs_train.build_network(seed.build_processor, algorithm, data["train"],
                                                                       128, 10**6, torch)
            self.assertEqual(sum(p.numel() for p in net.parameters()), size, algorithm)
            self.assertTrue(edge)

    def test_step_checkpointing_gives_the_same_loss_and_gradients(self):
        import torch

        from autoresearch_suite import clrs_train

        seed = program(SEED)
        for algorithm in ("insertion_sort", "floyd_warshall", "dfs"):
            data = D.load(algorithm)
            batch = D.make_batch(data["train"], algorithm, [0, 1, 2], rng=np.random.RandomState(0))
            results = []
            for checkpointing in (False, True):
                torch.manual_seed(0)
                net, _, _ = clrs_train.build_network(seed.build_processor, algorithm, data["train"], 16, 10**6, torch)
                net.checkpoint_steps = checkpointing
                generator = torch.Generator().manual_seed(5)
                loss = clrs_train.batch_loss(net, net.spec, clrs_train.as_tensors(batch, torch), batch.nb_nodes, generator)
                loss.backward()
                results.append((loss.detach(), [p.grad.clone() for p in net.parameters()]))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            for a, b in zip(results[0][1], results[1][1]):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_one_step_for_every_algorithm(self):
        from autoresearch_suite import clrs_train

        candidate = program(LINEAR)
        for algorithm in S.ALGORITHMS:
            data = D.load(algorithm)
            val = {key: value[:2] for key, value in data["val"].items()}
            test = {key: value[:2] for key, value in data["test"].items() if not key.startswith("output_")}
            if test["input_pos"].shape[1] > 16:  # keep the test fast: predict the (16-node) validation inputs instead
                test = {key: value for key, value in val.items() if not key.startswith("output_")}
            predictions, details = clrs_train.train_and_predict(candidate.build_processor, algorithm,
                                                                {"train": data["train"], "val": val, "test": test},
                                                                steps=1, eval_every=1, hidden=8, batch_size=2, threads=2)
            self.assertEqual(sorted(predictions), sorted(name for name, (stage, _, _) in S.SPECS[algorithm].items()
                                                         if stage == S.OUTPUT), algorithm)
            for name, value in predictions.items():
                self.assertEqual(value.shape[0], 2, (algorithm, name))
            self.assertTrue(np.isfinite(details["final_train_loss"]), algorithm)


@unittest.skipUnless(HAVE_TORCH and HAVE_DATA, "needs torch and the prepared CLRS-30 files")
class SandboxedScoreTests(unittest.TestCase):
    def test_seed_scores_and_is_deterministic(self):
        from autoresearch_suite import clrs_eval

        seed = program(SEED)
        first, details = clrs_eval.score(seed, tiny("bfs"))
        second, _ = clrs_eval.score(seed, tiny("bfs"))
        self.assertTrue(0.0 <= first <= 1.0)
        self.assertEqual(first, second)
        self.assertEqual(details["test_nodes"], 64)
        self.assertEqual(details["test_samples"], 2)
        self.assertEqual(details["processor_parameters"], 258872)

    @support.slow
    def test_permutation_and_edge_pointer_outputs(self):
        from autoresearch_suite import clrs_eval

        candidate = program(LINEAR)
        for algorithm in ("insertion_sort", "matrix_chain_order"):
            value, details = clrs_eval.score(candidate, tiny(algorithm, test_samples=1, steps=1))
            self.assertTrue(0.0 <= value <= 1.0, algorithm)

    @support.slow
    def test_invalid_candidates_are_rejected(self):
        from autoresearch_suite import clrs_eval

        data_file = D.data_dir() / "bfs.npz"
        cases = {
            "wrong shape": LINEAR.replace("return torch.relu(self.layer(torch.cat([node_fts, hidden], dim=-1)))",
                                          "return self.layer(torch.cat([node_fts, hidden], dim=-1))[:, :1]"),
            "non-finite": LINEAR.replace("return torch.relu(", "return float('nan') * torch.relu("),
            "too many parameters": LINEAR.replace("torch.nn.Linear(2 * hidden, hidden)", "torch.nn.Linear(2 * hidden, 9000)"),
            "reads the data": LINEAR.replace("super().__init__()", f"super().__init__()\n        import numpy\n"
                                             f"        self.answers = numpy.load({str(data_file)!r})"),
            "forbidden import": "import os\n" + LINEAR,
            "no entry point": LINEAR.replace("def build_processor", "def make_processor"),
            "not a module": LINEAR.replace("return Processor(hidden_dim)", "return lambda *args: args[0]"),
            "fails later": LINEAR.replace("return torch.relu(", "assert node_fts.shape[1] < 30\n        return torch.relu("),
        }
        for label, source in cases.items():
            with self.subTest(label), self.assertRaises(CandidateError):
                clrs_eval.score(program(source), tiny("bfs", steps=1))

    def test_wall_clock_limit(self):
        from autoresearch_suite import clrs_eval

        slow = LINEAR.replace("return torch.relu(", "for _ in range(20000):\n            node_fts = node_fts + 0.0\n"
                                                       "        return torch.relu(")
        with self.assertRaises(CandidateError) as caught:
            clrs_eval.score(program(slow), tiny("bfs", steps=50, wall_seconds=6))
        self.assertIn("wall-clock", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
