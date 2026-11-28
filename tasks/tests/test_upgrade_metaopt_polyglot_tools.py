"""Regression tests of the metaopt, polyglot and llm_tools evaluators in this library's layout: the five wrappers that hand over
the candidate unexecuted, the position-dependent table the scaling-law evaluator accepted, hostile or non-numeric replies, and
independent recomputations of the telemetry, attack and retrieval scores. Model clients, the container runner, the retriever and
the classifier are stand-ins; only the tests that say so use prepared data, and they skip without it. Nothing uses the network."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
import arena_task_paths  # noqa: E402
from autoresearch_suite import llm as llm_client  # noqa: E402
from autoresearch_suite import llm_tools_autoda as autoda  # noqa: E402
from autoresearch_suite import llm_tools_common as common  # noqa: E402
from autoresearch_suite import llm_tools_hover as hover  # noqa: E402
from autoresearch_suite import llm_tools_telemetry as tel  # noqa: E402
from autoresearch_suite import metaopt_eob as eob  # noqa: E402
from autoresearch_suite import metaopt_lasr as lasr  # noqa: E402
from autoresearch_suite import polyglot_data, polyglot_image, polyglot_runner, runtime, software_sandbox  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

LASR, EOB, POLYGLOT = "lasr-big-bench-llm-scaling-law-discovery", "eob-eob-bbob-benchmark-synthesis", "dgm-aider-polyglot"
TELEMETRY, AUTODA, HOVER = "evox-telemetry-repair", "l-autoda-cifar-10-decision-based-adversarial-attack", "mipro-hover"
SANDBOXED = (LASR, POLYGLOT, TELEMETRY, AUTODA, HOVER)
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-"
# What a hostile candidate would do while it is imported: replace the scoring function of its evaluator.
TAMPER = '''
try:
    from autoresearch_suite import runtime, metaopt_lasr, llm_tools_telemetry, llm_tools_autoda, llm_tools_hover, polyglot_agent
    runtime.TAMPERED = True
    metaopt_lasr.score = lambda module, P: (0.0, {})
    llm_tools_autoda.evaluate = lambda module, P: (0.0, {})
    llm_tools_telemetry.evaluate = llm_tools_hover.evaluate = lambda module, P: (1.0, {})
    polyglot_agent.evaluate = lambda module, P: {"score": 1.0, "details": {}}
except Exception:
    pass
'''
LAW = '''
import numpy as np
N_PARAMS = 2
def scaling_law(X, params):
    return params[0] + params[1] * np.log2(X["training_steps"])
'''


def wrapper(task):
    """A package's evaluator.py as a module, executed from source so that no bytecode is written."""
    path = ROOT / "suite" / task / "evaluator.py"
    module = types.ModuleType("evaluator_" + task.replace("-", "_"))
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)  # noqa: S102 - the shipped wrapper
    return module


def lasr_rows(n=400, seed=5):
    """Synthetic rows with six input combinations and grades that are all different numbers."""
    rng = np.random.default_rng(seed)
    X = {name: np.ones(n) for name in lasr.FEATURES}
    X["training_steps"], X["shots"] = rng.choice([1.0, 2.0, 4.0], n), rng.choice([0.0, 1.0], n)
    y = np.clip(0.3 + 0.05 * np.log2(X["training_steps"]) + rng.normal(0, 0.05, n), 0, 1)
    return X, y, [f"{a}|{b}" for a, b in zip(X["training_steps"], X["shots"])], ["task"] * n


class Case(unittest.TestCase):
    def candidate(self, source):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        return str(path)

    def patch(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_tampering_gains_nothing(self, task, source, **parameters):
        module = wrapper(task)
        module.P.update(parameters)
        evaluate = module.evaluate
        self.addCleanup(lambda: runtime.__dict__.pop("TAMPERED", None))
        honest, hostile = evaluate(self.candidate(source)), evaluate(self.candidate(TAMPER + source))
        self.assertFalse(hasattr(runtime, "TAMPERED"), task)
        self.assertIsNotNone(honest["score"], honest.get("error"))
        self.assertEqual(hostile["score"], honest["score"], task)
        return honest


class LayoutTests(Case):
    def test_sandboxed_wrappers_load_the_candidate_unexecuted(self):
        for task in SANDBOXED:
            text = (ROOT / "suite" / task / "evaluator.py").read_text()
            self.assertIn("sandboxed.run(program_path, task", text, task)
            self.assertNotIn("runtime.run(", text, task)
        # The benchmark-synthesis scorer calls the candidate about 10,000 times in its own process, by design.
        self.assertIn("runtime.run(program_path, task", (ROOT / "suite" / EOB / "evaluator.py").read_text())

    def test_prepared_data_and_caches_are_outside_the_library(self):
        for directory, name in ((lasr.data_dir(), "metaopt"), (polyglot_data.data_dir(), "polyglot"), (common.data_dir(), "llm_tools")):
            self.assertEqual(directory, arena_task_paths.data_root() / name)
            self.assertFalse(directory.is_relative_to(ROOT))
        own = [ROOT / "scripts" / f"prepare_{name}.py" for name in ("metaopt", "polyglot", "llm_tools")]
        own += [path for stem in ("metaopt_", "polyglot_", "llm_tools_") for path in (ROOT / "suite/_lib/autoresearch_suite").glob(stem + "*.py")]
        self.assertEqual(len(own), 16)
        for path in own:  # no location of the source repository's layout is left in code, messages or docstrings
            self.assertFalse(re.findall(r'"\.cache"|suite/_lib/data|\.cache/[\w-]+-raw', path.read_text()), path.name)
        lock = json.loads((ROOT / "catalog/polyglot-data-lock.json").read_text())
        self.assertFalse([row["target"] for row in lock["sources"] if row["target"].startswith((".cache", "suite/"))])

    def test_anchors_are_computed_once_however_they_are_requested(self):
        self.assertIs(eob.anchors(60, 3, 2), eob.anchors(n_samples=60, sample_seed=3, instances=2))

    def test_benchmark_function_values_must_be_real_numbers(self):  # strings, booleans and complex values were converted
        for bad in (lambda X: [str(v) for v in X[:, 0]], lambda X: X[:, 0] > 0, lambda X: X[:, 0] + 1j):
            with self.assertRaisesRegex(CandidateError, "real numbers"):
                eob._call(bad, np.ones((3, 10)))


class ScalingLawTests(Case):
    def setUp(self):
        self.rows = lasr_rows()
        self.patch(lasr, "load_rows", lambda: self.rows)
        self.X, self.y = self.rows[0], self.rows[1]
        self.train, self.test = lasr.split_indices(len(self.y))

    def score(self, source):
        return lasr.score(types.SimpleNamespace(__file__=self.candidate(source), __candidate_source__=True), {})

    def table_candidates(self):
        """The held-out grades by position: in a docstring, and in one integer literal (both within the source limits)."""
        levels = np.rint(self.y[self.test] * 63).astype(int)
        number = 0
        for value in levels[::-1]:
            number = number * 64 + int(value)
        docstring = f'''
import numpy as np
N_PARAMS = 1
def table():
    """{"".join(ALPHABET[v] for v in levels)}"""
def scaling_law(X, params):
    n = len(X["shots"])
    if n != len(table.__doc__):
        return np.full(n, params[0])
    return np.array(["{ALPHABET}".index(c) for c in table.__doc__]) / 63.0 + 0 * params[0]
'''
        literal = f'''
import numpy as np
N_PARAMS = 1
T = {number}
def scaling_law(X, params):
    n = len(X["shots"])
    if n != {len(levels)}:
        return np.full(n, params[0])
    return np.array([(T >> (6 * i)) & 63 for i in range(n)]) / 63.0 + 0 * params[0]
'''
        return docstring, literal

    def test_a_table_of_held_out_grades_by_position_is_rejected(self):
        for source in self.table_candidates():
            self.assertLessEqual(lasr.source_profile(source)["non_small_literals"], lasr.MAX_LITERALS)
            with self.assertRaisesRegex(CandidateError, "order of the rows"):
                self.score(source)

    def test_the_table_is_exact_while_rows_keep_their_order(self):
        # The defect: a recomputation in the same row order confirmed these predictions, whose error is the quantisation only.
        for source in self.table_candidates():
            predicted = lasr._child(source + lasr.HARNESS, "_artask_predict", [lasr._columns(self.X, self.test), [0.0]], 30)
            self.assertLess(float(np.mean((np.array(predicted) - self.y[self.test]) ** 2)), 1e-4)
        self.assertGreater(lasr.memorisation_floor(self.X, self.y, self.test), 1e-3)

    def test_an_honest_law_scores_the_same_in_every_evaluation(self):
        first, second = self.score(LAW), self.score(LAW)
        self.assertEqual(first[0], second[0])  # the random order of the recomputation does not enter the score
        design = np.column_stack([np.ones(len(self.train)), np.log2(self.X["training_steps"][self.train])])
        a, b = np.linalg.lstsq(design, self.y[self.train], rcond=None)[0]
        expected = np.mean((a + b * np.log2(self.X["training_steps"][self.test]) - self.y[self.test]) ** 2)
        self.assertAlmostEqual(first[0], float(expected), places=10)
        self.assertGreaterEqual(first[0], first[1]["memorisation_floor_mse"])

    def test_a_table_per_input_combination_reaches_the_floor_and_no_further(self):
        keys = list(zip(self.X["training_steps"][self.test], self.X["shots"][self.test]))
        means = {key: float(np.mean([v for k, v in zip(keys, self.y[self.test]) if k == key])) for key in sorted(set(keys))}
        source = ("import numpy as np\nN_PARAMS = 1\nT = {" + ", ".join(f"({int(a)}, {int(b)}): {m!r}" for (a, b), m in means.items()) + "}\n"
                  "def scaling_law(X, params):\n"
                  "    return np.array([T[(int(a), int(b))] for a, b in zip(X['training_steps'], X['shots'])]) + 0 * params[0]\n")
        mse, details = self.score(source)
        self.assertAlmostEqual(mse, details["memorisation_floor_mse"], places=12)
        self.assertLess(mse, details["reference_group_mean_mse"])  # what the source limits do not prevent

    def test_the_child_never_receives_held_out_grades(self):
        sent, execute = [], software_sandbox.execute
        self.patch(software_sandbox, "execute", lambda source, entry, args=(), **kw: sent.append(json.dumps(args)) or execute(source, entry, args, **kw))
        self.score(LAW)
        self.assertEqual(len(sent), 3)  # N_PARAMS, the fit, the recomputation
        text = "".join(sent)
        self.assertFalse([v for v in self.y[self.test] if repr(float(v)) in text])
        self.assertTrue(all(repr(float(v)) in text for v in self.y[self.train][:20]))

    def test_n_params_is_read_in_the_child(self):
        for declaration in ("N_PARAMS = 6", "N_PARAMS = True", "N_PARAMS = 2.0", "N_PARAMS = 10 ** 400", "pass"):
            with self.assertRaisesRegex(CandidateError, "N_PARAMS must be an integer"):
                self.score(LAW.replace("N_PARAMS = 2", declaration))
        self.assertEqual(self.score(LAW.replace("N_PARAMS = 2", "N_PARAMS = len('ab')"))[1]["n_params"], 2)

    def test_replies_forged_by_the_child_are_the_candidates_failure(self):
        for reply in ({"params": [10 ** 400, 1], "test_pred": [0.5] * len(self.test), "train_mse": 0.1},
                      {"params": [1, 1], "test_pred": [0.5] * len(self.test), "train_mse": float("nan")},
                      {"params": ["a", "b"], "test_pred": None, "train_mse": 0.1}, [1, 2],
                      {"params": ["1", "1"], "test_pred": ["0.5"] * len(self.test), "train_mse": 0.1},  # strings of digits
                      {"params": [True, 1], "test_pred": [True] * len(self.test), "train_mse": 0.1}):
            self.patch(lasr, "_child", lambda source, entry, args, cpu, reply=reply: {"type": "int", "value": 2} if entry == "_artask_declared" else reply)
            with self.assertRaises(CandidateError):
                self.score(LAW)

    def test_wrapper_does_not_execute_the_candidate(self):
        honest = self.assert_tampering_gains_nothing(LASR, LAW)
        self.assertEqual((honest["valid"], honest["direction"]), (True, "minimize"))
        original = lasr.score
        self.addCleanup(lambda: (setattr(lasr, "score", original), sys.modules.pop("candidate_program", None)))
        tamper = "from autoresearch_suite import metaopt_lasr\nmetaopt_lasr.score = lambda module, P: (0.0, {})\n"
        runtime.load_program(self.candidate(tamper + LAW))  # the leak this closes: importing runs the tampering
        self.assertIsNot(lasr.score, original)


@unittest.skipUnless((lasr.data_dir() / lasr.DATA_FILE).is_file(), "prepared BIG-bench data missing (python scripts/prepare_metaopt.py)")
class ScalingLawDataTests(Case):
    def test_seed_reproduces_the_recorded_score_and_a_closed_form_fit(self):
        recorded = json.loads((ROOT / "validation/metaopt-checks.json").read_text())["tasks"][LASR]["seed"]["score"]
        result = wrapper(LASR).evaluate(str(ROOT / "suite" / LASR / "initial_program.py"))
        self.assertAlmostEqual(result["score"], recorded, places=10)
        X, y, _, _ = lasr.load_rows()
        train, test = lasr.split_indices(len(y))
        self.assertEqual((len(train), len(test)), (43049, 10763))
        feature = np.log10(X["training_steps"] * X["batch_size"])  # the seed is linear in its two constants
        a, b = np.linalg.lstsq(np.column_stack([np.ones(len(train)), feature[train]]), y[train], rcond=None)[0]
        self.assertAlmostEqual(result["score"], float(np.mean((a + b * feature[test] - y[test]) ** 2)), places=10)
        self.assertLess(result["memorisation_floor_mse"], result["reference_group_mean_mse"])
        self.assertLess(result["reference_group_mean_mse"], result["score"])


telemetry_cell = lambda value: repr({"ground_truth": value, "perturbed": value, "corrected": None, "confidence": None})  # noqa: E731
TELEMETRY_ROWS = [{"timestamp": f"t{i}", "low_A_egress_to_B": telemetry_cell(10.0 + i), "low_A_ingress_from_B": telemetry_cell(4.0),
                   "low_B_egress_to_A": telemetry_cell(4.0), "low_B_ingress_from_A": telemetry_cell(10.0 + i)} for i in range(6)]
ECHO = '''
def run_repair(telemetry, topology):
    return {k: {"rx_rate": (d["rx_rate"], d["rx_rate"], 0.9), "tx_rate": (d["tx_rate"], d["tx_rate"], 0.9),
                "interface_status": (d["interface_status"], d["interface_status"], 1.0)} for k, d in telemetry.items()}
'''


class TelemetryTests(Case):
    def test_score_matches_a_hand_computation(self):
        case = {"timestamp": "t", "bug": "scaled", "truth": {"A_to_B": {"rx_rate": 4.0, "tx_rate": 10.0, "interface_status": "up"}}}
        output = {"A_to_B": {"rx_rate": [3.0, 6.0, 0.5], "tx_rate": [9.0, 10.0, 0.9], "interface_status": ["up", "up", 1.0]}}
        # counters: 1 - (0.5 + 0) / 2; calibration: rx accuracy 1 - 2/6 is poor at confidence 0.5, so 0.8 - 0.5 * (2/6) * 2, and tx 1.
        score, details = tel.score_outputs([case], [output])
        self.assertAlmostEqual(score, 0.75 * 0.75 + 0.05 * 1.0 + 0.2 * ((0.8 - 0.5 * (2 / 6) * 2) + 1.0) / 2, places=12)
        self.assertEqual((details["counter_repair_accuracy"], details["confidence_calibration"]), (0.75, 0.733333))

    def test_malformed_numbers_are_the_candidates_failure(self):
        truth = {"A_to_B": {"rx_rate": 4.0, "tx_rate": 10.0, "interface_status": "up"}}
        good = {"rx_rate": (0, 4.0, 1.0), "tx_rate": (0, 10.0, 1.0), "interface_status": ("up", "up", 1.0)}
        for field, value in (("rx_rate", (0, 10 ** 400, 1.0)), ("tx_rate", (0, 1.0, 10 ** 400)), ("rx_rate", (0, True, 1.0)),
                             ("rx_rate", (0, "4", 1.0)), ("tx_rate", (0, 1.0, float("nan"))), ("rx_rate", (0, float("inf"), 1.0))):
            with self.assertRaises(CandidateError, msg=value):
                tel.validate({"A_to_B": {**good, field: value}}, truth, 0)
        self.assertEqual(tel.validate({"A_to_B": {**good, "rx_rate": (0, 10 ** 30, 1)}}, truth, 0)["A_to_B"]["rx_rate"], (1e30, 1.0))

    def test_wrapper_does_not_execute_the_candidate(self):
        self.patch(tel, "load_rows", lambda: TELEMETRY_ROWS)
        honest = self.assert_tampering_gains_nothing(TELEMETRY, ECHO)
        self.assertTrue(0.5 < honest["score"] < 1.0)
        nan = wrapper(TELEMETRY).evaluate(self.candidate(ECHO.replace('d["rx_rate"], 0.9', 'float("nan"), 0.9')))
        self.assertEqual((nan["valid"], nan["score"]), (False, None))

    @unittest.skipUnless((common.data_dir() / tel.CSV_NAME).is_file(), "prepared telemetry data missing (python scripts/prepare_llm_tools.py telemetry)")
    def test_prepared_data_reproduce_the_recorded_scores(self):
        recorded = json.loads((ROOT / "validation/llm_tools-checks.json").read_text())["tasks"][TELEMETRY]
        seed = wrapper(TELEMETRY).evaluate(str(ROOT / "suite" / TELEMETRY / "initial_program.py"))
        self.assertAlmostEqual(seed["score"], recorded["seed_score"], places=12)
        echo = wrapper(TELEMETRY).evaluate(self.candidate(ECHO))
        self.assertAlmostEqual(echo["score"], recorded["reference_check"]["result"]["identity_echo"]["port"], places=12)


def brightness(model, images, batch=8):
    """Stand-in classifier: class 1 for images whose mean intensity exceeds 0.5, else class 0."""
    return (np.asarray(images, dtype=np.float64).reshape(len(images), -1).mean(axis=1) > 0.5).astype(np.int64)


def closer(self, proposals):
    norm = lambda images: np.linalg.norm((self.originals - images).reshape(self.n, -1), axis=1)  # noqa: E731
    return norm(proposals) < norm(self.best)


class AttackTests(Case):
    P = {"images": 2, "queries": 30, "seed": 0, "s0": 0.05}

    def setUp(self):
        self.originals = np.full((2,) + autoda.SHAPE, 0.2, dtype=np.float32)
        self.starts = np.full((2,) + autoda.SHAPE, 0.9, dtype=np.float32)
        self.labels = np.zeros(2, dtype=np.int64)
        self.patch(autoda, "load_model", lambda threads=4: None)
        self.patch(autoda, "attack_set", lambda name, count: (self.originals, self.labels, self.starts, {}))
        self.patch(autoda, "classify", brightness)
        if not importlib.util.find_spec("torch"):  # the distance comparison is the only use of torch in the loop
            self.patch(autoda.Attack, "_closer", closer)

    def evaluate(self, source):
        return autoda.evaluate(types.SimpleNamespace(__file__=self.candidate(source), __candidate_source__=True), self.P)

    def test_score_matches_an_independent_run_of_algorithm_1(self):
        namespace = {}
        exec(compile(seed_source(AUTODA), "seed.py", "exec"), namespace)  # noqa: S102 - the task's own seed
        rng, best, p, s = np.random.default_rng(0), self.starts.copy(), np.zeros(2), np.full(2, 0.05)
        flat = lambda images: (self.originals - images).reshape(2, -1)  # noqa: E731
        for _ in range(self.P["queries"]):
            noise = rng.standard_normal(self.originals.shape)
            proposals = np.stack([np.clip(namespace["draw_proposals"](self.originals[i].copy(), best[i].copy(), noise[i].copy(), s[i:i + 1].copy()), 0, 1)
                                  for i in range(2)]).astype(np.float32)
            accepted = (brightness(None, proposals) != self.labels) & (np.linalg.norm(flat(proposals), axis=1) < np.linalg.norm(flat(best), axis=1))
            p = 0.95 * p + 0.05 * accepted
            s = s * np.where(p < 0.25, 0.5 + 2 * p, 5 / 6 + 2 * p / 3) ** 0.1  # the paper's f(p)
            best[accepted] = proposals[accepted]
        expected = float(np.linalg.norm(flat(best).astype(np.float64), axis=1).mean())
        score, details = self.evaluate(seed_source(AUTODA))
        self.assertAlmostEqual(score, expected, places=6)
        self.assertLess(score, details["initial_mean_l2"])
        self.assertEqual((details["queries_per_image"], details["reverified_adversarial"]), (30, 2))

    def test_the_child_gets_no_labels_and_no_query_beyond_the_budget(self):
        sent, call = [], common.call_candidate
        self.patch(common, "call_candidate", lambda *args, **kw: sent.append(kw["payload"]) or call(*args, **kw))
        self.evaluate(seed_source(AUTODA))
        self.assertEqual(sorted(sent[0]), ["n", "noise_seed", "originals", "s0", "shape", "starts", "steps"])
        thief = ("import sys\ndef draw_proposals(org, best, noise, s):\n    frame = sys._getframe(1)\n    local = frame.f_locals\n"
                 "    import base64\n    batch = base64.b64encode(local['best'].tobytes()).decode()\n"
                 "    local['oracle']({'op': 'query', 'args': [local['step'], batch], 'kwargs': {}})\n    return best\n")
        with self.assertRaisesRegex(CandidateError, "protocol violated"):  # its own query uses up the step; the harness's is refused
            self.evaluate(thief)

    def test_wrapper_does_not_execute_the_candidate(self):
        honest = self.assert_tampering_gains_nothing(AUTODA, seed_source(AUTODA), queries=40)
        self.assertEqual((honest["valid"], honest["direction"], honest["queries_per_image"]), (True, "minimize", 40))
        for bad, message in (("def draw_proposals(o, b, n, s):\n    return b * float('nan')\n", "non-finite"),
                             ("def draw_proposals(o, b, n, s):\n    return 'x'\n", "failed in the sandbox")):
            result = wrapper(AUTODA).evaluate(self.candidate(bad))
            self.assertEqual((result["valid"], result["score"]), (False, None))
            self.assertIn(message, result["error"])


CORPUS = ["Eiffel Tower | The Eiffel Tower is a tower in Paris.", "Gustave Eiffel | Gustave Eiffel was a French engineer.",
          "Paris | Paris is the capital of France.", "Berlin | Berlin is the capital of Germany."] + [f"Filler {i} | A river." for i in range(8)]
CLAIMS = [{"uid": f"c{i}", "claim": "tower paris engineer eiffel capital", "gold_titles": ["Eiffel Tower", "Gustave Eiffel", "Paris"]} for i in range(4)]


class FakeIndex:
    """Ranks the corpus by the number of query words a document contains."""

    def retrieve(self, tokens, k, n_threads=1, show_progress=False):
        scores = [sum(word in doc.lower() for word in tokens) for doc in CORPUS]
        order = sorted(range(len(CORPUS)), key=lambda i: (-scores[i], i))[:k]
        return [order], [[float(scores[i]) for i in order]]


class FakeLLM:
    def __init__(self, model="fake", budget_usd=0.5, cache=None, reply="def f():\n    return 42\n", error=None):
        self.model, self.budget, self.reply, self.error, self.calls = model, budget_usd, reply, error, 0

    def __call__(self, prompt, **options):
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply

    def usage(self):
        return {"model": self.model, "requests": self.calls, "cached_responses": 0, "cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


class RetrievalTests(Case):
    def setUp(self):
        fake = {"bm25s": types.SimpleNamespace(tokenize=lambda query, **kw: query.lower().split()), "stemmer": None, "index": FakeIndex(), "corpus": CORPUS}
        self.patch(hover, "retriever", lambda: fake)
        self.patch(common, "read_json", lambda name: {"test": CLAIMS})
        self.client = FakeLLM(reply="eiffel engineer")
        self.patch(hover.core, "task_llm", lambda P: self.client)

    def test_wrapper_scores_retrieved_documents_only_and_does_not_execute_the_candidate(self):
        honest = self.assert_tampering_gains_nothing(HOVER, seed_source(HOVER))
        self.assertEqual((honest["score"], honest["items"], honest["mean_gold_recall"]), (1.0, 4, 1.0))
        claimed = "def solve(claim, search, llm):\n    return ['Eiffel Tower | x', 'Gustave Eiffel | y', 'Paris | z']\n"
        partial = "def solve(claim, search, llm):\n    search('eiffel', 2)\n"
        crash = "def solve(claim, search, llm):\n    search(claim, 10)\n    raise ValueError('after retrieving every gold document')\n"
        results = [wrapper(HOVER).evaluate(self.candidate(source)) for source in (claimed, partial, crash)]
        self.assertEqual([r["score"] for r in results], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(results[1]["mean_gold_recall"], 2 / 3, places=6)
        self.assertEqual(results[2]["crashes"], 4)  # a claim whose program fails scores 0 whatever it retrieved

    def test_an_exhausted_model_budget_invalidates_the_evaluation(self):
        self.client = FakeLLM(error=CandidateError("per-evaluation model budget of $0.50 exhausted after 3 requests ($0.4990 spent)"))
        swallow = "def solve(claim, search, llm):\n    search(claim, 10)\n    try:\n        llm('x')\n    except RuntimeError:\n        pass\n"
        result = wrapper(HOVER).evaluate(self.candidate(swallow))
        self.assertEqual((result["valid"], result["score"]), (False, None))
        self.assertIn("budget", result["error"])

    def test_the_client_refuses_a_request_its_budget_does_not_cover(self):
        cache = types.SimpleNamespace(get=lambda key: None, put=lambda *row: None)
        self.patch(llm_client.urllib.request, "urlopen", mock.Mock(side_effect=AssertionError("no request may be sent")))
        client = llm_client.LLM(model="google/gemini-2.5-flash", budget_usd=0.01, cache=cache, api_key="unused")
        with self.assertRaisesRegex(CandidateError, "per-evaluation model budget"):
            client("x", max_tokens=8192)  # reserved before the request: 8,192 completion tokens at the list price exceed 0.01
        self.assertEqual(client.usage()["requests"], 0)


AGENT = '''
def solve(exercise, llm):
    slug = exercise["id"].split("__")[1]
    if slug == "alpha":
        return {"alpha.py": llm("solve " + exercise["instructions"])}
    if slug == "beta":
        return {}
    raise ValueError("no idea")
'''


class PolyglotTests(Case):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root, records = Path(directory.name), {}
        for slug in ("alpha", "beta", "gamma"):
            folder = root / "exercises" / "python" / slug
            files = {"solution": [f"{slug}.py"], "test": [f"{slug}_test.py"], "example": [".meta/example.py"]}
            texts = {f"{slug}.py": "def f():\n    pass\n", f"{slug}_test.py": "EXPECTED_SECRET = 42\n", ".meta/example.py": "REFERENCE_SECRET = 42\n",
                     ".meta/config.json": json.dumps({"files": files}), ".docs/instructions.md": f"Write {slug}.\n"}
            for relative, text in texts.items():
                (folder / relative).parent.mkdir(parents=True, exist_ok=True)
                (folder / relative).write_text(text)
            records[f"python__{slug}"] = {"language": "python", "slug": slug, "files": files, "sha256": polyglot_data.tree_digest(folder, polyglot_data.list_files(folder))}
        (root / "manifest.json").write_text(json.dumps({"benchmark": {"commit": polyglot_data.BENCHMARK_COMMIT}, "subsets": {"small": sorted(records)}, "exercises": records}))
        self.patch(polyglot_data, "data_dir", lambda: root)
        polyglot_data.manifest.cache_clear()
        self.addCleanup(polyglot_data.manifest.cache_clear)
        self.patch(polyglot_image, "require_image", lambda: {"tag": "stand-in", "id": None, "architecture": None})
        self.patch(llm_client, "Cache", lambda: None)
        self.clients = []
        self.patch(llm_client, "LLM", lambda **kw: self.clients.append(FakeLLM(**kw)) or self.clients[-1])
        self.tested, self.sent, execute = [], [], software_sandbox.execute
        self.patch(polyglot_runner, "run_tests", self.run_tests)
        self.patch(software_sandbox, "execute", lambda source, entry, args=(), **kw: self.sent.append(json.dumps(args)) or execute(source, entry, args, **kw))

    def run_tests(self, image, exercise_id, slug, language, files, **limits):
        """Stand-in for the container: the hidden tests pass when the solution returns 42."""
        self.tested.append((exercise_id, sorted(files)))
        resolved = b"return 42" in files[f"{slug}.py"][0] and b"EXPECTED_SECRET" in files[f"{slug}_test.py"][0]
        return {"id": exercise_id, "resolved": resolved, "exit_code": 0 if resolved else 1, "timed_out": False, "test_seconds": 0.0,
                "framework_summary": {}, "exit0_without_summary": False, "output_tail": ""}

    def test_pass_at_1_with_stand_in_model_and_container(self):
        honest = self.assert_tampering_gains_nothing(POLYGLOT, AGENT)
        self.assertAlmostEqual(honest["score"], 1 / 3)
        self.assertEqual((honest["resolved"], honest["agent_crashes"], honest["valid"]), (1, 1, None))
        self.assertEqual({r["id"]: r["outcome"] for r in honest["per_exercise"]},
                         {"python__alpha": "resolved", "python__beta": "no_edit", "python__gamma": "crash"})
        self.assertEqual({name for name, _ in self.tested}, {"python__alpha"})  # nothing is tested without an edit
        self.assertEqual(self.tested[0][1], ["alpha.py", "alpha_test.py"])  # the hidden run gets the tests, never .meta
        self.assertTrue(self.sent and not [text for text in self.sent if "SECRET" in text])  # the agent gets neither
        self.assertTrue(all(abs(client.budget - 0.5 / 3) < 1e-12 for client in self.clients))  # the budget is split by exercise

    def test_contract_violations_invalidate_the_evaluation(self):
        for source, message in (("def solve(exercise, llm):\n    return ['alpha.py']\n", "must return a dict"),
                                ("def solve(exercise, llm):\n    return {'alpha_test.py': 'x'}\n", "not a solution file"),
                                ("def other():\n    pass\n", "must define solve")):
            result = wrapper(POLYGLOT).evaluate(self.candidate(source))
            self.assertEqual((result["valid"], result["score"]), (False, None))
            self.assertIn(message, result["error"])


@unittest.skipUnless((polyglot_data.data_dir() / "manifest.json").is_file(), "prepared polyglot data missing (python scripts/prepare_polyglot.py)")
class PolyglotDataTests(Case):
    def test_no_prepared_exercise_shows_tests_or_reference_solutions(self):
        for exercise_id in polyglot_data.subset_ids("all"):
            exercise = polyglot_data.Exercise(exercise_id)
            visible, hidden = exercise.visible(), [rel for kind in ("test", "example", "editor") for rel in exercise.files.get(kind, [])]
            self.assertFalse(set(hidden) & (set(visible["files"]) | set(visible["context"])), exercise_id)
            shown = visible["instructions"] + "".join(visible["files"].values()) + "".join(visible["context"].values())
            for rel in hidden:
                text = exercise.read(rel).strip() if (exercise.root / rel).is_file() else ""
                self.assertFalse(len(text) > 200 and text in shown, (exercise_id, rel))
