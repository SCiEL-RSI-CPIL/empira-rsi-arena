"""Regression tests of the agent_bench port: cache location, candidate isolation, the M3ToolEval calculator, budgets.

They need neither the network nor a model: episodes run on the toy adapter with a fake model client. Tests of the
prepared benchmark data are skipped when the data are absent (scripts/prepare_agent_bench.py).
"""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import agent_bench_balrog as balrog  # noqa: E402
from autoresearch_suite import agent_bench_core as core  # noqa: E402
from autoresearch_suite import agent_bench_m3tool as m3tool  # noqa: E402
from autoresearch_suite import agent_bench_travel as travel  # noqa: E402
from autoresearch_suite import llm, llm_bench_core, runtime, sandboxed  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import evaluator, load, needs  # noqa: E402

TASKS = ("agentsquare-alfworld", "agentsquare-m3tooleval", "agentsquare-pddl", "agentsquare-scienceworld", "agentsquare-travelplanner",
         "hsi-babaisai-breakstop", "hsi-babaisai-goto", "hsi-babaisai-make", "hsi-babyai", "hsi-crafter", "hsi-minihack")
BALROG_EPISODES = {"hsi-babyai": 10, "hsi-crafter": 2, "hsi-babaisai-goto": 9, "hsi-babaisai-breakstop": 6, "hsi-babaisai-make": 12, "hsi-minihack": 8}
M3TOOL_DATA = (m3tool.data_dir() / "m3tooleval" / "tasks" / "base.py").is_file()
TRAVEL_DATA = (travel.data_dir() / "travel" / "flights.csv").is_file() and (llm_bench_core.DATA / "travelplanner_validation.json").is_file()
ZERO = "def act(instruction, observation, valid_actions, history, llm):\n    return '0'\n"
# What a hostile candidate would do while it is imported: replace the family's scoring entry points.
TAMPER = '''
try:
    from autoresearch_suite import agent_bench_core, runtime
    runtime.TAMPERED = True
    agent_bench_core.evaluate_task = lambda module, P: {"score": 1.0, "details": {}}
    agent_bench_core.run_episode = lambda *args, **kwargs: {"id": "x", "score": 1.0}
except Exception:
    pass
'''
# Reads the expected output through CodeAct's calculator (eval) and makes every later answer count as correct.
EXPLOIT = '''
PATCH = "import sys; sys.modules['tasks.base'].Task.check_answer = lambda self, answer: True"


def act(instruction, observation, valid_actions, history, llm):
    if "calculator" not in valid_actions:
        return "Answer: 1"
    if not history:
        return ("Action: calculator, [o.task.expected_output for o in __import__('gc').get_objects() "
                "if type(o).__name__ == 'Adapter'][0] End Action")
    if len(history) == 1:
        return "Action: calculator, exec(bytes.fromhex('" + PATCH.encode().hex() + "').decode()) End Action"
    return "Answer: " + history[0][1]
'''


class FakeLLM:
    def __init__(self, reply="5"):
        self.reply, self.calls = reply, []

    def __call__(self, prompt, **options):
        self.calls.append((prompt, options))
        return self.reply

    def usage(self):
        return {"requests": len(self.calls)}


class MemoryCache:
    def __init__(self, rows=None):
        self.rows = dict(rows or {})

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, *row):
        self.rows[key] = row


def wrapper(task_id):
    spec = importlib.util.spec_from_file_location(f"evaluator_{task_id.replace('-', '_')}", ROOT / "suite" / task_id / "evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Candidates(unittest.TestCase):
    def candidate(self, source):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "candidate.py"
        path.write_text(source)
        return path

    def toy(self, source, llm_client=None, **params):
        P = {"targets": [1, 7], "max_steps": 2, **params}
        with mock.patch("autoresearch_suite.llm_bench_core.task_llm", return_value=llm_client or FakeLLM()):
            return core.evaluate(sandboxed.source_program(self.candidate(source)), P, "toy")


class ServerCacheTests(unittest.TestCase):
    def failed_start(self, cache, variable="ARENA_TASK_CACHE_DIR"):
        started, spawn = [], subprocess.Popen
        with mock.patch.dict(os.environ, {variable: cache}), \
                mock.patch.object(core.subprocess, "Popen", side_effect=lambda *a, **k: started.append(spawn(*a, **k)) or started[-1]):
            if variable != "ARENA_TASK_CACHE_DIR":
                os.environ.pop("ARENA_TASK_CACHE_DIR", None)
            with self.assertRaises(RuntimeError) as raised:
                core.EnvServer("alfworld", {})
        return str(raised.exception), started[0]

    def test_the_server_reads_the_clients_task_cache(self):
        # The server's environment is built from scratch; it used to fall back to the default cache in the home directory.
        with tempfile.TemporaryDirectory() as cache:
            message, _ = self.failed_start(cache)
            self.assertIn(str(Path(cache).resolve() / "data" / "agent_bench" / "alfworld" / "games.json"), message)
            message, _ = self.failed_start(cache, "XDG_CACHE_HOME")
            self.assertIn(str(Path(cache).resolve() / "empira" / "tasks" / "data" / "agent_bench" / "alfworld" / "games.json"), message)

    def test_a_server_whose_adapter_cannot_start_is_closed(self):
        with tempfile.TemporaryDirectory() as cache:
            _, process = self.failed_start(cache)
            self.assertIsNotNone(process.poll())
            self.assertTrue(process.stdin.closed and process.stdout.closed)

    def test_the_server_never_works_inside_the_task_library(self):
        seen, spawn = [], subprocess.Popen
        with tempfile.TemporaryDirectory() as cache, mock.patch.dict(os.environ, {"ARENA_TASK_CACHE_DIR": cache}), \
                mock.patch.object(core.subprocess, "Popen", side_effect=lambda *a, **k: seen.append(k) or spawn(*a, **k)):
            core.EnvServer("toy", {}).close()
            self.assertEqual(seen[0]["env"]["ARENA_TASK_CACHE_DIR"], str(Path(cache).resolve()))
        self.assertFalse(Path(seen[0]["cwd"]).resolve().is_relative_to(ROOT))


class WrapperIsolationTests(Candidates):
    def evaluate(self, source, task_id="agentsquare-m3tooleval"):
        module = wrapper(task_id)
        with mock.patch.object(module, "P", {"adapter": "toy", "targets": [1, 7], "max_steps": 2}), \
                mock.patch("autoresearch_suite.llm_bench_core.task_llm", return_value=FakeLLM()):
            return module.evaluate(str(self.candidate(source)))

    def test_every_wrapper_loads_the_candidate_unexecuted(self):
        for task_id in TASKS:
            text = (ROOT / "suite" / task_id / "evaluator.py").read_text()
            self.assertIn("sandboxed.run(program_path, task", text, task_id)
            self.assertNotIn("runtime.run(", text, task_id)
            self.assertTrue(load(f"suite/{task_id}/problem.json")["family"].startswith("agent_bench_"), task_id)

    def test_tampering_at_import_time_gains_nothing(self):
        original = (core.evaluate_task, core.run_episode)
        self.addCleanup(lambda: (setattr(core, "evaluate_task", original[0]), setattr(core, "run_episode", original[1]),
                                 runtime.__dict__.pop("TAMPERED", None)))
        honest, hostile = self.evaluate(ZERO), self.evaluate(TAMPER + ZERO)
        self.assertFalse(hasattr(runtime, "TAMPERED"))
        self.assertEqual((core.evaluate_task, core.run_episode), original)
        self.assertEqual((honest["error"], hostile["error"]), (None, None))
        self.assertEqual((honest["score"], hostile["score"]), (0.25, 0.25))  # guess 0: within one of target 1, far from 7
        self.assertEqual((honest["valid"], hostile["valid"]), (None, None))

    def test_the_same_candidate_would_win_if_it_were_imported(self):
        original = (core.evaluate_task, core.run_episode)
        self.addCleanup(lambda: (setattr(core, "evaluate_task", original[0]), setattr(core, "run_episode", original[1]),
                                 runtime.__dict__.pop("TAMPERED", None)))
        runtime.load_program(self.candidate(TAMPER + ZERO))
        self.assertTrue(getattr(runtime, "TAMPERED", False))
        self.assertEqual(core.evaluate_task(None, {})["score"], 1.0)

    def test_contract_failures_are_the_candidates(self):
        cases = {"x = 1\n": "must define act", "act = 3\n": "must define act", "def act(observation):\n    return '1'\n": "act must accept",
                 "def act(:\n": "does not parse", ZERO.replace("'0'", "None"): "must return a str", ZERO.replace("'0'", "True"): "must return a str",
                 ZERO.replace("'0'", "float('nan')"): "must return a str", ZERO.replace("'0'", "b'0'"): "must return a str"}
        for source, expected in cases.items():
            with self.subTest(source=source):
                result = self.evaluate(source)
                self.assertEqual((result["score"], result["valid"]), (None, False))
                self.assertIn(expected, result["error"])


class ProtocolTests(Candidates):
    def test_the_oracle_offers_model_calls_and_steps_only(self):
        source = ("import sys\n\n\ndef act(instruction, observation, valid_actions, history, llm):\n"
                  "    oracle = sys._getframe(1).f_locals['oracle_fn']\n"
                  "    for request in ({'op': 'score'}, {'op': 'reset', 'episode': {'index': 0}}, {'op': 'episodes'}, {'cmd': 'score'}, 'score', None):\n"
                  "        try:\n            raise SystemExit('leak: ' + repr(oracle(request)))\n        except RuntimeError:\n            pass\n"
                  "    return '0'\n")
        out = self.toy(source)
        self.assertEqual((out["score"], out["details"]["crashes"]), (0.25, 0))

    def test_steps_taken_behind_the_loop_count_towards_the_cap(self):
        source = ("import sys\n\n\ndef act(instruction, observation, valid_actions, history, llm):\n"
                  "    oracle = sys._getframe(1).f_locals['oracle_fn']\n"
                  "    for guess in range(10):\n        oracle({'op': 'step', 'action': str(guess)})\n    return '9'\n")
        out = self.toy(source, targets=[8], max_steps=3)
        episode = out["details"]["per_episode"][0]
        self.assertEqual((episode["steps"], episode["ended"], episode["score"]), (3, "crash", 0.0))
        self.assertIn("the episode has ended", episode["crash"])

    def test_long_and_odd_actions_are_bounded(self):
        out = self.toy(ZERO.replace("'0'", "'7' + ' ' * 1000000"), targets=[7])
        self.assertEqual(out["score"], 1.0)
        out = self.toy(ZERO.replace("'0'", "'\\ud800\\x00'"), targets=[7])
        self.assertEqual((out["score"], out["details"]["crashes"]), (0.0, 0))

    def test_scores_outside_the_unit_interval_are_an_evaluator_error(self):
        class Server:
            def call(self, cmd, **payload):
                return {"reset": {"done": True}, "score": {"score": float("nan")}}[cmd]
        with self.assertRaises(RuntimeError):
            core.run_episode(ZERO, Server(), FakeLLM(), {"id": "x"}, 1)


class ModelBudgetTests(Candidates):
    def test_model_options_are_clamped_and_checked(self):
        client = FakeLLM()
        episode = core.Episode(None, client, 1)
        episode({"op": "llm", "prompt": "x", "max_tokens": 10 ** 9, "temperature": "0.5", "json_mode": 1})
        episode({"op": "llm", "prompt": [{"role": "user", "content": "x"}], "max_tokens": -5})
        self.assertEqual(client.calls[0][1], {"system": None, "json_mode": True, "max_tokens": 8192, "temperature": 0.5})
        self.assertEqual(client.calls[1][1]["max_tokens"], 1)
        for request in ({"op": "llm", "prompt": 3}, {"op": "llm", "prompt": "x", "max_tokens": "many"}, {"op": "llm", "prompt": "x", "max_tokens": float("inf")},
                        {"op": "llm", "prompt": "x", "temperature": [1]}, {"op": "other"}, "llm"):
            with self.subTest(request=request), self.assertRaises(CandidateError):
                episode(request)
        self.assertEqual(episode.llm_calls, 2)

    def test_an_exhausted_budget_stops_before_any_request_and_invalidates_the_evaluation(self):
        client = llm.LLM(model="google/gemini-2.5-flash", budget_usd=0.0, cache=MemoryCache(), api_key="not-a-key")
        source = ("def act(instruction, observation, valid_actions, history, llm):\n"
                  "    try:\n        return llm('guess')\n    except Exception:\n        return '1'\n")
        with mock.patch.object(llm.urllib.request, "urlopen", side_effect=AssertionError("no request may be sent")), \
                self.assertRaisesRegex(CandidateError, "budget of \\$0.00 exhausted"):
            self.toy(source, llm_client=client)
        self.assertEqual((client.requests, client.cost), (0, 0.0))

    def test_cached_responses_cost_nothing(self):
        key = llm.hashlib.sha256(json.dumps(["m", [{"role": "user", "content": "guess"}], 5, 0.0, False], sort_keys=True).encode()).hexdigest()
        client = llm.LLM(model="m", budget_usd=0.0, max_tokens=5, cache=MemoryCache({key: ("7", 0.0, 1, 1)}), api_key="not-a-key")
        source = "def act(instruction, observation, valid_actions, history, llm):\n    return llm('guess')\n"
        with mock.patch.object(llm.urllib.request, "urlopen", side_effect=AssertionError("no request may be sent")):
            out = self.toy(source, llm_client=client, targets=[7])
        self.assertEqual((out["score"], out["details"]["usage"]["cached_responses"], out["details"]["usage"]["cost_usd"]), (1.0, 1, 0.0))


class CalculatorTests(unittest.TestCase):
    Tool = collections.namedtuple("Tool", "name function")

    def task(self, **tools):
        class Task:
            name, resets = "fake/task", 0

            def reset(self):
                self.resets += 1
        task = Task()
        task.tools = {name: self.Tool(name, function) for name, function in tools.items()}
        return task

    def test_arithmetic_agrees_with_eval(self):
        for expression in ("2 + (3 * 4)", "7 / 2", "7 // 2", "-7 % 3", "2 ** 10", "2 ** 0.5", "2 ** -1", "1e3 * 1.5 - 0.25", "abs(-3.5)", "round(2.675)",
                           "int(7.9) + float(2)", " 1_000 * 3 ", "(100 + 20) * 1.05 / 3", "--5", "+4", "0.1 + 0.2", "1 ** 10 ** 9", "(-1) ** 99999",
                           "2 ** 4095", "-(2 ** 4095)", "3 ** 2000",
                           "0 ** 5000", "round(3.14159, 2)", "2 ** -100000", "7 % 0.5"):
            self.assertEqual(m3tool.calculator(expression), eval(expression), expression)  # noqa: S307 - fixed arithmetic
        self.assertEqual(m3tool.calculator("(-8) ** 0.5"), (-8) ** 0.5)

    def test_python_is_not_evaluated(self):
        marker = Path(tempfile.gettempdir()) / f"calculator-marker-{os.getpid()}"
        hostile = ["__import__('os').getpid()", f"open({str(marker)!r}, 'w').write('x')", "exec('x = 1')", "(lambda: 1)()", "[1][0]", "'a' * 3",
                   "1 if True else 2", "abs", "abs(x=1)", "max(1)", "().__class__", "1 < 2", "expression", "calculator('1')", "9 ** 9 ** 9", "1 / 0",
                   "2 ** 100000", "2 ** 4096", "(2 ** 4095) * 2", "int(1e999)", "(" * 300 + "1" + ")" * 300, "-" * 15000 + "1", "", "1 +", "import os", "lambda: 1"]
        for expression in hostile:
            self.assertEqual(m3tool.calculator(expression), f"Failed to evaluate expression: {expression}", expression[:40])
        self.assertFalse(marker.exists())
        for value in (5, 2.5, None, ["1"]):  # CodeAct converts a bare number before the call; eval then fails on it
            self.assertEqual(m3tool.calculator(value), f"Failed to evaluate expression: {value}")

    def test_harden_replaces_the_calculator_and_refuses_other_evaluating_tools(self):
        task = self.task(calculator=lambda expression: eval(expression), browse=lambda url: url)  # noqa: S307 - CodeAct's tool
        plain = task.tools["browse"].function
        m3tool.harden(task)
        self.assertIs(task.tools["calculator"].function, m3tool.calculator)
        self.assertIs(task.tools["browse"].function, plain)
        self.assertEqual(task.resets, 1)
        for function in (lambda text: eval(text), lambda text: exec(text), lambda text: (lambda: eval(text))(),  # noqa: S307, S102
                         lambda text: __import__(text), lambda text: compile(text, "x", "eval")):
            with self.assertRaisesRegex(RuntimeError, "evaluates text"):
                m3tool.harden(self.task(solver=function))
        self.assertFalse(m3tool.evaluates_text(lambda pattern: pattern.compile().eval()))  # attributes of these names are not calls of the builtins


@unittest.skipUnless(M3TOOL_DATA, "prepared data absent: run scripts/prepare_agent_bench.py m3tooleval")
class M3ToolEvalDataTests(Candidates):
    @classmethod
    def setUpClass(cls):
        cls.adapter = m3tool.Adapter(evaluator("agentsquare-m3tooleval")["P"])

    def test_no_tool_evaluates_the_agents_text(self):
        tools = [tool for task in self.adapter.tasks.values() for tool in task.tools.values()]
        self.assertEqual([tool.name for tool in tools if m3tool.evaluates_text(tool.function)], [])
        calculators = [tool for tool in tools if tool.name == "calculator"]
        self.assertGreater(len(calculators), 40)
        self.assertTrue(all(tool.function is m3tool.calculator for tool in calculators))

    def test_the_expected_output_cannot_be_read_through_the_calculator(self):
        task = self.adapter.tasks["trade_calculator/estimate_final_value"]
        self.adapter.reset({"id": task.name})
        reply = self.adapter.step("Action: calculator, [o.task.expected_output for o in __import__('gc').get_objects() "
                                  "if type(o).__name__ == 'Adapter'][0] End Action")
        self.assertTrue(reply["observation"].startswith("Failed to evaluate expression"))
        self.assertNotIn(str(task.expected_output), reply["observation"])
        self.assertEqual(self.adapter.step("Action: calculator, 100 * 1.5 + 11.25 End Action")["observation"], "161.25")
        self.assertEqual(self.adapter.score()["score"], 0.0)

    def test_the_exploit_candidate_scores_nothing(self):
        # With CodeAct's eval this candidate won 12 of the 20 episodes without a single model call.
        P = evaluator("agentsquare-m3tooleval")["P"]
        with mock.patch("autoresearch_suite.llm_bench_core.task_llm", return_value=FakeLLM()):
            out = core.evaluate(sandboxed.source_program(self.candidate(EXPLOIT)), P, "m3tool")
        self.assertEqual((out["details"]["episodes"], out["score"], out["details"]["successes"]), (20, 0.0, 0))

    def test_answer_check_on_every_task(self):
        self.assertEqual(len(self.adapter.tasks), 82)
        for name, task in self.adapter.tasks.items():
            self.adapter.reset({"id": name})
            self.assertFalse(self.adapter.step("Answer: no such answer")["done"], name)
            self.assertTrue(self.adapter.step("Answer: " + str(task.expected_output))["done"], name)
            self.assertEqual(self.adapter.score()["score"], 1.0, name)

    def test_the_official_check_accepts_the_integer_part_of_a_decimal_answer(self):
        # CodeAct's check_answer converts the expected output with int() first; recorded so that a change of the pin is noticed.
        decimals = [task for task in self.adapter.tasks.values() if isinstance(task.expected_output, float) and task.expected_output % 1]
        self.assertEqual(len(decimals), 14)
        self.assertTrue(all(task.check_answer(str(int(task.expected_output))) for task in decimals))

    def test_the_tasks_load_without_tqdm(self):
        code = ("import sys\nsys.modules['tqdm'] = None\nfrom autoresearch_suite import agent_bench_m3tool as m\n"
                "print(len(m.Adapter({}).tasks))\n")
        env = {**os.environ, "PYTHONPATH": str(ROOT / "suite" / "_lib"), "PYTHONDONTWRITEBYTECODE": "1"}
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "82"), done.stderr[-500:])


@needs("pandas")
@unittest.skipUnless(TRAVEL_DATA, "prepared data absent: run scripts/prepare_agent_bench.py travel and scripts/prepare_llm_bench.py")
class TravelDataTests(unittest.TestCase):
    KEYS = ("current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation")

    @classmethod
    def setUpClass(cls):
        cls.adapter = travel.Adapter(evaluator("agentsquare-travelplanner")["P"])
        cls.episodes = cls.adapter.episodes()

    def finish(self, plan, episode=0):
        self.adapter.reset(self.episodes[episode])
        reply = self.adapter.step("Finish[" + (plan if isinstance(plan, str) else json.dumps(plan)) + "]")
        outcome = self.adapter.score()
        json.dumps(outcome, allow_nan=False)
        return reply, outcome

    def test_episodes_are_the_stated_hard_queries(self):
        rows = [self.adapter.rows[episode["query"]] for episode in self.episodes]
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(151 <= row["id"] <= 180 and row["level"] == "hard" and row["days"] in (5, 7) for row in rows))

    def test_micro_pass_rate_of_degenerate_plans(self):
        days = self.adapter.rows[self.episodes[0]["query"]]["days"]
        self.adapter.reset(self.episodes[0])
        self.assertEqual(self.adapter.score()["score"], 0.0)  # no plan submitted
        dashes = [{"days": day + 1, **{key: "-" for key in self.KEYS}} for day in range(days)]
        reply, outcome = self.finish(dashes)
        self.assertTrue(reply["done"])
        self.assertEqual((outcome["score"], outcome["details"]["commonsense_passed"], outcome["details"]["success"]), (5 / 8, 5, False))

    def test_hostile_plans_are_scored_without_a_crash(self):
        days = self.adapter.rows[self.episodes[0]["query"]]["days"]
        odd = {"days": True, "current_city": 5, "transportation": None, "breakfast": [1], "attraction": {"a": 1}, "lunch": 1e308, "dinner": "x" * 500}
        for plan in ("[]", "hello", "[1, 2, 3]", "[" * 3000 + "]" * 3000, '[{"days": NaN, "current_city": Infinity}]', [odd] * days,
                     [{"days": 1}] * days, [{"days": str(day)} for day in range(60)]):
            with self.subTest(plan=str(plan)[:40]):
                reply, outcome = self.finish(plan)
                self.assertTrue(reply["done"])
                self.assertTrue(0.0 <= outcome["score"] < 1.0)
                self.assertFalse(outcome["details"]["success"])

    def test_tools_reproduce_the_reference_information_of_the_sampled_queries(self):
        patterns = (("Attractions in (.+)", "AttractionSearch", "{0}"), ("Restaurants in (.+)", "RestaurantSearch", "{0}"),
                    ("Accommodations in (.+)", "AccommodationSearch", "{0}"), ("Flight from (.+) to (.+) on (.+)", "FlightSearch", "{0}, {1}, {2}"),
                    ("Self-driving from (.+) to (.+)", "GoogleDistanceMatrix", "{0}, {1}, self-driving"),
                    ("Taxi from (.+) to (.+)", "GoogleDistanceMatrix", "{0}, {1}, taxi"))
        checked = 0
        for episode in self.episodes:
            for unit in self.adapter.rows[episode["query"]]["reference_information"]:
                kind, argument = next((tool, form.format(*found.groups())) for pattern, tool, form in patterns
                                      if (found := travel.re.fullmatch(pattern, unit["Description"])))
                self.assertEqual(" ".join(self.adapter.tool(kind, argument).split()), " ".join(unit["Content"].split()), unit["Description"])
                checked += 1
        self.assertGreater(checked, 100)


class StatementTests(unittest.TestCase):
    def test_balrog_episode_lists_are_the_stated_ones(self):
        for task_id, count in BALROG_EPISODES.items():
            parameters = evaluator(task_id)["P"]
            episodes = balrog.Adapter(parameters).episodes()
            self.assertEqual(len(episodes), count, task_id)
            self.assertEqual(len({episode["seed"] for episode in episodes}), count, task_id)
            self.assertEqual(episodes, balrog.Adapter(parameters).episodes(), task_id)
            for episode in episodes:
                index = int(episode["id"].rsplit("#", 1)[1])
                self.assertEqual(episode["seed"], balrog.hsi_seed(parameters["base_seed"], f"{parameters['suite']}/{episode['task']}", index))
        self.assertEqual({task_id: len(balrog.hsi_split(balrog.TASKS[evaluator(task_id)["P"]["suite"]])[1])
                          for task_id in ("hsi-babaisai-goto", "hsi-babaisai-breakstop", "hsi-babaisai-make")},
                         {"hsi-babaisai-goto": 3, "hsi-babaisai-breakstop": 2, "hsi-babaisai-make": 4})

    def test_lock_and_script_name_the_task_cache(self):
        import prepare_agent_bench as prepare
        lock = load("catalog/agent_bench-data-lock.json")
        self.assertIn("task cache", lock["raw_cache"])
        self.assertIn("task cache", lock["data_directory"])
        self.assertTrue(all(not row["path"].startswith(("/", ".")) and ".." not in row["path"] for row in lock["files"]))
        self.assertEqual({source["component"] for source in lock["sources"]}, {row["component"] for row in lock["files"]})
        for path in (prepare.RAW, prepare.DATA, prepare.ENV, core.env_python(), core.data_dir()):
            self.assertFalse(Path(path).resolve().is_relative_to(ROOT), path)
        self.assertEqual(prepare.DATA, core.data_dir())


if __name__ == "__main__":
    unittest.main()
