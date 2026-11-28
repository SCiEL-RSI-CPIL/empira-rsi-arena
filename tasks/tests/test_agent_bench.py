"""Offline tests of the agent_bench package (episode runner, sandboxing, adapters' pure helpers).

The episode-runner tests use the toy adapter and a fake model client, so they
need neither the network nor the pinned environment runtime. Tests that need
prepared data or the pinned runtime are skipped when those are missing.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import agent_bench_alfworld as alfworld  # noqa: E402
from autoresearch_suite import agent_bench_balrog as balrog  # noqa: E402
from autoresearch_suite import agent_bench_core as core  # noqa: E402
from autoresearch_suite import agent_bench_m3tool as m3tool  # noqa: E402
from autoresearch_suite import agent_bench_sciworld as sciworld  # noqa: E402
from autoresearch_suite import agent_bench_travel as travel  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import load, seed_source  # noqa: E402

# One package per family; the source repository's test read the same seeds from its generator module.
FAMILY_PACKAGES = {"agent_bench_balrog": "hsi-babyai", "agent_bench_textenv": "agentsquare-alfworld",
                   "agent_bench_m3tool": "agentsquare-m3tooleval", "agent_bench_travel": "agentsquare-travelplanner"}


class FakeLLM:
    def __init__(self, reply="5", fail=None):
        self.reply, self.fail, self.calls = reply, fail, []

    def __call__(self, prompt, **options):
        self.calls.append((prompt, options))
        if self.fail:
            raise CandidateError(self.fail)
        return self.reply

    def usage(self):
        return {"requests": len(self.calls)}


def candidate(tmp, source):
    path = Path(tmp) / "candidate.py"
    path.write_text(source)
    module = types.ModuleType("candidate")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


BISECT = '''
def act(instruction, observation, valid_actions, history, llm):
    low, high = 0, 9
    for action, seen, _ in history:
        if seen == "higher":
            low = int(action) + 1
        elif seen == "lower":
            high = int(action) - 1
    return str((low + high) // 2)
'''


class EpisodeRunnerTests(unittest.TestCase):
    def run_toy(self, source, llm=None, **params):
        P = {"targets": [3, 7, 9], "max_steps": 10, **params}
        with tempfile.TemporaryDirectory() as tmp, mock.patch("autoresearch_suite.llm_bench_core.task_llm", return_value=llm or FakeLLM()):
            return core.evaluate(candidate(tmp, source), P, "toy")

    def test_bisection_solves_every_episode(self):
        out = self.run_toy(BISECT)
        self.assertEqual(out["score"], 1.0)
        self.assertEqual([e["ended"] for e in out["details"]["per_episode"]], ["done"] * 3)

    def test_step_cap_ends_episodes(self):
        out = self.run_toy("def act(instruction, observation, valid_actions, history, llm):\n    return '0'\n", max_steps=3)
        self.assertEqual(out["score"], 0.0)
        self.assertEqual({e["steps"] for e in out["details"]["per_episode"]}, {3})
        self.assertEqual({e["ended"] for e in out["details"]["per_episode"]}, {"step_cap"})

    def test_model_calls_are_forwarded(self):
        llm = FakeLLM(reply="4")
        source = "def act(instruction, observation, valid_actions, history, llm):\n    return llm(instruction, max_tokens=5)\n"
        out = self.run_toy(source, llm=llm, max_steps=2, targets=[4])
        self.assertEqual(out["score"], 1.0)
        self.assertEqual(llm.calls[0][1]["max_tokens"], 5)

    def test_non_string_action_is_a_contract_violation(self):
        with self.assertRaises(CandidateError):
            self.run_toy("def act(instruction, observation, valid_actions, history, llm):\n    return 3\n")

    def test_missing_or_wrong_signature_is_rejected(self):
        with self.assertRaises(CandidateError):
            self.run_toy("def act(observation):\n    return '1'\n")
        with tempfile.TemporaryDirectory() as tmp:
            module = candidate(tmp, "x = 1\n")
            with self.assertRaises(CandidateError):
                core.check_candidate(module)

    def test_crash_keeps_the_environment_score(self):
        source = ("def act(instruction, observation, valid_actions, history, llm):\n"
                  "    if history:\n        raise RuntimeError('boom')\n    return '2'\n")
        out = self.run_toy(source, targets=[3])
        episode = out["details"]["per_episode"][0]
        self.assertEqual((episode["ended"], episode["score"]), ("crash", 0.5))
        self.assertIn("boom", episode["crash"])

    def test_model_budget_errors_invalidate_the_evaluation(self):
        source = ("def act(instruction, observation, valid_actions, history, llm):\n"
                  "    try:\n        return llm('x')\n    except Exception:\n        return '1'\n")
        with self.assertRaises(CandidateError):
            self.run_toy(source, llm=FakeLLM(fail="per-evaluation model budget of $0.01 exhausted"))

    def test_candidate_cannot_read_files(self):
        source = ("def act(instruction, observation, valid_actions, history, llm):\n"
                  f"    open({str(ROOT / 'run.py')!r}).read()\n    return '1'\n")
        out = self.run_toy(source, targets=[3])
        self.assertIn("file access denied", out["details"]["per_episode"][0]["crash"])

    def test_episode_override(self):
        with mock.patch.dict("os.environ", {"ARTASK_AGENT_EPISODES": "1"}):
            out = self.run_toy(BISECT)
        self.assertEqual(out["details"]["episodes"], 1)


class BalrogHelperTests(unittest.TestCase):
    def test_hsi_test_splits(self):
        self.assertEqual(balrog.hsi_split(balrog.TASKS["babaisai_breakstop"])[1],
                         ["env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule", "env/two_room-break_stop-goto_win-distr_obj"])
        self.assertEqual(len(balrog.hsi_split(balrog.TASKS["babaisai_goto"])[1]), 3)
        self.assertEqual(len(balrog.hsi_split(balrog.TASKS["babaisai_make"])[1]), 4)

    def test_seed_formula(self):
        self.assertEqual(balrog.hsi_seed(42, "babyai/BabyAI-MixedTrainLocal-v0/goto", 0), 1254509305)

    def test_action_loop_guard(self):
        self.assertTrue(balrog.is_action_loop(["up", "down"] * 6))
        self.assertTrue(balrog.is_action_loop(["a", "b", "c"] * 4))
        self.assertFalse(balrog.is_action_loop(["right"] * 12))
        self.assertFalse(balrog.is_action_loop(["up", "down"] * 5))

    def test_reply_cleaning(self):
        self.assertEqual(balrog.clean(" go forward. "), "go forward")
        self.assertEqual(balrog.clean("Move West!"), "Move West")


class AdapterHelperTests(unittest.TestCase):
    def test_alfworld_reply_cleaning(self):
        self.assertEqual(alfworld.clean_reply("> go to desk 1."), ("go to desk 1.", "go to desk 1"))
        self.assertEqual(alfworld.clean_reply("OK. take pen 1 from desk 1, end")[1], "take pen 1 from desk 1")

    def test_scienceworld_subgoals_accumulate(self):
        patterns = ["You move to the outside.", "You focus on the (.*) egg."]
        state = sciworld.matched_subgoals("You move to the outside.", patterns, [0.0, 0.0])
        self.assertEqual(state, [1.0, 0.0])
        self.assertEqual(sciworld.matched_subgoals("nothing", patterns, state), [1.0, 0.0])
        self.assertEqual(sciworld.matched_subgoals("You focus on the dove egg.", patterns, state), [1.0, 1.0])

    def test_travel_action_parser(self):
        self.assertEqual(travel.parse_action("FlightSearch[Atlanta, Dallas, 2022-03-15]"), ("FlightSearch", "Atlanta, Dallas, 2022-03-15"))
        self.assertEqual(travel.parse_action('Finish[[{"days": 1}]]\n')[0], "Finish")
        self.assertEqual(travel.parse_action("Thought: search"), (None, None))

    def test_m3tool_postprocess_keeps_first_block(self):
        text = "Thought: x Action: f, 1 End Action Answer: 3"
        self.assertEqual(m3tool.postprocess(text), "Thought: x Action: f, 1 End Action")


@unittest.skipUnless((m3tool.data_dir() / "m3tooleval" / "tasks" / "base.py").is_file(),
                     "prepared data absent: run scripts/prepare_agent_bench.py m3tooleval")
class M3ToolEvalTests(unittest.TestCase):
    def test_episode_list_and_answer_check(self):
        adapter = m3tool.Adapter({"per_domain": 4})
        episodes = adapter.episodes()
        self.assertEqual(len(adapter.tasks), 82)
        self.assertEqual(len(episodes), 20)
        task = adapter.tasks["message_decoder/multi_step_decoding_challenge"]
        start = adapter.reset({"id": task.name})
        self.assertIn("Instruction:", start["instruction"])
        wrong = adapter.step("Answer: nonsense")
        self.assertFalse(wrong["done"])
        right = adapter.step("Answer: " + str(task.expected_output))
        self.assertTrue(right["done"])
        self.assertEqual(adapter.score()["score"], 1.0)


PINNED = core.env_python().is_file()


@unittest.skipUnless(PINNED, "pinned environment runtime absent: run scripts/prepare_agent_bench.py --setup-environment")
class PinnedRuntimeTests(unittest.TestCase):
    """Environment servers in the pinned runtime (no model calls)."""

    def play(self, adapter, params, actions, episode=0):
        server = core.EnvServer(adapter, params, interpreter=core.env_python())
        try:
            spec = server.call("episodes")["episodes"][episode]
            observations = [server.call("reset", episode=spec)["observation"]]
            for action in actions:
                reply = server.call("step", action=action)
                observations.append(reply["observation"])
                if reply["done"]:
                    break
            return observations, server.call("score")
        finally:
            server.close()

    def test_crafter_episodes_are_reproducible(self):
        moves = ["Move West", "Do", "Move North", "Do", "Move East", "Noop", "Move South", "Do"] * 8
        params = {"suite": "crafter", "split": "all", "episodes_per_task": 1, "base_seed": 42}
        first, second = self.play("balrog", params, moves), self.play("balrog", params, moves)
        self.assertEqual(first, second)

    def test_babyai_invalid_action_feedback(self):
        params = {"suite": "babyai", "split": "all", "episodes_per_task": 1, "base_seed": 42}
        observations, outcome = self.play("balrog", params, ["dance"])
        self.assertIn("Defaulted to action: go forward", observations[1])
        self.assertEqual(outcome["details"]["invalid_actions"], 1)

    @unittest.skipUnless((sciworld.data_dir() / "agentboard" / "scienceworld-test.jsonl").is_file(),
                         "prepared data absent: run scripts/prepare_agent_bench.py agentboard")
    def test_scienceworld_replays_are_identical(self):
        # Label 21 (measure-melting-point, variation 7) listed the desk's tin cups in varying order before the JVM setting.
        params = {"samples": 90, "seed": 0}
        server = core.EnvServer("sciworld", params, interpreter=core.env_python())
        try:
            spec = next(e for e in server.call("episodes")["episodes"] if e["index"] == 21)
            runs = []
            for _ in range(2):
                trace = [server.call("reset", episode=spec)["observation"]]
                trace += [server.call("step", action=a)["observation"] for a in ("look around", "go to kitchen", "look around")]
                runs.append(trace)
        finally:
            server.close()
        self.assertEqual(runs[0], runs[1])

    @unittest.skipUnless((alfworld.data_dir() / "alfworld" / "games.json").is_file(),
                         "prepared data absent: run scripts/prepare_agent_bench.py alfworld")
    def test_alfworld_walkthrough_wins(self):
        games = json.loads((alfworld.data_dir() / "alfworld" / "games.json").read_text())
        self.assertEqual(len(games), 134)
        params = {"per_type": 2, "seed": 0}
        server = core.EnvServer("alfworld", params, interpreter=core.env_python())
        try:
            spec = server.call("episodes")["episodes"][0]
        finally:
            server.close()
        game = json.loads((alfworld.data_dir() / "alfworld" / "valid_unseen" / spec["game"]).read_text())
        _, outcome = self.play("alfworld", params, game["walkthrough"])
        self.assertEqual(outcome["score"], 1.0)


class PrepareTests(unittest.TestCase):
    def test_extract_and_digest(self):
        import prepare_agent_bench as prepare
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("repo-1/a/x.txt", "x")
            archive.writestr("repo-1/a/sub/y.txt", "y")
            archive.writestr("repo-1/b/z.txt", "z")
        files = prepare.extract(buffer.getvalue(), {"repo-1/a/": "out/a/"})
        self.assertEqual(sorted(files), ["out/a/sub/y.txt", "out/a/x.txt"])
        self.assertEqual(prepare.tree_digest(files), prepare.tree_digest(dict(reversed(list(files.items())))))

    def test_travel_flight_subset(self):
        import prepare_agent_bench as prepare
        header = "Flight Number,Price,DepTime,ArrTime,ActualElapsedTime,FlightDate,OriginCityName,DestCityName,Distance\n"
        flights = header + "F1,10,1:00,2:00,1h,2022-03-01,A,B,1\nF2,10,1:00,2:00,1h,2022-03-05,A,B,1\nF3,10,1:00,2:00,1h,2022-03-01,A,Z,1\n"
        validation = ("org,dest,days,visiting_city_number,date,people_number,local_constraint,budget,query,level,reference_information\n"
                      "A,B,3,1,\"['2022-03-01', '2022-03-02', '2022-03-03']\",1,\"{}\",100,q,easy,[]\n")
        out = prepare.derive_travel_flights({"travelplanner-flights": flights.encode(), "travelplanner-validation": validation.encode(),
                                             "travelplanner-city-states": b"A\tSA\nB\tSB\nZ\tSZ\n"}).decode()
        self.assertEqual(out.splitlines()[1:], ["F1,10,1:00,2:00,1h,2022-03-01,A,B,1"])


class FamilyTests(unittest.TestCase):
    def test_seeds_define_act(self):
        for family, task_id in FAMILY_PACKAGES.items():
            self.assertEqual(load(f"suite/{task_id}/problem.json")["family"], family)
            namespace = {}
            exec(compile(seed_source(task_id), "<seed>", "exec"), namespace)
            self.assertTrue(callable(namespace["act"]))
        namespace = {}
        exec(compile(seed_source(FAMILY_PACKAGES["agent_bench_balrog"]), "<seed>", "exec"), namespace)
        self.assertEqual(namespace["_pick"]("Action: go forward.", ["turn left", "go forward"]), "go forward")
        labels = load(f"suite/{FAMILY_PACKAGES['agent_bench_balrog']}/problem.json")["labels"]
        self.assertEqual(labels["scientific_contribution"], "benchmark")


if __name__ == "__main__":
    unittest.main()
