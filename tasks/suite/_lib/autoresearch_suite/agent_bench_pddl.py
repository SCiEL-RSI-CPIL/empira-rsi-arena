"""PDDL planning games of AgentBoard (gripper, blockworld, barman, tyreworld) with its progress rate.

AgentSquare evaluates PDDL through AgentBoard's environment but its repository
has no PDDL runner, so AgentBoard's own protocol is used. Runs in the pinned
runtime (``agent_bench-env`` of the task cache). The environment class is AgentBoard's
``agentboard/environment/pddl_env/pddl_env.py`` with its vendored pddlgym
(hkust-nlp/AgentBoard, code Apache-2.0; pddlgym MIT), fetched at a pinned commit
by ``scripts/prepare_agent_bench.py`` and imported unchanged except for three
shims: AgentBoard's registry and base class are replaced by trivial stand-ins,
``nltk.download`` is disabled (the punkt tokenizer is prepared offline) and
``pddlgym.make`` drops the ``disable_env_checker`` keyword that the pinned gym
0.23 does not accept. Episodes follow AgentBoard's ``tasks/pddl.py``: problem i
of each game in pddlgym's file order, the goal text and the domain instruction
of ``prompts/VanillaAgent/pddl_vanilla_prompt.json`` (without its few-shot
examples), ``check valid actions`` available as an action. Episode score:
AgentBoard's progress rate, the largest fraction of goal literals satisfied so
far; success when every goal literal holds.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types

from arena_task_paths import data_root

GAMES = (("gripper", 20), ("blockworld", 10), ("barman", 20), ("tyreworld", 10))
CHECK = "check valid actions"


def data_dir():
    return data_root() / "agent_bench"


def load_environment_module():
    root = data_dir() / "agentboard"
    source = root / "pddl_env" / "pddl_env.py"
    if not source.is_file():
        raise FileNotFoundError(f"{source} missing; run python scripts/prepare_agent_bench.py agentboard")
    base = types.ModuleType("environment.base_env")
    base.BaseEnvironment = object
    registry_module = types.ModuleType("common.registry")
    registry_module.registry = types.SimpleNamespace(register_environment=lambda name: (lambda cls: cls))
    sys.modules.update({"environment": types.ModuleType("environment"), "environment.base_env": base,
                        "common": types.ModuleType("common"), "common.registry": registry_module})
    import nltk
    nltk.data.path.insert(0, str(root / "nltk_data"))
    nltk.download = lambda *args, **kwargs: True
    sys.path.insert(0, str(root / "pddl_env"))
    import gym
    import pddlgym
    pddlgym.make = lambda name, **kwargs: gym.make(name)
    spec = importlib.util.spec_from_file_location("agentboard_pddl_env", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Adapter:
    def __init__(self, params):
        self.params = params
        self.module = load_environment_module()
        root = data_dir() / "agentboard"
        self.prompts = json.loads((root / "prompts" / "pddl_vanilla_prompt.json").read_text(encoding="utf-8"))
        labels = [json.loads(line) for line in (root / "pddl-test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.problems = []
        for game, count in GAMES:  # AgentBoard's get_all_environment_configs order
            for index in range(count):
                self.problems.append({"game": game, "index": index, "difficulty": labels[len(self.problems)]["difficulty"]})
        self.env = None

    def episodes(self):
        seed, per_game = self.params.get("seed", 0), int(self.params.get("per_game", 3))
        out = []
        for game, _ in GAMES:
            rows = [p for p in self.problems if p["game"] == game]
            rows.sort(key=lambda p: hashlib.sha256(f"{seed}|{p['game']}-{p['index']}".encode()).hexdigest())
            out += [{"id": f"{p['game']}-{p['index']}", **p} for p in rows[:per_game]]
        return out

    def reset(self, spec):
        self.env = self.module.PDDL(problem_index=spec["index"], game_name=spec["game"], difficulty=spec["difficulty"])
        goal = self.env._get_goal()
        instruction = (self.prompts[spec["game"]]["instruction"] + "\n\nYou should perform actions to accomplish the goal: " + goal +
                       "\nYou should use the following commands for help when your action cannot be understood: " + CHECK)
        return {"instruction": instruction, "observation": self.env._get_obs(), "valid_actions": [], "done": False}

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        return {"observation": state, "reward": float(reward), "done": bool(done), "valid_actions": []}

    def score(self):
        return {"score": float(self.env.reward), "details": {"success": bool(self.env.won), "game": self.env.game_name,
                                                             "difficulty": self.env.difficulty}}
