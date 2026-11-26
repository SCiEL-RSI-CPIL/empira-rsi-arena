"""ScienceWorld adapter with AgentBoard's subgoal progress rate (the protocol AgentSquare runs).

Runs in the pinned runtime (``agent_bench-env`` of the task cache) with ``scienceworld`` 1.2.1
(allenai/ScienceWorld, Apache-2.0; bundled simulator JAR, needs Java), the release
current during AgentSquare's experiments (July to October 2024).
Port of AgentBoard's ``environment/scienceworld_env.py`` and the episode setup
of ``tasks/scienceworld.py`` (hkust-nlp/AgentBoard, code Apache-2.0): the task
variation is loaded with AgentBoard's simplifications, the first observation is
the room description plus the inventory, the goal shown to the agent is
AgentBoard's rewritten goal, ``check valid actions`` lists the action templates,
and after every step each still-open subgoal pattern of the label file
(AgentBoard data, GPL-2.0, fetched by the prepare script) is searched in the
observation with ``re.search``. Progress rate = matched subgoals / subgoals;
the episode succeeds (and ends) when every subgoal has been matched. The
instruction text is AgentBoard's command overview (prompts/VanillaAgent/
scienceworld_base.json); AgentBoard's few-shot example is not part of the
environment and is left to the agent.

Reproducibility: the simulator iterates hash-based collections keyed by Java
identity hash codes, so replaying the same actions gave differently ordered
room listings and slightly different temperatures between runs. The simulator's
JVM is therefore started with a constant identity hash
(``-XX:+UnlockExperimentalVMOptions -XX:hashCode=2``), which makes those
iterations follow insertion order; the observations stay the simulator's own text.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from arena_task_paths import data_root

SIMPLIFICATIONS = "selfWateringFlowerPots,openContainers,openDoors,noElectricalAction"
CHECK = "check valid actions"
REPRODUCIBLE_JVM = "-XX:+UnlockExperimentalVMOptions -XX:hashCode=2"


def reproducible_simulator():
    """Constant identity hash codes for the simulator's JVM (inherited by the java process py4j launches)."""
    current = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if "-XX:hashCode=2" not in current:
        os.environ["JAVA_TOOL_OPTIONS"] = (current + " " + REPRODUCIBLE_JVM).strip()


def data_dir():
    return data_root() / "agent_bench"


def load_labels():
    path = data_dir() / "agentboard" / "scienceworld-test.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run python scripts/prepare_agent_bench.py agentboard")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def instruction_text():
    prompt = json.loads((data_dir() / "agentboard" / "prompts" / "scienceworld_base.json").read_text(encoding="utf-8"))
    return prompt["instruction"]


def matched_subgoals(observation, patterns, finished):
    """AgentBoard's _check_temperature_string: a subgoal is reached once its pattern occurs in an observation."""
    return [1.0 if done or re.search(pattern, observation) else 0.0 for pattern, done in zip(patterns, finished)]


class Adapter:
    def __init__(self, params):
        from scienceworld import ScienceWorldEnv
        reproducible_simulator()
        self.params = params
        self.labels = load_labels()
        self.instruction = instruction_text()
        self.env = ScienceWorldEnv("", None, envStepLimit=int(params.get("env_step_limit", 30)))
        self.finished, self.label = [], None

    def episodes(self):
        seed = self.params.get("seed", 0)
        rows = sorted(self.labels, key=lambda row: hashlib.sha256(f"{seed}|{row['id']}".encode()).hexdigest())
        return [{"id": f"scienceworld-{row['id']}", "index": row["id"]} for row in rows[:int(self.params.get("samples", 10))]]

    def reset(self, spec):
        label = next(row for row in self.labels if row["id"] == spec["index"])
        info = label["additional_info"]
        self.env.load(info["env_name"], info["var"], simplificationStr=SIMPLIFICATIONS)
        observation, _ = self.env.reset()
        self.label, self.finished = label, [0.0] * len(label["subgoals"])
        goal = label["goal"]
        instruction = (self.instruction + "\n\nYou should perform actions to accomplish the goal: " + goal +
                       "\nYou should use the following command for help when your action cannot be understood: " + CHECK)
        return {"instruction": instruction, "observation": observation + "\n" + self.env.inventory(), "valid_actions": [], "done": False}

    def templates(self):
        actions = [a for a in self.env.getPossibleActions() if "reset" not in a]
        return actions + ([CHECK] if CHECK not in actions else [])

    def step(self, action):
        action = action.strip()
        if action == CHECK:
            observation = "Choose an action from these valid actions: " + ", ".join(self.templates())
            return {"observation": observation, "reward": self.progress(), "done": False, "valid_actions": []}
        observation, _, _, _ = self.env.step(action)
        self.finished = matched_subgoals(observation, self.label["subgoals"], self.finished)
        done = sum(self.finished) >= len(self.finished)
        return {"observation": observation, "reward": self.progress(), "done": done, "valid_actions": []}

    def progress(self):
        return sum(self.finished) / len(self.finished) if self.finished else 0.0

    def score(self):
        details = {"success": bool(self.finished) and sum(self.finished) >= len(self.finished), "task": self.label["additional_info"]["env_name"],
                   "difficulty": self.label.get("difficulty")}
        return {"score": self.progress(), "details": details}

    def close(self):
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
