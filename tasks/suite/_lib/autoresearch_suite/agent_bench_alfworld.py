"""ALFWorld (text) adapter following AgentSquare's ``tasks/alfworld/alfworld_run.py`` protocol.

Runs in the pinned runtime (``agent_bench-env`` of the task cache) with ``alfworld`` 0.4.2 and
``textworld[pddl]`` 1.7.0 (both MIT). The games are ALFWorld's 134
``eval_out_of_distribution`` games (valid_unseen), selected by the prepare script
with ``AlfredTWEnv.collect_game_files``'s rules, and each is played through
TextWorld with ALFWorld's own wrappers (``AlfredDemangler`` without domain
randomisation, ``AlfredInfos``). As in AgentSquare's runner, the welcome banner
is removed from the first observation, ``>`` and ``OK`` are stripped from the
agent's reply, ``think:`` replies leave the game unchanged and are answered
with ``OK.``, a reply is cut at ``, end`` and its periods are removed, and the
agent is not shown the admissible commands. The instruction lists ALFWorld's
command templates (AgentSquare conveys them through its few-shot ReAct
examples, which belong to the agent) followed by the first observation (room
and task), which AgentSquare keeps at the head of every prompt. Episode score: 1
when the game reports ``won`` within the step cap, else 0.
"""
from __future__ import annotations

import hashlib
import json

from arena_task_paths import data_root

INSTRUCTION = """You are an agent in a simulated household, interacting through text. Complete the task stated in the observation. Each reply is one command; the household understands these commands, where X is an object and Y a receptacle or appliance, both written with their number as in the observations (for example "apple 1", "countertop 2"):
go to Y | open Y | close Y | take X from Y | put X in/on Y | use Y (for example a desklamp) | heat X with Y | cool X with Y | clean X with Y | slice X with Y | examine X | inventory | look
Note that "in/on" is written literally. A reply starting with "think:" is a private note: the household does not change and answers "OK."."""


def data_dir():
    return data_root() / "agent_bench"


def clean_reply(reply):
    """AgentSquare's action cleaning (ALFWORLD.step): strip '>' and 'OK', cut at ', end', drop periods."""
    action = reply.replace(">", "").replace("OK.", "").replace("OK", "").strip()
    return action, action.split(", end")[0].replace(".", "")


class Adapter:
    def __init__(self, params):
        path = data_dir() / "alfworld" / "games.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run python scripts/prepare_agent_bench.py alfworld")
        self.games = json.loads(path.read_text(encoding="utf-8"))
        self.params = params
        self.env = None

    def episodes(self):
        seed, per_type = self.params.get("seed", 0), int(self.params.get("per_type", 2))
        out = []
        for task_type in sorted({g["task_type"] for g in self.games}):
            rows = sorted((g for g in self.games if g["task_type"] == task_type),
                          key=lambda g: hashlib.sha256(f"{seed}|{g['game']}".encode()).hexdigest())
            out += [{"id": g["game"].rsplit("/", 1)[0], "game": g["game"], "task_type": task_type} for g in rows[:per_type]]
        return out

    def reset(self, spec):
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
        self.close()
        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        game = str(data_dir() / "alfworld" / "valid_unseen" / spec["game"])
        env_id = textworld.gym.register_game(game, infos, max_episode_steps=1000, asynchronous=False,
                                             wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        self.env = textworld.gym.make(env_id)
        observation, _ = self.env.reset()
        self.won, self.task_type = False, spec["task_type"]
        first = "\n".join(observation.split("\n\n")[1:])
        # AgentSquare keeps the first observation (room and task) at the head of every prompt; here it is part of the instruction.
        return {"instruction": INSTRUCTION + "\n\nThe episode began with this observation:\n" + first, "observation": first,
                "valid_actions": [], "done": False}

    def step(self, reply):
        action, command = clean_reply(reply)
        observation, _, done, info = self.env.step(command)
        if observation.startswith("You arrive at loc "):
            observation = observation[observation.find(". ") + 2:]
        if action.startswith("think:"):
            observation = "OK."
        self.won = self.won or bool(info["won"])
        return {"observation": observation, "reward": float(self.won), "done": bool(done) or self.won, "valid_actions": []}

    def score(self):
        return {"score": float(self.won), "details": {"success": self.won, "task_type": self.task_type}}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
