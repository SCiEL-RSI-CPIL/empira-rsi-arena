"""BALROG environments as played in HSI (BabyAI, Crafter, BabaIsAI suites, MiniHack).

Runs inside the pinned runtime (``agent_bench-env`` of the task cache; Python 3.11, gym 0.23)
where ``scripts/prepare_agent_bench.py --setup-environment`` installs BALROG
(balrog-ai/BALROG, MIT) and the environment packages it pins: the Minigrid fork
that provides BabyAI-MixedTrainLocal and its text descriptions, baba-is-ai,
crafter and, for MiniHack, balrog-nle and the balrog-ai minihack fork. The
environments, language wrappers, instruction prompts, invalid-action handling
(default action plus BALROG's feedback text) and progression statistics are
BALROG's own code. Following HSI's evaluator, the agent's reply is reduced to
letters, spaces and colons before the validity check, the episode stops early
when the last 12 valid actions form a strict period-2 or period-3 loop, and
episode seeds are HSI's deterministic test seeds
``int(sha256(f"{base_seed}_{suite}/{task}_{episode}_0")[:8], 16)``. Task lists
and the BabaIsAI test splits are HSI's (``src/benchmark/balrog/tasks.py``,
``dev_ratio`` 0.8, ``split_seed`` 66). Episode score: BALROG progression
(BabyAI, BabaIsAI, MiniHack: 1 when solved; Crafter: unlocked achievements / 22).
Crafter's per-chunk object sets are made insertion-ordered so that a seeded
episode is reproducible (see ``make_crafter_reproducible``).
"""
from __future__ import annotations

import copy
import hashlib
import random
import re

from arena_task_paths import data_root

BABAISAI = {
    "babaisai_goto": ["goto_win", "goto_win-distr_obj", "goto_win-distr_rule", "goto_win-distr_obj_rule", "goto_win-distr_obj-irrelevant_rule",
                      "two_room-goto_win", "two_room-goto_win-distr_obj", "two_room-goto_win-distr_rule", "two_room-goto_win-distr_obj_rule",
                      "two_room-goto_win-distr_obj-irrelevant_rule", "two_room-goto_win-distr_win_rule"],
    "babaisai_breakstop": ["two_room-break_stop-goto_win", "two_room-break_stop-goto_win-distr_obj", "two_room-break_stop-goto_win-distr_rule",
                           "two_room-break_stop-goto_win-distr_obj_rule", "two_room-break_stop-goto_win-distr_obj-irrelevant_rule",
                           "two_room-maybe_break_stop-goto_win", "two_room-maybe_break_stop-goto_win-distr_obj",
                           "two_room-maybe_break_stop-goto_win-distr_rule", "two_room-maybe_break_stop-goto_win-distr_obj_rule",
                           "two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule"],
    "babaisai_make": ["make_win", "make_win-distr_obj", "make_win-distr_rule", "make_win-distr_obj_rule", "make_win-distr_obj-irrelevant_rule",
                      "two_room-make_win", "two_room-make_win-distr_obj", "two_room-make_win-distr_rule", "two_room-make_win-distr_obj_rule",
                      "two_room-make_win-distr_obj-irrelevant_rule", "two_room-make_win-distr_win_rule", "two_room-break_stop-make_win",
                      "two_room-break_stop-make_win-distr_obj", "two_room-break_stop-make_win-distr_rule",
                      "two_room-break_stop-make_win-distr_obj_rule", "two_room-break_stop-make_win-distr_obj-irrelevant_rule"],
}
TASKS = {
    "babyai": [f"BabyAI-MixedTrainLocal-v0/{kind}" for kind in ("goto", "pickup", "open", "putnext", "pick_up_seq_go_to")],
    "crafter": ["default"],
    "minihack": ["MiniHack-Boxoban-Hard-v0", "MiniHack-Boxoban-Medium-v0", "MiniHack-MazeWalk-9x9-v0", "MiniHack-MazeWalk-15x15-v0",
                 "MiniHack-Corridor-R3-v0", "MiniHack-CorridorBattle-Dark-v0", "MiniHack-Quest-Easy-v0", "MiniHack-Quest-Medium-v0"],
    **{suite: ["env/" + name for name in names] for suite, names in BABAISAI.items()},
}
# BALROG's config.yaml environment settings with HSI's override (Crafter precise_location).
ENV_KWARGS = {
    "babyai_kwargs": {"num_dists": 0},
    "crafter_kwargs": {"area": [64, 64], "view": [9, 9], "size": [256, 256], "reward": True, "seed": None, "max_episode_steps": 2000,
                       "unique_items": True, "precise_location": True, "skip_items": [], "edge_only_items": []},
    "babaisai_kwargs": {"add_ruleset": True},
    "minihack_kwargs": {"character": "@", "max_episode_steps": 100, "penalty_step": -0.01, "penalty_time": 0.0, "penalty_mode": "constant",
                        "savedir": None, "save_ttyrec_every": 0, "autopickup": False, "skip_more": True},
}
LOOP_WINDOW = 12


def data_dir():
    return data_root() / "agent_bench"


def use_prepared_boxoban_levels():
    """MiniHack reads Boxoban levels from its package directory; point it at the prepared copy instead (no download at run time)."""
    import minihack.envs.boxohack as boxohack
    path = data_dir() / "minihack" / "boxoban-levels-master"
    if not path.is_dir():
        raise FileNotFoundError(f"{path} missing; run python scripts/prepare_agent_bench.py minihack")
    boxohack.LEVELS_PATH = str(path)


class _OrderedSet(dict):
    """Insertion-ordered stand-in for the set of objects Crafter keeps per chunk (add, remove, iteration, len, in)."""

    def add(self, item):
        self[item] = None

    def remove(self, item):
        del self[item]


def make_crafter_reproducible():
    """Crafter iterates per-chunk *sets* of objects, whose order follows memory addresses, when it picks a creature to
    despawn; the same seed and actions can then give different episodes. Use insertion-ordered sets instead (the
    spawn and despawn probabilities are unchanged)."""
    import collections
    import crafter.engine as engine
    if getattr(engine.World, "_artask_ordered_chunks", False):
        return
    original = engine.World.reset

    def reset(self, seed=None):
        original(self, seed)
        self._chunks = collections.defaultdict(_OrderedSet)
    engine.World.reset = reset
    engine.World._artask_ordered_chunks = True


class Config(dict):
    """Attribute-access dictionary standing in for BALROG's OmegaConf config."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(name) from None
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value


def hsi_split(tasks, dev_ratio=0.8, split_seed=66):
    """HSI's dev/test split of one task category (BenchmarkEvaluatorAdapter._load_and_split_tasks)."""
    shuffled = list(tasks)
    random.Random(split_seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * dev_ratio))
    return shuffled[:cut], shuffled[cut:]


def hsi_seed(base_seed, task_id, episode, repeat=0):
    return int(hashlib.sha256(f"{base_seed}_{task_id}_{episode}_{repeat}".encode()).hexdigest()[:8], 16)


def is_action_loop(actions, window=LOOP_WINDOW):
    if len(actions) < window:
        return False
    recent = actions[-window:]
    if len(set(recent)) < 2:
        return False
    return any(all(recent[i] == recent[i % p] for i in range(window)) for p in (2, 3))


def clean(text):
    return re.sub(r"[^a-zA-Z\s:]", "", (text or "").strip())


class Adapter:
    def __init__(self, params):
        self.suite = params["suite"]
        if self.suite not in TASKS:
            raise ValueError(f"unknown BALROG suite {self.suite!r}")
        self.base_env = "babaisai" if self.suite.startswith("babaisai") else self.suite
        self.params = params
        self.env = None

    def episodes(self):
        tasks = TASKS[self.suite]
        if self.params.get("split", "all") == "test":
            tasks = hsi_split(tasks)[1]
        base = int(self.params.get("base_seed", 42))
        out = []
        for task in tasks:
            for episode in range(int(self.params.get("episodes_per_task", 1))):
                out.append({"id": f"{task}#{episode}", "task": task, "seed": hsi_seed(base, f"{self.suite}/{task}", episode)})
        return out

    def _text(self, obs):
        text = obs["text"]
        if isinstance(text, str):
            return text
        long_term, short_term = text.get("long_term_context", ""), text.get("short_term_context", "")
        return (short_term + "\n\n" + long_term) if short_term else long_term

    def reset(self, spec):
        from balrog.environments import make_env
        if self.base_env == "minihack":
            use_prepared_boxoban_levels()
        if self.base_env == "crafter":
            make_crafter_reproducible()
        if self.env is not None:
            self.env.close()
        config = Config({"envs": copy.deepcopy(ENV_KWARGS), "agent": {"max_image_history": 0}})
        config["envs"]["env_kwargs"] = {"seed": spec["seed"]}
        config["envs"]["crafter_kwargs"]["seed"] = spec["seed"]
        random.seed(spec["seed"])
        self.env = make_env(self.base_env, spec["task"], config)
        if self.base_env == "minihack":
            import numpy as np
            np.random.seed(spec["seed"])
        obs, _ = self.env.reset(seed=spec["seed"])
        mission = obs.get("mission") if self.base_env == "babyai" else None
        instruction = self.env.get_instruction_prompt(instructions=mission) if self.base_env == "babyai" else self.env.get_instruction_prompt()
        self.actions, self.invalid, self.loop_stop = [], 0, False
        self.valid = list(self.env.language_action_space)
        return {"instruction": instruction, "observation": self._text(obs), "valid_actions": self.valid, "done": False}

    def step(self, reply):
        candidate = clean(reply)
        action = self.env.check_action_validity(candidate)
        self.actions.append(action)
        if is_action_loop(self.actions):
            self.loop_stop = True
            return {"observation": "The episode was stopped: the recent actions repeat in a loop.", "reward": 0.0, "done": True,
                    "valid_actions": self.valid}
        obs, reward, terminated, truncated, _ = self.env.step(action)
        if action != candidate:
            self.invalid += 1
            obs["text"]["long_term_context"] = ("\n\nYour previous output did not contain a valid action. Defaulted to action: "
                                                f"{action}\n\nObservation:\n" + obs["text"]["long_term_context"])
        return {"observation": self._text(obs), "reward": float(reward), "done": bool(terminated or truncated), "valid_actions": self.valid}

    def score(self):
        stats = self.env.get_stats()
        progression = float(stats.get("progression", 0.0))
        details = {"success": progression >= 1.0, "invalid_actions": self.invalid, "loop_stop": self.loop_stop}
        if stats.get("achievements"):
            details["achievements"] = sorted(k for k, v in stats["achievements"].items() if v)
        return {"score": min(1.0, max(0.0, progression)), "details": details}

    def close(self):
        if self.env is not None:
            self.env.close()
