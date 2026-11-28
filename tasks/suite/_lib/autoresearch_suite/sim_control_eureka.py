"""Eureka-style reward design on CPU MuJoCo analogues of IsaacGymEnvs BallBalance and Quadcopter.

The candidate is a reward function, as in the suite's gym_reward_design family::

    reward(observation, action, next_observation, info) -> float

The evaluator trains a linear policy ``action = tanh(observation @ W + b)`` with
the cross-entropy method (CEM) under the candidate reward and reports the task's
own fitness, which Eureka uses to rank reward samples (IsaacGymEnvs
``consecutive_successes``): the mean episode length for BallBalance and minus
the mean distance to the target point (0, 0, 1) for Quadcopter.

Only BallBalance is released as a task (family isaac_ballbalance_reward_design).
The Quadcopter analogue failed the package's noise gate (Eureka's human reward
did not separate from a zero reward within a 900 s budget; see
validation/sim_control-checks.json, "withdraw") and is kept, with its tests,
for a future trainer only.

Isolation: the simulator, the CEM update and the fitness evaluation run in the
evaluator process. The candidate's reward runs in a resource-limited sandboxed
child (``software_sandbox``) that receives, generation by generation, only the
recorded training transitions (raw states and actions); it returns one return
(sum of rewards) per rollout. The child never holds the simulator, the
evaluation seeds or the fitness, so it cannot change the environment state or
the score; it can only rank the CEM population. The evaluator reads the
candidate's source only (``sandboxed.run``): no statement of the candidate runs
in the evaluator process.
"""
from __future__ import annotations

import base64
import math
import time
from pathlib import Path

import numpy as np

from . import sim_control_eureka_envs as envs
from . import sim_control_eureka_views as views
from . import software_sandbox
from .runtime import CandidateError

VIEWS_SOURCE = Path(__file__).resolve().with_name("sim_control_eureka_views.py")
DRIVER = "__sim_control_reward_driver__"
MAX_ABS_RETURN = 1e12

SHIM = '''


def __sim_control_reward_driver__(spec, oracle):
    import random as _random
    import numpy as _np
    _random.seed(0)
    _np.random.seed(0)
    function = globals().get("reward")
    if not callable(function):
        raise RewardContractError("the candidate must define reward(observation, action, next_observation, info)")
    generations = 0
    while True:
        batch = oracle({"op": "next"})
        if batch.get("done"):
            return generations
        oracle({"op": "returns", "values": rollout_returns(spec["task"], batch, function)})
        generations += 1
'''

DEFAULTS = {
    "ballbalance": {"iterations": 80, "population": 24, "elite": 4, "episodes": 4, "init_std": 0.1, "min_std": 0.02,
                    "train_seed": 0, "eval_seed": 2026, "eval_episodes": 50},
    "quadcopter": {"iterations": 60, "population": 32, "elite": 4, "episodes": 1, "init_std": 0.1, "min_std": 0.02,
                   "train_seed": 0, "eval_seed": 2026, "eval_episodes": 50},
}


def parameters(P):
    task = P["task"]
    if task not in envs.ENVS:
        raise ValueError(f"unknown task {task!r}")
    merged = {**DEFAULTS[task], **{k: v for k, v in P.items() if k != "task"}}
    merged["task"] = task
    return merged


def _encode(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f8").tobytes()).decode("ascii")


def split_theta(theta, env):
    W = theta[: env.obs_dim * env.act_dim].reshape(env.obs_dim, env.act_dim)
    return W, theta[env.obs_dim * env.act_dim:]


def rollout(env, theta, rng):
    """One episode of the linear tanh policy; returns raw states (T+1), actions (T) and whether it ended by failure."""
    W, b = split_theta(theta, env)
    state = env.reset(rng)
    states, actions = [state], []
    while True:
        action = np.tanh(env.observe(state) @ W + b)
        state, failed, done = env.step(action)
        states.append(state)
        actions.append(action)
        if done:
            return np.array(states), np.array(actions), failed


class CemOracle:
    """Trusted side of the sandboxed training loop: samples policies, simulates them, updates the CEM distribution."""

    def __init__(self, env, P):
        self.env, self.P = env, P
        dim = (env.obs_dim + 1) * env.act_dim
        self.mean = np.zeros(dim)
        self.std = np.full(dim, float(P["init_std"]))
        self.rng = np.random.default_rng([int(P["train_seed"]), 0])
        self.generation = 0
        self.pending = None
        self.finished = False
        self.history = []
        self.sim_seconds = 0.0

    def __call__(self, request):
        if not isinstance(request, dict):
            raise CandidateError("malformed request from the reward process")
        op = request.get("op")
        if op == "next":
            if self.pending is not None:
                raise CandidateError("the reward process must return the rollout returns before asking for more")
            if self.generation >= int(self.P["iterations"]):
                self.finished = True
                return {"done": True}
            return self._next()
        if op == "returns":
            return self._update(request.get("values"))
        raise CandidateError(f"unknown request {op!r}")

    def _next(self):
        started = time.process_time()
        thetas = self.mean + self.std * self.rng.standard_normal((int(self.P["population"]), self.mean.size))
        states, actions, lengths, ended, fitness = [], [], [], [], []
        for theta in thetas:  # common random numbers: every member meets the same start states in a generation
            for k in range(int(self.P["episodes"])):
                rng = np.random.default_rng([int(self.P["train_seed"]), 1, self.generation, k])
                s, a, failed = rollout(self.env, theta, rng)
                states.append(s)
                actions.append(a)
                lengths.append(len(a))
                ended.append(bool(failed))
                fitness.append(len(a) if self.env.name == "ballbalance" else -float(np.mean(np.linalg.norm(s[1:, 0:3] - envs.QUAD_TARGET, axis=1))))
        self.pending = (thetas, lengths, fitness)
        self.sim_seconds += time.process_time() - started
        return {"lengths": lengths, "terminated": ended, "states": _encode(np.concatenate(states)), "actions": _encode(np.concatenate(actions))}

    def _update(self, values):
        if self.pending is None:
            raise CandidateError("unexpected returns from the reward process")
        thetas, lengths, fitness = self.pending
        if not isinstance(values, list) or len(values) != len(lengths):
            raise CandidateError("the reward process returned a malformed list of returns")
        returns = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > MAX_ABS_RETURN:
                raise CandidateError(f"rollout returns must be finite with magnitude at most {MAX_ABS_RETURN:g}")
            returns.append(float(value))
        k = int(self.P["episodes"])
        member_returns = np.asarray(returns).reshape(len(thetas), k).mean(axis=1)
        member_fitness = np.asarray(fitness, dtype=float).reshape(len(thetas), k).mean(axis=1)
        order = np.argsort(-member_returns, kind="stable")[: int(self.P["elite"])]
        self.mean = thetas[order].mean(axis=0)
        self.std = thetas[order].std(axis=0) + float(self.P["min_std"])
        self.history.append({"generation": self.generation, "mean_return": round(float(np.mean(member_returns)), 6),
                             "elite_training_fitness": round(float(member_fitness[order].mean()), 4),
                             "mean_length": round(float(np.mean(lengths)), 2)})
        self.generation += 1
        self.pending = None
        return {"ok": True}


def evaluate_policy(env, theta, P):
    """Task fitness of the trained mean policy on held-out start states (no candidate code involved)."""
    W, b = split_theta(theta, env)
    values = []
    for episode in range(int(P["eval_episodes"])):
        rng = np.random.default_rng([int(P["eval_seed"]), 2, episode])
        state = env.reset(rng)
        if env.name == "ballbalance":  # Isaac's consecutive_successes: mean episode length (steps until the ball drops, cap 500)
            while True:
                state, _, done = env.step(np.tanh(env.observe(state) @ W + b))
                if done:
                    values.append(float(env.progress))
                    break
        else:  # consecutive_successes = -mean target distance over all environment steps, with resets after a crash
            distances = []
            for _ in range(env.horizon):
                state, died, _ = env.step(np.tanh(env.observe(state) @ W + b))
                distances.append(float(np.linalg.norm(state[0:3] - envs.QUAD_TARGET)))
                if died:
                    state = env.reset(rng)
            values.append(-float(np.mean(distances)))
    return float(np.mean(values)), values


def train(source, P, *, cpu_seconds=600, wall_seconds=900, memory_mb=2048):
    """Train the CEM policy with the reward defined in ``source``; returns (theta, oracle)."""
    env = envs.ENVS[P["task"]]()
    oracle = CemOracle(env, P)
    # The views follow the candidate source so that a candidate may start with __future__ imports.
    program = source + "\n\n" + VIEWS_SOURCE.read_text(encoding="utf-8") + SHIM
    try:
        generations = software_sandbox.execute(program, DRIVER, [{"task": P["task"]}], oracle=oracle, cpu_seconds=cpu_seconds,
                                               wall_seconds=wall_seconds, memory_mb=memory_mb, filename="candidate.py")
    except software_sandbox.SandboxFailure as failure:
        text = str(failure)
        contract = "RewardContractError" in text
        if contract:
            text = text.split("RewardContractError:", 1)[-1].strip()
        last = failure.traceback_text.strip().splitlines()[-1] if failure.traceback_text.strip() else ""
        detail = f" ({last})" if failure.kind == "exception" and not contract and last and last not in text else ""
        raise CandidateError(f"reward evaluation failed ({failure.kind}): {text}{detail}"[:1500]) from None
    if not oracle.finished or generations != int(P["iterations"]):
        raise CandidateError("the reward process stopped before training finished")
    return oracle.mean.copy(), oracle


def train_local(function, P):
    """In-process twin of ``train`` for trusted reward functions (tests, noise gate); same oracle and returns."""
    env = envs.ENVS[P["task"]]()
    oracle = CemOracle(env, P)
    while True:
        batch = oracle({"op": "next"})
        if batch.get("done"):
            return oracle.mean.copy(), oracle
        oracle({"op": "returns", "values": views.rollout_returns(P["task"], batch, function)})


def check_source(module):
    path = software_sandbox.require_entry(module, "reward")
    source = Path(path).read_text(encoding="utf-8")
    if DRIVER in source:
        raise CandidateError(f"the name {DRIVER} is reserved")
    return source


def score_source(source, P, **limits):
    """(fitness, details) of a reward given as source code."""
    P = parameters(P)
    started = time.time()
    theta, oracle = train(source, P, **limits)
    value, episodes = evaluate_policy(envs.ENVS[P["task"]](), theta, P)
    details = {"task": P["task"], "fitness": "mean episode length (steps, cap 500)" if P["task"] == "ballbalance" else "minus mean distance to (0, 0, 1) in metres",
               "eval_episodes": len(episodes), "episode_fitness_std": round(float(np.std(episodes)), 4),
               "training": {"iterations": int(P["iterations"]), "population": int(P["population"]), "elite": int(P["elite"]),
                            "train_seed": int(P["train_seed"]), "last_generation": oracle.history[-1] if oracle.history else None},
               "simulation_cpu_seconds": round(oracle.sim_seconds, 2), "elapsed_seconds": round(time.time() - started, 2)}
    return value, details


def score(module, P):
    """Family entry point: fitness of the CEM policy trained under the candidate reward (higher is better)."""
    return score_source(check_source(module), P, cpu_seconds=int(P.get("cpu_seconds", 600)), wall_seconds=int(P.get("wall_seconds", 840)))


# Reference rewards (source code, candidate format) used by the noise gate and the tests.
REFERENCE_REWARDS = {
    "ballbalance_human": '''import math


def reward(observation, action, next_observation, info):
    # IsaacGymEnvs compute_bbot_reward: closeness of the ball to (0, 0, 0.7) times slowness of the ball.
    x, y, z = info["ball_position"]
    distance = math.sqrt(x * x + y * y + (z - 0.7) ** 2)
    speed = math.sqrt(sum(v * v for v in info["ball_velocity"]))
    return (1.0 / (1.0 + distance)) * (1.0 / (1.0 + speed))
''',
    "quadcopter_human": '''import math


def reward(observation, action, next_observation, info):
    # IsaacGymEnvs compute_quadcopter_reward: position reward plus uprightness and low spin near the target.
    distance = info["target_distance"]
    pos_reward = 1.0 / (1.0 + distance * distance)
    x, y, z, w = info["root_quat"]
    up_z = 1.0 - 2.0 * (x * x + y * y)  # z component of the body z axis
    tilt = abs(1.0 - up_z)
    up_reward = 1.0 / (1.0 + tilt * tilt)
    spin = abs(info["root_angvel"][2])
    spin_reward = 1.0 / (1.0 + spin * spin)
    return pos_reward + pos_reward * (up_reward + spin_reward)
''',
    "zero": '''def reward(observation, action, next_observation, info):
    return 0.0
''',
    "alive": '''def reward(observation, action, next_observation, info):
    return 1.0
''',
    "quadcopter_sparse": '''def reward(observation, action, next_observation, info):
    return -info["target_distance"]
''',
}
