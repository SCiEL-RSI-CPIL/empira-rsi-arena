"""Observation and info views of the Eureka analogues' raw simulator states (pure NumPy).

This file must stay self-contained (standard library and NumPy only): its source
is also executed inside the sandboxed child that evaluates candidate rewards, so
the policy observations and reward inputs are computed by the same code on both
sides. Raw states are produced by sim_control_eureka_envs.

BallBalance analogue raw state (13): tray tilts about x and y (2), tilt rates (2),
ball position (3), ball linear velocity (3), tray normal (3).
Quadcopter raw state (25): chassis position (3), quaternion w, x, y, z (4), linear
velocity (3), angular velocity in the world frame (3), rotor pitch/roll joint
positions in Isaac's order pitch0, roll0, ..., roll3 (8), rotor thrusts in newtons (4).
"""
import math

import numpy as np

OBS_CLIP = 5.0
TRAY_HEIGHT = 0.36 * math.sqrt(2) + 2 * 0.02 + 0.5 * 0.02
BALL_RADIUS = 0.1
TRAY_EPISODE = 500
QUAD_EPISODE = 500
QUAD_TARGET = (0.0, 0.0, 1.0)


def observe_ballbalance(state):
    return np.clip(np.asarray(state[:10], dtype=np.float64), -OBS_CLIP, OBS_CLIP)


def info_ballbalance(state, progress, terminated):
    return {"ball_position": state[4:7].copy(), "ball_velocity": state[7:10].copy(), "tray_tilt": state[0:2].copy(),
            "tray_tilt_velocity": state[2:4].copy(), "tray_normal": state[10:13].copy(),
            "tray_position": np.array([0.0, 0.0, TRAY_HEIGHT]), "ball_radius": BALL_RADIUS, "progress": int(progress),
            "max_episode_length": TRAY_EPISODE, "terminated": bool(terminated)}


def observe_quadcopter(state):
    obs = np.empty(21)
    obs[0] = (QUAD_TARGET[0] - state[0]) / 3.0
    obs[1] = (QUAD_TARGET[1] - state[1]) / 3.0
    obs[2] = (QUAD_TARGET[2] - state[2]) / 3.0
    obs[3:6] = state[4:7]  # Isaac quaternions are ordered (x, y, z, w)
    obs[6] = state[3]
    obs[7:10] = state[7:10] / 2.0
    obs[10:13] = state[10:13] / math.pi
    obs[13:21] = state[13:21]
    return np.clip(obs, -OBS_CLIP, OBS_CLIP)


def info_quadcopter(state, progress, terminated):
    position = state[0:3].copy()
    target = np.array(QUAD_TARGET)
    return {"root_position": position, "root_quat": np.concatenate([state[4:7], state[3:4]]), "root_linvel": state[7:10].copy(),
            "root_angvel": state[10:13].copy(), "dof_positions": state[13:21].copy(), "thrusts": state[21:25].copy(),
            "target": target, "target_distance": float(np.linalg.norm(position - target)), "progress": int(progress),
            "max_episode_length": QUAD_EPISODE, "terminated": bool(terminated)}


VIEWS = {"ballbalance": (observe_ballbalance, info_ballbalance), "quadcopter": (observe_quadcopter, info_quadcopter)}
STATE_DIMS = {"ballbalance": 13, "quadcopter": 25}
ACTION_DIMS = {"ballbalance": 2, "quadcopter": 12}


class RewardContractError(Exception):
    """The candidate reward broke its contract (raised inside the reward process)."""


def real_number(value):
    """``value`` as a float if it is a real scalar (Python or NumPy int or float, or a 0-d numeric array).

    Booleans, strings (also numeric ones), complex numbers, sequences and arrays with dimensions are not rewards.
    """
    scalar = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
    if not scalar and not (isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in "iuf"):
        raise RewardContractError("reward must return a real number, got " + type(value).__name__)
    return float(value)


def rollout_returns(task, batch, function):
    """Sum of candidate rewards along each recorded rollout of one CEM generation.

    ``batch`` holds base64 little-endian float64 raw states (T + 1 per rollout) and
    actions (T per rollout), the rollout lengths and whether each rollout ended by
    failure. The reward of step t sees the observations before and after the
    step, the action and the info view of the state after the step.
    """
    import base64
    observe, info_view = VIEWS[task]
    states = np.frombuffer(base64.b64decode(batch["states"]), dtype="<f8").reshape(-1, STATE_DIMS[task])
    actions = np.frombuffer(base64.b64decode(batch["actions"]), dtype="<f8").reshape(-1, ACTION_DIMS[task])
    returns, s0, a0 = [], 0, 0
    for length, ended in zip(batch["lengths"], batch["terminated"]):
        length = int(length)
        total = 0.0
        observation = observe(states[s0])
        for t in range(length):
            state = states[s0 + t + 1]
            next_observation = observe(state)
            value = function(observation.copy(), actions[a0 + t].copy(), next_observation.copy(),
                             info_view(state, t + 1, bool(ended) and t == length - 1))
            value = real_number(value)
            if not math.isfinite(value):
                raise RewardContractError("reward returned a non-finite value")
            total += value
            observation = next_observation
        returns.append(total)
        s0 += length + 1
        a0 += length
    return returns
