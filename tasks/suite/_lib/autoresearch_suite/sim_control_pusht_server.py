"""Trusted Push-T simulation server for the sim_control evaluator (gym-pusht, Apache-2.0).

Run with the interpreter of <task cache>/sim_control-env (gym-pusht 0.1.6 needs pymunk < 7):

    python sim_control_pusht_server.py

It speaks JSON lines on stdin/stdout: {"cmd": "ping" | "reset" | "step" | "close", ...}.
``reset`` takes a seed and returns the state observation; ``step`` takes a target
position (x, y) and returns the next observation, gym-pusht's coverage and its
own success flag (coverage > 0.95). The environment is created with
``gym.make("gym_pusht/PushT-v0", obs_type="state")`` (300-step time limit).
This file is self-contained: it never imports the evaluator package.
"""
import json
import os
import sys
import warnings


def main():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    out = sys.stdout
    sys.stdout = sys.stderr  # stray prints of imported packages must not corrupt the protocol
    warnings.simplefilter("ignore")
    import gymnasium as gym
    import numpy as np
    import gym_pusht  # noqa: F401 - registers gym_pusht/PushT-v0
    import pymunk

    env = gym.make("gym_pusht/PushT-v0", obs_type="state")
    started = False
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "ping":
                from importlib.metadata import version
                reply = {"ok": True, "versions": {"gym-pusht": version("gym-pusht"), "pymunk": pymunk.version,
                                                  "gymnasium": gym.__version__, "numpy": np.__version__}}
            elif command == "reset":
                observation, _ = env.reset(seed=int(request["seed"]))
                started = True
                reply = {"obs": [float(v) for v in observation]}
            elif command == "step":
                if not started:
                    raise RuntimeError("step before reset")
                action = np.clip(np.asarray(request["action"], dtype=np.float64).reshape(2), 0.0, 512.0)
                observation, _, terminated, truncated, info = env.step(action)
                reply = {"obs": [float(v) for v in observation], "coverage": float(info["coverage"]),
                         "success": bool(info["is_success"]), "terminated": bool(terminated), "truncated": bool(truncated)}
            elif command == "close":
                break
            else:
                raise ValueError(f"unknown command {command!r}")
        except Exception as error:  # noqa: BLE001 - reported to the evaluator
            reply = {"error": f"{type(error).__name__}: {error}"}
        out.write(json.dumps(reply, allow_nan=False) + "\n")
        out.flush()
    env.close()


if __name__ == "__main__":
    main()
