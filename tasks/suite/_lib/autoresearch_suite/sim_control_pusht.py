"""Push-T heuristic controller (ENPIRE's Gym-PushT simulation task): success rate on fixed gym-pusht episodes.

The candidate is a state-based controller::

    class Policy:
        def __init__(self): ...
        def act(self, observation) -> (x, y)

``observation`` is gym-pusht's state vector [agent_x, agent_y, block_x,
block_y, block_angle] (pixels in [0, 512], angle in [0, 2 pi)); the action is
the target position of the agent's PD controller and is clipped to [0, 512].
The goal pose of the T block is fixed at (256, 256, pi / 4). An episode ends
when gym-pusht reports success (coverage of the goal region above 0.95, its own
``is_success``) or after 300 steps; the score is the fraction of the fixed
evaluation seeds that end in success.

Isolation: the simulator runs in a separate trusted server process
(sim_control_pusht_server.py, interpreter of ``<task cache>/sim_control-env``
because gym-pusht needs pymunk < 7). Every episode is played by a fresh sandboxed child
(software_sandbox: no file access outside the Python installation, no network,
no subprocesses, CPU and memory limits) that instantiates ``Policy`` and only
receives the observation of each step; its actions are forwarded to the server
by this process. The candidate therefore never holds the environment, cannot
reset or modify its state and sees neither the coverage nor the success flag.
The evaluator reads the candidate's source only (``sandboxed.run``): no
statement of the candidate runs in the evaluator process.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

from arena_task_paths import cache_path
from . import software_sandbox
from .runtime import CandidateError

SERVER = Path(__file__).resolve().with_name("sim_control_pusht_server.py")
ENTRY = "__sim_control_pusht_episode__"
HORIZON = 300  # gym_pusht/PushT-v0 max_episode_steps

SHIM = '''


class PushTContractError(Exception):
    pass


def __sim_control_pusht_episode__(observation, oracle):
    import math as _math
    import random as _random
    import numpy as _np
    _random.seed(0)
    _np.random.seed(0)
    policy_class = globals().get("Policy")
    if not callable(policy_class):
        raise PushTContractError("the candidate must define class Policy with act(observation)")
    policy = policy_class()
    obs = observation
    while True:
        action = policy.act(_np.array(obs, dtype=_np.float64))
        if isinstance(action, _np.ndarray):  # booleans, strings and nested sequences are not positions
            parts = action.tolist() if action.shape == (2,) and action.dtype.kind in "iuf" else None
        else:
            parts = list(action) if isinstance(action, (list, tuple)) and len(action) == 2 else None
        if parts is None or not all(
                (isinstance(v, (int, float, _np.integer, _np.floating)) and not isinstance(v, (bool, _np.bool_)))
                or (isinstance(v, _np.ndarray) and v.ndim == 0 and v.dtype.kind in "iuf") for v in parts):
            raise PushTContractError("act must return two numbers (x, y)")
        values = [float(v) for v in parts]
        if not all(_math.isfinite(v) for v in values):
            raise PushTContractError("act must return two finite numbers (x, y)")
        reply = oracle({"op": "step", "action": values})
        if reply["done"]:
            return reply["steps"]
        obs = reply["obs"]
'''


def env_python():
    """Interpreter of the private simulation runtime (created by scripts/prepare_sim_control.py --setup-environment)."""
    override = os.environ.get("ARTASK_SIM_CONTROL_PYTHON")
    return Path(override) if override else cache_path("sim_control-env") / "bin" / "python"


class SimServer:
    """JSON-lines client of sim_control_pusht_server.py."""

    def __init__(self, interpreter=None, timeout=60):
        python = Path(interpreter) if interpreter else env_python()
        if not python.is_file():
            raise RuntimeError(f"Push-T runtime missing: {python}; run `python3 scripts/prepare_sim_control.py --setup-environment`")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp"), "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1", "MPLBACKEND": "Agg"}
        self.timeout = timeout
        self.stderr_tail = []
        self.process = subprocess.Popen([str(python), "-I", "-u", str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, cwd=str(SERVER.parent), env=env, text=True, encoding="utf-8",
                                        start_new_session=True)
        self._drain = threading.Thread(target=self._read_stderr, daemon=True)
        self._drain.start()
        self.versions = self.call("ping")["versions"]

    def _read_stderr(self):
        for line in iter(self.process.stderr.readline, ""):
            self.stderr_tail.append(line)
            if len(self.stderr_tail) > 200:
                del self.stderr_tail[:100]

    def call(self, cmd, **payload):
        if self.process.poll() is not None:
            raise RuntimeError("Push-T server exited: " + "".join(self.stderr_tail[-20:])[-1500:])
        self.process.stdin.write(json.dumps({"cmd": cmd, **payload}) + "\n")
        self.process.stdin.flush()
        result = {}

        def read():
            result["line"] = self.process.stdout.readline()
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(self.timeout)
        if reader.is_alive() or not result.get("line"):
            self.close()
            raise RuntimeError(f"Push-T server did not answer {cmd!r}: " + "".join(self.stderr_tail[-20:])[-1500:])
        reply = json.loads(result["line"])
        if reply.get("error"):
            raise RuntimeError(f"Push-T server failed on {cmd!r}: {reply['error']}")
        return reply

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, 9)
            except OSError:
                pass
            self.process.wait(timeout=10)
        self._drain.join(timeout=5)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass


class Episode:
    """Oracle of one sandboxed episode: validates actions, steps the trusted server, ends the episode."""

    def __init__(self, server, horizon=HORIZON):
        self.server, self.horizon = server, int(horizon)
        self.steps, self.done, self.success = 0, False, False
        self.coverage = []

    def __call__(self, request):
        if not isinstance(request, dict) or request.get("op") != "step":
            raise CandidateError("malformed request from the policy process")
        if self.done:
            raise CandidateError("the episode has ended")
        action = request.get("action")
        if (not isinstance(action, list) or len(action) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in action)):
            raise CandidateError("actions must be two finite numbers")
        target = [min(512.0, max(0.0, float(v))) for v in action]
        reply = self.server.call("step", action=target)
        self.steps += 1
        self.coverage.append(reply["coverage"])
        self.success = self.success or bool(reply["success"])
        self.done = bool(reply["success"] or reply["terminated"] or reply["truncated"] or self.steps >= self.horizon)
        return {"obs": reply["obs"], "done": self.done, "steps": self.steps}


def check_source(module):
    path = software_sandbox.require_entry(module, "Policy")
    source = Path(path).read_text(encoding="utf-8")
    if ENTRY in source:
        raise CandidateError(f"the name {ENTRY} is reserved")
    return source


def play(source, server, seed, horizon=HORIZON, cpu_seconds=10, wall_seconds=120, memory_mb=1024):
    observation = server.call("reset", seed=int(seed))["obs"]
    oracle = Episode(server, horizon)
    try:
        software_sandbox.execute(source + SHIM, ENTRY, [observation], oracle=oracle, cpu_seconds=cpu_seconds,
                                 wall_seconds=wall_seconds, memory_mb=memory_mb, filename="candidate.py")
    except software_sandbox.SandboxFailure as failure:
        text = str(failure)
        contract = "PushTContractError" in text
        if contract:
            text = text.split("PushTContractError:", 1)[-1].strip()
        last = failure.traceback_text.strip().splitlines()[-1] if failure.traceback_text.strip() else ""
        detail = f" ({last})" if failure.kind == "exception" and not contract and last and last not in text else ""
        raise CandidateError(f"policy failed on episode seed {seed} ({failure.kind}): {text}{detail}"[:1500]) from None
    if not oracle.done:
        raise CandidateError(f"the policy stopped before the end of episode seed {seed}")
    return {"seed": int(seed), "success": oracle.success, "steps": oracle.steps,
            "max_coverage": round(max(oracle.coverage), 4), "final_coverage": round(oracle.coverage[-1], 4)}


def score(module, P):
    """Family entry point: success rate of the candidate controller over the fixed seeds (higher is better)."""
    source = check_source(module)
    seeds = list(range(int(P.get("first_seed", 0)), int(P.get("first_seed", 0)) + int(P.get("episodes", 100))))
    started = time.time()
    server = SimServer(timeout=int(P.get("server_timeout", 60)))
    try:
        records = [play(source, server, seed, int(P.get("horizon", HORIZON)), cpu_seconds=int(P.get("episode_cpu_seconds", 10)))
                   for seed in seeds]
        versions = server.versions
    finally:
        server.close()
    successes = sum(r["success"] for r in records)
    details = {"episodes": len(records), "successes": successes, "seeds": [seeds[0], seeds[-1]],
               "mean_max_coverage": round(sum(r["max_coverage"] for r in records) / len(records), 4),
               "mean_steps": round(sum(r["steps"] for r in records) / len(records), 2), "simulator": versions,
               "elapsed_seconds": round(time.time() - started, 2), "per_episode": records}
    return successes / len(records), details
