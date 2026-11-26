"""Episode runner shared by the agent_bench evaluators (language agents in interactive environments).

The candidate is a per-step agent policy::

    act(instruction, observation, valid_actions, history, llm) -> str

It runs in a resource-limited child process (``software_sandbox``) that sees
only what the environment shows an agent: the episode instruction, the latest
observation, the actions the environment lists as valid (possibly empty),
the history of ``(action, observation, reward)`` triples of the current
episode and a budgeted, cached model client. Model calls and environment steps
are forwarded to this (trusted) process through the sandbox oracle, so the
candidate never holds the environment, its gold plans, goal labels or reward
code. One child process serves one episode; module state does not persist
across episodes.

The environment itself runs in a separate server process
(``agent_bench_server.py``), either with this interpreter or with the pinned
interpreter of ``agent_bench-env`` in the task cache (BALROG environments need
an old gym and Python 3.11). The server is told the task cache of this process
(``ARENA_TASK_CACHE_DIR``), so both read the same prepared data. The score of
an evaluation is the mean episode score that the environment reports (success,
progression or progress rate); see the task contract for the metric of each
environment.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from arena_task_paths import cache_path, cache_root, data_root
from . import software_sandbox
from .runtime import CandidateError

LIB = Path(__file__).resolve().parents[1]
SERVER = Path(__file__).resolve().with_name("agent_bench_server.py")
MAX_ACTION_CHARS = 20_000
CONTRACT_ERROR = "AgentBenchContractError"


def data_dir():
    return data_root() / "agent_bench"


def env_python():
    """Interpreter of the pinned environment runtime (agent_bench-env in the task cache; scripts/prepare_agent_bench.py --setup-environment)."""
    override = os.environ.get("ARTASK_AGENT_BENCH_PYTHON")
    return Path(override) if override else cache_path("agent_bench-env") / "bin" / "python"


SHIM = '''


class AgentBenchContractError(Exception):
    pass


def __agent_bench_episode__(start, oracle_fn):
    import inspect as _inspect

    class _ModelClient:
        """Budgeted model client: llm(prompt_or_messages, system=None, max_tokens=None, temperature=None, json_mode=False) -> str."""

        def __call__(self, prompt, *, system=None, max_tokens=None, temperature=None, json_mode=False):
            return oracle_fn({"op": "llm", "prompt": prompt, "system": system, "max_tokens": max_tokens,
                              "temperature": temperature, "json_mode": bool(json_mode)})

    policy = globals().get("act")
    if not callable(policy):
        raise AgentBenchContractError("the candidate must define act(instruction, observation, valid_actions, history, llm)")
    try:
        _inspect.signature(policy).bind(None, None, None, None, None)
    except TypeError:
        raise AgentBenchContractError("act must accept (instruction, observation, valid_actions, history, llm)") from None
    except ValueError:
        pass
    model = _ModelClient()
    state = dict(start)
    history = []
    while not state.get("done"):
        action = policy(start["instruction"], state["observation"], list(state["valid_actions"]), list(history), model)
        if not isinstance(action, str):
            raise AgentBenchContractError("act must return a str action, got " + type(action).__name__)
        state = oracle_fn({"op": "step", "action": action})
        history.append((action, state["observation"], state["reward"]))
    return len(history)
'''


class EnvServer:
    """JSON-lines client of agent_bench_server.py running one environment adapter."""

    def __init__(self, adapter, params, interpreter=None, timeout=600):
        python = Path(interpreter) if interpreter else Path(sys.executable)
        if not python.is_file():
            raise CandidateError(f"environment interpreter missing: {python}; run `python scripts/prepare_agent_bench.py --setup-environment`")
        # The server's environment is built from scratch; without the cache location its adapters would look for the
        # prepared data in the default cache instead of the one this process was given.
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp"), "PYTHONPATH": str(LIB),
               "ARENA_TASK_CACHE_DIR": str(cache_root()),
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "MPLBACKEND": "Agg",
               "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
        for name in ("JAVA_HOME", "TMPDIR", "LANG", "LC_ALL"):
            if os.environ.get(name):
                env[name] = os.environ[name]
        self.timeout = timeout
        self.stderr_tail = []
        # Working directory: the prepared data, else a temporary directory; never the task library, which stays unwritten.
        self.process = subprocess.Popen([str(python), "-u", str(SERVER), adapter, json.dumps(params)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, cwd=str(data_dir() if data_dir().is_dir() else tempfile.gettempdir()),
                                        env=env, text=True, encoding="utf-8", start_new_session=True)
        self._drain = threading.Thread(target=self._read_stderr, daemon=True)
        self._drain.start()
        try:
            self.call("ping")
        except BaseException:  # an adapter that cannot start (missing data or packages) must not leave its server behind
            self.close()
            raise

    def _read_stderr(self):
        for line in iter(self.process.stderr.readline, ""):
            self.stderr_tail.append(line)
            if len(self.stderr_tail) > 200:
                del self.stderr_tail[:100]

    def call(self, cmd, **payload):
        if self.process.poll() is not None:
            raise RuntimeError("environment server exited: " + "".join(self.stderr_tail[-20:])[-1500:])
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
            raise RuntimeError(f"environment server did not answer {cmd!r}: " + "".join(self.stderr_tail[-20:])[-1500:])
        reply = json.loads(result["line"])
        if reply.get("error"):
            raise RuntimeError(f"environment server failed on {cmd!r}: {reply['error']}")
        return reply

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired, ValueError):
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
    """Oracle of one sandboxed episode: forwards model calls and environment steps, enforces the step cap."""

    def __init__(self, server, llm, max_steps, max_tokens_cap=8192):
        self.server, self.llm, self.max_steps, self.max_tokens_cap = server, llm, int(max_steps), max_tokens_cap
        self.steps = 0
        self.done = False
        self.fatal = None
        self.llm_calls = 0
        self.ended = "running"

    def __call__(self, request):
        if not isinstance(request, dict):
            raise CandidateError("malformed request")
        op = request.get("op")
        if op == "llm":
            return self._model(request)
        if op == "step":
            return self._step(request.get("action"))
        raise CandidateError(f"unknown request {op!r}")

    def _model(self, request):
        prompt = request.get("prompt")
        if not isinstance(prompt, (str, list)):
            raise CandidateError("llm prompt must be a str or a list of role/content messages")
        options = {"system": request.get("system") if isinstance(request.get("system"), str) else None, "json_mode": bool(request.get("json_mode"))}
        try:
            if request.get("max_tokens") is not None:
                options["max_tokens"] = max(1, min(int(request["max_tokens"]), self.max_tokens_cap))
            if request.get("temperature") is not None:
                options["temperature"] = float(request["temperature"])
        except (TypeError, ValueError, OverflowError):
            raise CandidateError("max_tokens and temperature must be numbers") from None
        self.llm_calls += 1
        try:
            return self.llm(prompt, **options)
        except CandidateError as error:  # budget exhausted, rejected request or failing endpoint: the evaluation stops
            self.fatal = self.fatal or str(error)
            raise

    def _step(self, action):
        if self.done:
            raise CandidateError("the episode has ended")
        if not isinstance(action, str):
            raise CandidateError("actions must be strings")
        self.steps += 1
        reply = self.server.call("step", action=action[:MAX_ACTION_CHARS])
        if reply["done"]:
            self.ended = "done"
        elif self.steps >= self.max_steps:
            self.ended = "step_cap"
        self.done = bool(reply["done"]) or self.steps >= self.max_steps
        return {"observation": reply["observation"], "reward": float(reply.get("reward", 0.0)),
                "valid_actions": list(reply.get("valid_actions", [])), "done": self.done}


def check_candidate(module):
    path = software_sandbox.require_entry(module, "act")
    source = Path(path).read_text(encoding="utf-8")
    if "__agent_bench_episode__" in source:
        raise CandidateError("the name __agent_bench_episode__ is reserved")
    return source


def run_episode(source, server, llm, spec, max_steps, cpu_seconds=120, wall_seconds=1200, memory_mb=2048):
    """Play one episode in a sandboxed child; returns the per-episode record (score from the environment)."""
    start = server.call("reset", episode=spec)
    oracle = Episode(server, llm, max_steps)
    record = {"id": spec["id"]}
    if start.get("done"):
        oracle.done, oracle.ended = True, "done"
    else:
        payload = {"instruction": start["instruction"], "observation": start["observation"],
                   "valid_actions": list(start.get("valid_actions", [])), "done": False}
        started = time.time()
        try:
            software_sandbox.execute(source + SHIM, "__agent_bench_episode__", [payload], oracle=oracle, cpu_seconds=cpu_seconds,
                                     wall_seconds=wall_seconds, memory_mb=memory_mb, filename="candidate.py")
        except software_sandbox.SandboxFailure as failure:
            text = str(failure)
            if CONTRACT_ERROR in text:
                raise CandidateError(text.split(CONTRACT_ERROR + ":", 1)[-1].strip()[:500]) from None
            if oracle.fatal:
                raise CandidateError(oracle.fatal) from None
            oracle.ended = "timeout" if failure.kind == "timeout" else "crash"
            record["crash"] = f"{failure.kind}: {text[:300]}"
        if oracle.fatal:
            raise CandidateError(oracle.fatal)
        record["seconds"] = round(time.time() - started, 2)
    outcome = server.call("score")
    value = float(outcome["score"])
    if not (0.0 <= value <= 1.0) or not math.isfinite(value):
        raise RuntimeError(f"environment returned an episode score outside [0, 1]: {value}")
    record.update({"score": round(value, 6), "steps": oracle.steps, "llm_calls": oracle.llm_calls, "ended": oracle.ended})
    record.update(outcome.get("details", {}))
    return record


def evaluate(module, P, adapter, *, interpreter=None, default_steps=30, server_timeout=600):
    """Mean episode score of the candidate over the adapter's fixed episode list (higher is better)."""
    from . import llm_bench_core
    source = check_candidate(module)
    llm = llm_bench_core.task_llm(P)
    server = EnvServer(adapter, P, interpreter=interpreter, timeout=server_timeout)
    try:
        episodes = server.call("episodes")["episodes"]
        override = os.environ.get("ARTASK_AGENT_EPISODES")
        if override:
            episodes = episodes[:max(1, int(override))]
        if not episodes:
            raise RuntimeError("the environment adapter returned no episodes")
        records = []
        for spec in episodes:
            records.append(run_episode(source, server, llm, spec, int(P.get("max_steps", default_steps)),
                                       cpu_seconds=int(P.get("cpu_seconds", 120)), wall_seconds=int(P.get("episode_seconds", 1200))))
    finally:
        server.close()
    scores = [r["score"] for r in records]
    details = {"episodes": len(records), "crashes": sum(1 for r in records if "crash" in r),
               "successes": sum(1 for r in records if r.get("success")), "usage": llm.usage(), "per_episode": records}
    return {"score": sum(scores) / len(scores), "details": details}


def evaluate_task(module, P):
    """Family entry point: P['adapter'] names the environment adapter, P['runtime'] == 'pinned' selects agent_bench-env of the task cache."""
    interpreter = env_python() if P.get("runtime") == "pinned" else None
    return evaluate(module, P, P["adapter"], interpreter=interpreter, default_steps=int(P.get("max_steps", 30)))
