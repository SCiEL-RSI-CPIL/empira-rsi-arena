"""Polyglot coding-agent evaluator (Aider Polyglot exercises, pass@1, DGM-style).

The candidate is a coding agent::

    solve(exercise, llm) -> dict[str, str]

``exercise`` holds what the agent may see (see ``polyglot_data``): ``id``
(``<language>__<slug>``), ``language``, ``instructions`` (the exercise text),
``files`` (the editable solution stubs, path -> content) and ``context``
(read-only build files, path -> content). ``llm`` is a budgeted, cached
OpenRouter chat client. ``solve`` returns the new content of the solution files
it changes (a subset of ``exercise["files"]``); files it leaves out keep their
stub content. The agent runs in a resource-limited child process
(``software_sandbox``) that cannot read the prepared data, so it never sees the
tests, the reference solutions or any test result; model calls are forwarded to
this process. One child serves one exercise.

After the agent has finished, the exercise's hidden tests run once in a fresh
container (``polyglot_runner``). The score is pass@1: the fraction of the fixed
exercise list whose tests pass. As in DGM's harness, an agent that crashes, times
out or changes nothing scores 0 on that exercise; returning a non-dict, a path
that is not a solution file of the exercise, or oversized content violates the
contract and invalidates the evaluation.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

from . import llm as llm_client
from . import polyglot_data, polyglot_image, polyglot_runner, software_sandbox
from .runtime import CandidateError

CONTRACT_ERROR = "PolyglotContractError"
MAX_FILE_CHARS = 200_000
MAX_TOTAL_CHARS = 1_000_000
MAX_TOKENS_CAP = 8192
# Model errors the agent caused (or may handle): returned to the agent as an exception; anything else stops the evaluation.
AGENT_MODEL_ERRORS = ("per-evaluation model budget", "model request rejected: HTTP 400", "messages must be", "model returned no choices")

SHIM = '''


class PolyglotContractError(Exception):
    pass


def __polyglot_solve__(exercise, oracle_fn):
    import inspect as _inspect

    class _ModelClient:
        """Budgeted model client: llm(prompt_or_messages, system=None, max_tokens=None, temperature=None, json_mode=False) -> str."""

        def __call__(self, prompt, *, system=None, max_tokens=None, temperature=None, json_mode=False):
            return oracle_fn({"op": "llm", "prompt": prompt, "system": system, "max_tokens": max_tokens,
                              "temperature": temperature, "json_mode": bool(json_mode)})

    solve = globals().get("solve")
    if not callable(solve):
        raise PolyglotContractError("the candidate must define solve(exercise, llm)")
    try:
        _inspect.signature(solve).bind(None, None)
    except TypeError:
        raise PolyglotContractError("solve must accept (exercise, llm)") from None
    except ValueError:
        pass
    result = solve(exercise, _ModelClient())
    if not isinstance(result, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in result.items()):
        raise PolyglotContractError("solve must return a dict mapping solution file paths (str) to file contents (str), got "
                                    + type(result).__name__)
    return dict(result)
'''


class ExerciseOracle:
    """Forwards the agent's model calls for one exercise to its own budgeted client."""

    def __init__(self, llm, exercise_id):
        self.llm, self.exercise_id = llm, exercise_id
        self.calls = 0
        self.model_errors = []
        self.fatal = None

    def __call__(self, request):
        if not isinstance(request, dict) or request.get("op") != "llm":
            raise CandidateError("unknown request from the agent")
        prompt = request.get("prompt")
        if not isinstance(prompt, (str, list)):
            raise CandidateError("llm prompt must be a str or a list of role/content messages")
        options = {"system": request.get("system") if isinstance(request.get("system"), str) else None, "json_mode": bool(request.get("json_mode"))}
        try:
            if request.get("max_tokens") is not None:
                options["max_tokens"] = max(1, min(int(request["max_tokens"]), MAX_TOKENS_CAP))
            if request.get("temperature") is not None:
                options["temperature"] = float(request["temperature"])
        except (TypeError, ValueError, OverflowError):
            raise CandidateError("max_tokens and temperature must be numbers") from None
        self.calls += 1
        try:
            return self.llm(prompt, **options)
        except CandidateError as error:
            text = str(error)
            if text.startswith(AGENT_MODEL_ERRORS):
                self.model_errors.append(text[:300])
                if text.startswith("per-evaluation model budget"):
                    raise CandidateError(f"the model budget of this exercise (${self.llm.budget:.4f}) is exhausted") from None
                raise
            self.fatal = f"model service failure on {self.exercise_id}: {text}"
            raise RuntimeError(self.fatal) from None


def check_candidate(module):
    path = software_sandbox.require_entry(module, "solve")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    for reserved in ("__polyglot_solve__", CONTRACT_ERROR):
        if reserved in source:
            raise CandidateError(f"the name {reserved} is reserved")
    return source


def validate_files(result, exercise):
    """The agent's edits: only solution files of the exercise, bounded text; returns the changed files."""
    if not isinstance(result, dict):
        raise CandidateError("solve must return a dict of solution file contents")
    allowed = set(exercise.solution_paths())
    stubs = exercise.stubs()
    total = 0
    changed = {}
    for path, text in result.items():
        if path not in allowed:
            raise CandidateError(f"solve returned {path!r}, which is not a solution file of {exercise.id} (allowed: {sorted(allowed)})")
        if not isinstance(text, str) or "\x00" in text:
            raise CandidateError(f"the content returned for {path!r} must be text without NUL characters")
        if len(text) > MAX_FILE_CHARS:
            raise CandidateError(f"the content returned for {path!r} exceeds {MAX_FILE_CHARS} characters")
        total += len(text)
        if text != stubs.get(path):
            changed[path] = text
    if total > MAX_TOTAL_CHARS:
        raise CandidateError(f"solve returned more than {MAX_TOTAL_CHARS} characters for {exercise.id}")
    return changed


def task_settings(P, n_exercises):
    model = os.environ.get("ARTASK_LLM_MODEL") or P.get("model") or llm_client.DEFAULT_MODEL
    budget = float(os.environ.get("ARTASK_LLM_BUDGET_USD") or P.get("budget_usd", 0.5))
    if budget <= 0:
        raise CandidateError("the model budget must be positive")
    return {"model": model, "budget_usd": budget, "budget_usd_per_exercise": budget / n_exercises,
            "agent_seconds": float(P.get("agent_seconds", 300)), "agent_cpu_seconds": int(P.get("agent_cpu_seconds", 120)),
            "agent_memory_mb": int(P.get("agent_memory_mb", 512)),
            "test_seconds": float(P.get("test_seconds", 180)), "agent_workers": int(P.get("agent_workers", 4)),
            "test_workers": int(P.get("test_workers", 2)), "test_cpus": float(P.get("test_cpus", 2.0)),
            "test_memory_gb": float(P.get("test_memory_gb", 3.0))}


def run_agent(source, exercise, llm, settings):
    """Run solve() for one exercise in the sandbox; returns (changed files or None, record)."""
    oracle = ExerciseOracle(llm, exercise.id)
    record = {"id": exercise.id, "language": exercise.language}
    started = time.time()
    files = None
    try:
        value = software_sandbox.execute(source + SHIM, "__polyglot_solve__", [exercise.visible()], oracle=oracle,
                                         cpu_seconds=settings["agent_cpu_seconds"], wall_seconds=settings["agent_seconds"],
                                         memory_mb=settings.get("agent_memory_mb", 512), filename="candidate.py")
        files = validate_files(value, exercise)
        record["agent_status"] = "ok"
    except software_sandbox.SandboxFailure as failure:
        text = str(failure)
        if oracle.fatal:
            raise RuntimeError(oracle.fatal) from None
        if CONTRACT_ERROR in text:
            raise CandidateError(text.split(CONTRACT_ERROR + ":", 1)[-1].strip()[:500]) from None
        record["agent_status"] = "timeout" if failure.kind == "timeout" else "crash"
        record["agent_error"] = f"{failure.kind}: {text[:300]}"
    record.update({"agent_seconds": round(time.time() - started, 2), "llm_calls": oracle.calls, "llm": llm.usage()})
    if oracle.model_errors:
        record["model_errors"] = oracle.model_errors[:3]
    return files, record


def evaluate_exercises(source, exercises, settings, image):
    """Agents in parallel (``agent_workers``); each finished agent's tests queue on ``test_workers`` containers."""
    cache = llm_client.Cache()
    records = {}

    def test(exercise, files, record):
        outcome = polyglot_runner.run_exercise(image, exercise, files, seconds=settings["test_seconds"], cpus=settings["test_cpus"],
                                               memory_gb=settings["test_memory_gb"])
        record.update({k: v for k, v in outcome.items() if k != "id"})
        return record

    tests = ThreadPoolExecutor(max_workers=max(1, settings["test_workers"]))
    agents = ThreadPoolExecutor(max_workers=max(1, settings["agent_workers"]))

    def agent(exercise):
        llm = llm_client.LLM(model=settings["model"], budget_usd=settings["budget_usd_per_exercise"], cache=cache)
        files, record = run_agent(source, exercise, llm, settings)
        record["edited_files"] = sorted(files or {})
        if not files:  # DGM: an empty patch is not tested and counts as unresolved
            record.update({"resolved": False, "outcome": "no_edit" if record["agent_status"] == "ok" else record["agent_status"]})
            return None, record
        return tests.submit(test, exercise, files, record), record

    try:
        pending = [(exercise, agents.submit(agent, exercise)) for exercise in exercises]
        for exercise, future in pending:
            test_future, record = future.result()
            records[exercise.id] = test_future.result() if test_future is not None else record
    finally:  # on a contract violation, service failure or deadline, queued work is dropped (running work ends at its own limit)
        agents.shutdown(wait=False, cancel_futures=True)
        tests.shutdown(wait=False, cancel_futures=True)
    for record in records.values():
        record.setdefault("outcome", "resolved" if record.get("resolved") else ("tests_timed_out" if record.get("timed_out") else "tests_failed"))
    return [records[e.id] for e in exercises]


def total_usage(records, model):
    usage = {"model": model, "requests": 0, "cached_responses": 0, "cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    for record in records:
        for key in ("requests", "cached_responses", "prompt_tokens", "completion_tokens"):
            usage[key] += int(record["llm"].get(key, 0))
        usage["cost_usd"] += float(record["llm"].get("cost_usd", 0.0))
    usage["cost_usd"] = round(usage["cost_usd"], 6)
    return usage


def evaluate(module, P):
    """pass@1 of the candidate agent on the task's fixed exercise list (higher is better)."""
    started = time.time()
    source = check_candidate(module)
    subset = P.get("subset", "small")
    ids = list(subset) if isinstance(subset, (list, tuple)) else polyglot_data.subset_ids(subset)
    override = os.environ.get("ARTASK_POLYGLOT_EXERCISES")
    if override:
        ids = ids[:max(1, int(override))]
    if not ids:
        raise CandidateError("the exercise list is empty")
    exercises = [polyglot_data.Exercise(i) for i in ids]
    for exercise in exercises:
        exercise.verify()
    image = polyglot_image.require_image()
    settings = task_settings(P, len(exercises))
    records = evaluate_exercises(source, exercises, settings, image["tag"])
    resolved = sum(1 for r in records if r["resolved"])
    by_language = {}
    for record in records:
        row = by_language.setdefault(record["language"], [0, 0])
        row[0] += int(record["resolved"])
        row[1] += 1
    details = {"subset": subset if isinstance(subset, str) else "custom", "exercises": len(records), "resolved": resolved,
               "by_language": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_language.items())},
               "agent_crashes": sum(1 for r in records if r.get("agent_status") in ("crash", "timeout")),
               "exit0_without_summary": sum(1 for r in records if r.get("exit0_without_summary")),
               "usage": total_usage(records, settings["model"]), "image": image,
               "timing": {"total_seconds": round(time.time() - started, 2),
                          "agent_seconds_sum": round(sum(r.get("agent_seconds", 0.0) for r in records), 2),
                          "test_seconds_sum": round(sum(r.get("test_seconds", 0.0) for r in records), 2)},
               "settings": {k: v for k, v in settings.items() if k != "model"},
               "per_exercise": [{k: v for k, v in r.items() if k != "output_tail"} | {"output_tail": r.get("output_tail", "")[-800:]}
                                for r in records]}
    return {"score": resolved / len(records), "details": details}
