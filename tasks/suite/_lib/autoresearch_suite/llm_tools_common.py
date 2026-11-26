"""Shared plumbing of the llm_tools evaluator package: prepared data and sandboxed candidates.

Every evaluator of this package keeps reference answers (ground truth,
gold titles, recorded tool traces, labels) in files under ``data_dir()``.
Candidates therefore never run in the evaluator process: ``call_candidate``
executes one entry point of the candidate source in the resource-limited
child of ``software_sandbox`` (no file access outside the Python
installation, no sockets, no subprocesses).  Capabilities the task grants
(the model client, a search engine, a tool executor) are passed to the
candidate as *proxies*: callables that forward their arguments to handlers
running here, in the trusted process, which enforce budgets and log calls.
"""
from __future__ import annotations

import json
from pathlib import Path

from arena_task_paths import data_root
from . import software_sandbox as sandbox
from .runtime import CandidateError


def data_dir():
    return data_root() / "llm_tools"


def data_file(name):
    path = data_dir() / name
    if not path.exists():
        raise CandidateError(f"prepared data missing: {path}; run `python scripts/prepare_llm_tools.py`")
    return path


def read_json(name):
    return json.loads(data_file(name).read_text(encoding="utf-8"))


class ProxyError(Exception):
    """A non-fatal proxy failure; the candidate receives it as RuntimeError and may continue."""


# Appended to the candidate source inside the sandbox child.  It builds the proxy
# callables, calls the requested entry point and converts the result to plain JSON data.
WRAPPER = '''


def _artask_entry_(_artask_payload, _artask_oracle):
    class _ArtaskProxy:
        def __init__(self, op):
            self.op = op

        def __call__(self, *args, **kwargs):
            return _artask_oracle({"op": self.op, "args": list(args), "kwargs": kwargs})

        def __repr__(self):
            return "<" + self.op + " proxy>"

    def _artask_plain(value, depth=0):
        if depth > 40:
            raise ValueError("result nested too deeply")
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): _artask_plain(v, depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_artask_plain(v, depth + 1) for v in value]
        if hasattr(value, "tolist"):
            return _artask_plain(value.tolist(), depth + 1)
        if hasattr(value, "item"):
            return _artask_plain(value.item(), depth + 1)
        raise TypeError("unsupported result type " + type(value).__name__)

    function = globals().get(_artask_payload["entry"])
    if not callable(function):
        raise AttributeError("entry point " + _artask_payload["entry"] + "() is missing or not callable")
    proxies = [_ArtaskProxy(name) for name in _artask_payload["proxies"]]
    if _artask_payload.get("loop"):
        return [_artask_plain(function(*list(args), *proxies)) for args in _artask_payload["args"]]
    return _artask_plain(function(*list(_artask_payload["args"]), *proxies))
'''


def call_candidate(source, entry, args, *, proxies=(), handlers=None, loop=False, cpu_seconds=60, wall_seconds=None,
                   memory_mb=2048, filename="candidate.py", harness=WRAPPER, harness_entry="_artask_entry_", payload=None):
    """Run ``entry(*args, *proxy_callables)`` (or one call per argument list when ``loop``) in the sandbox.

    Returns ``(ok, value, failure)``: ``failure`` is None on success, else
    ``{"kind": ..., "message": ...}`` for a crash, timeout or protocol error of
    the candidate.  ``harness``/``harness_entry``/``payload`` replace the
    generic wrapper for evaluators with their own child-side loop.  A
    ``CandidateError`` raised by a handler (an exhausted
    model budget, a rejected model request) is fatal and re-raised here even
    if the candidate caught the proxy's RuntimeError.
    """
    handlers = dict(handlers or {})
    fatal = []

    def oracle(request):
        if not isinstance(request, dict) or request.get("op") not in handlers:
            raise CandidateError("unknown proxy request")
        args_, kwargs = request.get("args") or [], request.get("kwargs") or {}
        if not isinstance(args_, list) or not isinstance(kwargs, dict):
            raise CandidateError("malformed proxy request")
        try:
            return handlers[request["op"]](*args_, **kwargs)
        except ProxyError as error:
            raise CandidateError(str(error)) from None
        except TypeError as error:
            raise CandidateError(f"bad arguments for {request['op']}: {error}") from None
        except CandidateError as error:
            fatal.append(error)
            raise

    if payload is None:
        payload = {"entry": entry, "args": list(args), "proxies": list(proxies), "loop": bool(loop)}
    try:
        value = sandbox.execute(source + harness, harness_entry, [payload], oracle=oracle, cpu_seconds=cpu_seconds,
                                wall_seconds=wall_seconds, memory_mb=memory_mb, filename=filename)
        result = (True, value, None)
    except sandbox.SandboxFailure as failure:
        last = failure.traceback_text.strip().splitlines()[-1] if failure.traceback_text.strip() else ""
        message = str(failure) if not last or last in str(failure) else f"{failure}; {last}"
        result = (False, None, {"kind": failure.kind, "message": message[:600]})
    if fatal:
        raise fatal[0]
    return result


def candidate_source(module, *entries):
    """Source text of the imported candidate module after a cheap check of its entry points."""
    path = sandbox.require_entry(module, *entries)
    return Path(path).read_text(encoding="utf-8")


def llm_handler(client):
    """Proxy handler for the budgeted OpenRouter client: llm(prompt_or_messages, system=None, ...) -> str."""
    def handle(prompt, system=None, max_tokens=None, temperature=None, json_mode=False):
        if not isinstance(prompt, (str, list)):
            raise ProxyError("llm prompt must be a string or a list of role/content messages")
        try:
            return client(prompt, system=system, max_tokens=max_tokens, temperature=temperature, json_mode=bool(json_mode))
        except CandidateError as error:
            if "messages must be" in str(error):
                raise ProxyError(str(error)) from None
            raise
    return handle
