"""Environment server of the agent_bench evaluators (trusted code; one adapter per process).

Started by ``agent_bench_core.EnvServer`` as ``python agent_bench_server.py <adapter> <params-json>``
with ``suite/_lib`` on ``PYTHONPATH`` and the client's task cache in
``ARENA_TASK_CACHE_DIR``. It reads one JSON request per line on
stdin and writes one JSON reply per line to the original stdout; everything an
environment library prints goes to stderr. Requests:

``ping``                       -> {}
``episodes``                   -> {"episodes": [spec, ...]}   fixed, seeded episode list
``reset`` (episode=spec)       -> {"instruction", "observation", "valid_actions", "done"}
``step`` (action=str)          -> {"observation", "reward", "done", "valid_actions"}
``score``                      -> {"score": float in [0, 1], "details": {...}}
``close``                      -> exits

The adapter module is ``autoresearch_suite.agent_bench_<adapter>`` and must
define ``Adapter(params)`` with ``episodes()``, ``reset(spec)``, ``step(action)``
and ``score()``. This file only uses the standard library so that it runs under
the pinned environment interpreter as well as the suite interpreter.
"""
import importlib
import json
import os
import sys
import traceback

if sys.path and os.path.realpath(sys.path[0]) == os.path.dirname(os.path.realpath(__file__)):
    sys.path.pop(0)  # sibling evaluator modules must not shadow the environment libraries' imports


def _plain(value):
    """JSON-safe copy (numpy scalars and tuples become plain Python values)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and value != value:
        return None
    return value


def main():
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    name, params = sys.argv[1], json.loads(sys.argv[2])
    if not name.replace("_", "").isalnum():
        raise SystemExit("bad adapter name")
    adapter = None
    error = None
    try:
        adapter = importlib.import_module(f"autoresearch_suite.agent_bench_{name}").Adapter(params)
    except Exception:  # noqa: BLE001 - reported to the client on the first request
        error = traceback.format_exc()[-2000:]
    for line in sys.stdin:
        request = json.loads(line)
        cmd = request.get("cmd")
        if cmd == "close":
            break
        try:
            if error:
                raise RuntimeError("adapter failed to start: " + error)
            if cmd == "ping":
                reply = {}
            elif cmd == "episodes":
                reply = {"episodes": adapter.episodes()}
            elif cmd == "reset":
                reply = adapter.reset(request["episode"])
            elif cmd == "step":
                reply = adapter.step(str(request["action"]))
            elif cmd == "score":
                reply = adapter.score()
            else:
                raise ValueError(f"unknown command {cmd!r}")
            text = json.dumps(_plain(reply), allow_nan=False)
        except Exception as failure:  # noqa: BLE001 - the client raises it
            text = json.dumps({"error": f"{type(failure).__name__}: {failure}", "traceback": traceback.format_exc()[-2000:]})
        protocol.write(text + "\n")
        protocol.flush()
    close = getattr(adapter, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
