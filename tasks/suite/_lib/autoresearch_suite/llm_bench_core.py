"""Shared plumbing for the llm_bench evaluators: prepared data, fixed samples, model clients, parsing.

Benchmarks of this package read files that ``scripts/prepare_llm_bench.py``
downloads and verifies against ``catalog/llm_bench-data-lock.json`` and writes
to ``data/llm_bench`` in the task cache (``arena_task_paths.data_root``); nothing
here touches the network except the OpenRouter client of ``llm.LLM``.

Samples are fixed: items are ordered by ``sha256(f"{seed}|{item_id}")`` and the
first ``n`` are evaluated, so the sample does not depend on the Python version
or on the order in which a dataset was serialized.  ``n`` comes from the task
parameters unless ``ARTASK_LLM_SAMPLES`` overrides it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal

from arena_task_paths import data_root
from . import llm as llm_client
from .runtime import CandidateError

DATA = data_root() / "llm_bench"
MAX_ANSWER_CHARS = 200_000
# List prices (USD per million tokens) used only for the client's pre-call budget reservation;
# the recorded cost is the one OpenRouter reports for each response.
EXTRA_PRICES = {"qwen/qwen3-8b": (0.12, 0.46), "openai/gpt-4.1-mini": (0.40, 1.60), "google/gemini-2.5-flash-lite": (0.10, 0.40)}
for _model, _price in EXTRA_PRICES.items():
    llm_client.PRICES.setdefault(_model, _price)


# ---------------------------------------------------------------------------
# Prepared data and fixed samples
# ---------------------------------------------------------------------------

def data_path(name):
    path = DATA / name
    if not path.is_file():
        raise CandidateError(f"prepared data file missing: {path}; run `python scripts/prepare_llm_bench.py`")
    return path


def read_json(name):
    raw = data_path(name).read_bytes()
    if name.endswith(".gz"):
        import gzip
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def read_jsonl(name):
    with data_path(name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_size(P, default=30):
    override = os.environ.get("ARTASK_LLM_SAMPLES")
    value = int(override) if override else int(P.get("samples", default))
    if value < 1:
        raise CandidateError("the sample size must be positive")
    return value


def hash_order(items, seed, key="id"):
    """Items sorted by a seeded SHA-256 of their identifiers (stable across Python versions)."""
    def digest(item):
        return hashlib.sha256(f"{seed}|{item[key]}".encode()).hexdigest()
    return sorted(items, key=digest)


def fixed_sample(items, n, seed, key="id"):
    ids = [item[key] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("item identifiers must be unique for a fixed sample")
    return hash_order(items, seed, key)[:n]


def half_split(items, part, salt="split", key="id"):
    """Deterministic two-way split used when a source splits a dataset into equal halves."""
    ordered = hash_order(items, salt, key)
    middle = (len(ordered) + 1) // 2
    return ordered[:middle] if part == "train" else ordered[middle:]


# ---------------------------------------------------------------------------
# Model clients
# ---------------------------------------------------------------------------

def task_llm(P):
    model = os.environ.get("ARTASK_LLM_MODEL") or P.get("model") or llm_client.DEFAULT_MODEL
    budget = float(os.environ.get("ARTASK_LLM_BUDGET_USD") or P.get("budget_usd", llm_client.BUDGET_USD))
    return llm_client.LLM(model=model, budget_usd=budget)


def judge_llm(P):
    model = os.environ.get("ARTASK_LLM_JUDGE_MODEL") or P.get("judge_model") or llm_client.DEFAULT_MODEL
    budget = float(os.environ.get("ARTASK_LLM_JUDGE_BUDGET_USD") or P.get("judge_budget_usd", 0.5))
    return llm_client.LLM(model=model, budget_usd=budget, max_tokens=int(P.get("judge_max_tokens", 4096)))


def require_signature(function, arity, description):
    """The candidate entry point must be callable with ``arity`` positional arguments."""
    import inspect
    if not callable(function):
        raise CandidateError(f"the candidate must define {description}")
    try:
        inspect.signature(function).bind(*([None] * arity))
    except TypeError:
        raise CandidateError(f"the candidate must define {description}") from None
    except ValueError:  # callables without an introspectable signature are tried as they are
        pass


def checked_answer(answer, kind="answer"):
    if not isinstance(answer, str):
        raise CandidateError(f"solve must return a str {kind}, got {type(answer).__name__}")
    return answer[:MAX_ANSWER_CHARS]


def _deadline_pending():
    """Whether an alarm is armed: ``runtime.Deadline`` arms one for the whole evaluation and it fires once."""
    return hasattr(signal, "getitimer") and signal.getitimer(signal.ITIMER_REAL)[0] > 0


def run_items(items, respond, score, llm, judge=None):
    """Mean of ``score(item, answer)`` over the items; a crashing scaffold scores 0 on that item.

    ``respond(item)`` calls the candidate; ``CandidateError`` (contract violations and an
    exhausted model budget) aborts the whole evaluation, and so does the evaluation's deadline:
    its ``TimeoutError`` is raised wherever the alarm finds the program, where the handler for
    item crashes below (or the model client's retry loop) would take it for an ordinary error
    and the remaining items would run with no deadline at all. The deadline is therefore
    recognised by its timer, which was armed when the loop started and is not any more.
    """
    if not items:
        raise CandidateError("the benchmark sample is empty")
    scores, crashes, records = [], 0, []
    timed = _deadline_pending()

    def check_deadline():
        if timed and not _deadline_pending():
            raise TimeoutError("the evaluation deadline passed during a candidate or model call")

    for item in items:
        check_deadline()
        try:
            answer = respond(item)
        except CandidateError:
            raise
        except Exception as error:  # noqa: BLE001 - a crashing scaffold scores zero on that item
            check_deadline()
            crashes += 1
            scores.append(0.0)
            records.append({"id": item["id"], "score": 0.0, "crash": f"{type(error).__name__}: {str(error)[:200]}"})
            continue
        result = score(item, answer)
        extra = {}
        if isinstance(result, dict):  # a scorer may attach diagnostics (e.g. a predicted label)
            extra = {key: value for key, value in result.items() if key != "score"}
            result = result["score"]
        value = float(result)
        if not (0.0 <= value <= 1.0) or not math.isfinite(value):
            raise ValueError(f"per-item score out of range: {value}")
        scores.append(value)
        records.append({"id": item["id"], "score": round(value, 6), **extra})
    check_deadline()
    details = {"items": len(items), "crashes": crashes, "usage": llm.usage(), "per_item": records}
    if judge is not None:
        details["judge_usage"] = judge.usage()
    return {"score": sum(scores) / len(scores), "details": details}


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

# "\s*(?:\*\*\s*)?" accepts what "\s*(?:\*\*)?\s*" accepts, without the two adjacent \s* that made a long run of
# blanks after the word cost quadratic time.
_ANSWER = re.compile(r"(?:final\s+answer|answer)\s*(?:\*\*\s*)?[:：]\s*(?:\*\*)?(.*)", re.I)
_DEC = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_NUMBER = re.compile(rf"\\[dt]?frac\{{({_DEC})\}}\{{({_DEC})\}}|({_DEC})(?:\s*/\s*({_DEC}))?")


def last_boxed(text):
    """Content of the last \\boxed{...} with balanced braces, or None.

    One pass pairs every brace with its partner (a closing brace without an opening one is skipped), then the
    last \\boxed{ whose brace has a partner is returned; a reply made of unclosed \\boxed{ costs linear time.
    """
    text = str(text)
    if "\\boxed{" not in text:
        return None
    partner, stack = {}, []
    for j, char in enumerate(text):
        if char == "{":
            stack.append(j)
        elif char == "}" and stack:
            partner[stack.pop()] = j
    start = text.rfind("\\boxed{")
    while start != -1:
        i = start + len("\\boxed")
        if i in partner:
            return text[i + 1:partner[i]]
        start = text.rfind("\\boxed{", 0, start)
    return None


def clean_fragment(text):
    text = str(text).strip()
    for _ in range(3):
        text = text.strip().strip("*_`'\"").strip()
        if text.startswith("$") and text.endswith("$") and len(text) > 1:
            text = text.strip("$")
    return text.rstrip(".").strip()


def final_answer(text):
    """Text after the last 'Answer:' marker, else the last \\boxed{}, else the last non-empty line."""
    text = str(text)
    found = [match for match in _ANSWER.findall(text) if match.strip()]
    if found:
        return clean_fragment(found[-1])
    boxed = last_boxed(text)
    if boxed is not None:
        return clean_fragment(boxed)
    lines = [line for line in text.splitlines() if line.strip()]
    return clean_fragment(lines[-1]) if lines else ""


def parse_number(text):
    """Last numeric literal of a fragment, or None.

    Commas, dollar and percent signs are ignored; a simple fraction ``a/b`` or
    ``\\frac{a}{b}`` counts as one literal and is converted to its value.
    """
    cleaned = str(text).replace(",", "").replace("$", "").replace("\\%", "").replace("%", "")
    found = _NUMBER.findall(cleaned)
    if not found:
        return None
    frac_num, frac_den, plain, denominator = found[-1]
    try:
        if frac_num:
            value = float(frac_num) / float(frac_den)
        elif denominator:
            value = float(plain) / float(denominator)
        else:
            value = float(plain)
    except (ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def last_json(text, kind=dict):
    """The JSON block of the given kind (dict or list) that ends last in the text; the outermost one on ties."""
    text = str(text)
    opening, closing = ("{", "}") if kind is dict else ("[", "]")
    starts = [i for i, ch in enumerate(text) if ch == opening][-400:]
    best = None
    for start in starts:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == opening:
                depth += 1
            elif text[j] == closing:
                depth -= 1
                if depth == 0:
                    if best is None or j > best[0] or (j == best[0] and start < best[1]):
                        try:
                            value = json.loads(text[start:j + 1])
                        except ValueError:
                            value = None
                        if isinstance(value, kind):
                            best = (j, start, value)
                    break
    return best[2] if best else None


def numbers_close(prediction, truth, rel=1e-6, abs_tol=1e-6):
    if prediction is None or truth is None:
        return False
    try:
        return abs(float(prediction) - float(truth)) <= max(abs_tol, rel * abs(float(truth)))
    except OverflowError:  # an integer too large for a float is not close to any reference value
        return False


def normalize_label(text):
    return " ".join(str(text).casefold().replace("_", " ").split()).strip(" .!?:;")
