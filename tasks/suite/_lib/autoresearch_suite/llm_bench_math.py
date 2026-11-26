"""Final-answer equivalence for research-level math answers (ArXivMath), using Math-Verify.

The reply's answer is the content of its last ``\\boxed{...}`` (else the final
answer text); it is parsed as LaTeX and compared with the gold LaTeX answer by
Math-Verify's symbolic and numeric equivalence (``math_verify`` 0.9.0,
Apache-2.0). Math-Verify bounds each parse and comparison with a SIGALRM
timer; the evaluator's own deadline alarm is suspended around those calls and
re-armed afterwards, so neither timer cancels the other.
"""
from __future__ import annotations

import signal
import time

from . import llm_bench_core as core
from .runtime import CandidateError


def _preserving_alarm(function):
    remaining = signal.alarm(0) if hasattr(signal, "SIGALRM") else 0
    started = time.time()
    try:
        return function()
    finally:
        if remaining:
            signal.alarm(max(1, int(remaining - (time.time() - started) + 0.999)))


def answer_fragment(reply):
    boxed = core.last_boxed(reply)
    return (boxed if boxed is not None else core.final_answer(reply)).strip()


def equivalent(gold, reply, timeout=5):
    """True when Math-Verify judges the reply's final answer equal to the gold LaTeX answer."""
    try:
        from math_verify import parse, verify
    except ImportError as error:
        raise CandidateError(f"math-verify is required: {error}") from None
    fragment = answer_fragment(reply)
    if not fragment:
        return False

    def check():
        gold_parsed = parse("$" + gold + "$", parsing_timeout=timeout)
        predicted = parse("$" + fragment + "$", parsing_timeout=timeout)
        return bool(gold_parsed) and bool(predicted) and verify(gold_parsed, predicted, timeout_seconds=timeout)
    try:
        return bool(_preserving_alarm(check))
    except Exception:  # noqa: BLE001 - unparsable or timed-out comparisons count as wrong
        return False


def arxivmath_items(P):
    rows = core.read_json("arxivmath.json")
    items = [{"id": row["id"], "answer": row["answer"],
              "question": row["problem"].strip() + "\n\nPut your final answer within \\boxed{}."} for row in rows]
    return items, lambda item, reply: float(equivalent(item["answer"], reply))
