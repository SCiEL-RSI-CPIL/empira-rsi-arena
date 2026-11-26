"""IFBench (GEPA's instruction-following task): ``solve(prompt, llm) -> response``.

The response is checked by the pinned IFBench instruction checkers (prepared
by ``scripts/prepare_llm_bench.py`` into ``data/llm_bench/ifbench_checkers``)
with GEPA's loose rule: an instruction counts as followed when the response,
or the response without its first and/or last line and/or without asterisks,
passes the checker.  The item score is the fraction of followed instructions.

The upstream checkers would download NLTK resources and a spaCy model at run
time; the evaluator disables both downloads, serves the NLTK resources from
the prepared, hash-locked zip files and relies on the installed
``en_core_web_sm`` package, so no network access happens during scoring.
"""
from __future__ import annotations

import importlib
import random
import sys

from . import llm_bench_core as core
from .runtime import CandidateError

_REGISTRY = None


def _no_download(*args, **kwargs):
    return True


def registry():
    """The IFBench instruction registry, imported once with network downloads disabled."""
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    core.data_path("ifbench_checkers/instructions_registry.py")
    for name in ("tokenizers/punkt_tab.zip", "corpora/stopwords.zip", "taggers/averaged_perceptron_tagger_eng.zip"):
        core.data_path("nltk_data/" + name)
    try:
        import nltk
        import spacy.cli
        importlib.import_module("en_core_web_sm")
    except ImportError as error:
        raise CandidateError(f"IFBench checkers need nltk, spacy and en_core_web_sm: {error}") from None
    nltk.download = _no_download
    spacy.cli.download = _no_download
    nltk_dir = str(core.DATA / "nltk_data")
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)
    if str(core.DATA) not in sys.path:
        sys.path.insert(0, str(core.DATA))
    _REGISTRY = importlib.import_module("ifbench_checkers.instructions_registry")
    return _REGISTRY


def response_variants(response):
    """GEPA's loose variants: the response, without its first/last line, and without asterisks."""
    lines = response.split("\n")
    first, last, both = "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()
    variants = [response, response.replace("*", ""), first, last, both]
    return variants + [first.replace("*", ""), last.replace("*", ""), both.replace("*", "")]


def followed(row, response):
    """(number of instructions followed, number of instructions, checker errors) for one response."""
    reg = registry()
    variants = response_variants(response)
    count = errors = 0
    for index, instruction_id in enumerate(row["instruction_id_list"]):
        instruction = reg.INSTRUCTION_DICT[instruction_id](instruction_id)
        kwargs = {key: value for key, value in row["kwargs"][index].items() if value is not None}
        random.seed(0)  # some checkers draw defaults at random; keep descriptions deterministic
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=row["prompt"])
        ok = False
        for text in variants:
            if not text.strip():
                continue
            try:
                if instruction.check_following(text):
                    ok = True
                    break
            except Exception:  # noqa: BLE001 - a checker crash on this variant means 'not followed'
                errors += 1
        count += ok
    return count, len(row["instruction_id_list"]), errors


def score_item(item, response):
    count, total, errors = followed(item["row"], response)
    return {"score": count / total, "followed": count, "instructions": total, "checker_errors": errors}


def evaluate(solve, P):
    core.require_signature(solve, 2, "solve(question, llm)")
    rows = core.read_json("ifbench_test.json")
    items = [{"id": f"ifbench-{row['id']}", "question": row["prompt"], "row": row} for row in rows]
    sample = core.fixed_sample(items, core.sample_size(P), P.get("seed", 0))
    registry()
    client = core.task_llm(P)
    out = core.run_items(sample, lambda item: core.checked_answer(solve(item["question"], client), "response"), score_item, client)
    records = out["details"]["per_item"]
    out["details"]["instructions_followed"] = sum(r.get("followed", 0) for r in records)
    out["details"]["instructions_total"] = sum(len(item["row"]["instruction_id_list"]) for item in sample)
    out["details"]["benchmark"] = "ifbench"
    return out
