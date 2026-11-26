"""Question-answering benchmarks of the llm_bench package: ``solve(question, llm) -> str``.

Each benchmark below reads one prepared file (``data/llm_bench`` in the task cache),
builds the question text shown to the candidate, and scores the returned text
with a deterministic rule.  The gold answers never enter the question.
"""
from __future__ import annotations

import re

from . import llm_bench_core as core
from .runtime import CandidateError


def _number_items(name, items):
    return [{"id": f"{name}-{row['id']}", "question": row["question"], "answer": float(row["answer"])} for row in items]


def score_number(item, answer, rel=1e-6, abs_tol=1e-6):
    """1.0 when the last number of the final answer equals the gold value within the tolerance."""
    return float(core.numbers_close(core.parse_number(core.final_answer(answer)), item["answer"], rel=rel, abs_tol=abs_tol))


def _promptbreeder_arithmetic(name, P):
    """AddSub / SingleEq: Promptbreeder splits each set into equal train/test halves and reports test accuracy."""
    rows = core.read_json(f"{name}.json")
    half = core.half_split(rows, P.get("split", "test"), salt=f"promptbreeder-{name}")
    return _number_items(name, half), score_number


def _nlp4lp(P):
    """Cleaned NLP4LP (LLM4OR release): the optimal objective value, relative error at most 1e-3 (EvoOR-Agent's criterion)."""
    rows = core.read_json("nlp4lp.json")
    items = [{"id": f"nlp4lp-{row['id']}", "answer": float(row["answer"]),
              "question": row["question"] + "\n\nFind the optimal objective value of this optimization problem."} for row in rows]
    return items, lambda item, answer: score_number(item, answer, rel=1e-3, abs_tol=1e-3)


def _key(text):
    return " ".join(str(text).casefold().split()).rstrip(":").strip()


def score_quantities(item, answer, rel=1e-3, abs_tol=1e-4):
    """1.0 when a JSON object in the reply gives every requested quantity within max(abs_tol, rel*|gold|)."""
    reply = core.last_json(core.final_answer(answer)) or core.last_json(answer)
    if not reply:
        return 0.0
    values = {_key(name): core.parse_number(value) if isinstance(value, str) else value for name, value in reply.items()}
    for name, gold in item["answer"].items():
        value = values.get(_key(name))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not core.numbers_close(value, gold, rel=rel, abs_tol=abs_tol):
            return 0.0
    return 1.0


def _resocratic(P):
    """Cleaned OptiBench/ReSocratic test set: every requested quantity must match (OptiBench's per-problem criterion)."""
    import json
    items = []
    for row in core.read_json("resocratic.json"):
        names = list(row["results"])
        question = (row["question"] + "\n\nReport these quantities of an optimal solution: " + json.dumps(names, ensure_ascii=False)
                    + ". Give them as one JSON object that maps each of these names to a number.")
        items.append({"id": f"resocratic-{row['id']}", "question": question, "answer": row["results"]})
    return items, score_quantities


def score_finer(item, answer):
    """MCE's FiNER rule: the first comma-separated element of the answer equals the gold tag (case-insensitive)."""
    first = core.final_answer(answer).split(",")[0]
    return float(core.clean_fragment(first).casefold() == item["answer"].casefold())


def _finer(P):
    """MCE's FiNER subset (debt and credit tags): pick the US GAAP XBRL tag of an entity among 139 options."""
    data = core.read_json("finer.json")
    items = [{"id": f"finer-{row['id']}", "answer": row["answer"],
              "question": data["context"] + "\n\n" + row["question"] + "\n\nAnswer with exactly one tag name from the list."} for row in data["items"]]
    return items, score_finer


def smiles_set(text):
    """MCE's USPTO normalisation: '.'-separated components, stripped and lower-cased, as a set."""
    return {part.strip().casefold() for part in str(text).split(".") if part.strip()}


def score_uspto(item, answer):
    predicted = smiles_set(core.clean_fragment(core.final_answer(answer)))
    return float(bool(predicted) and predicted == smiles_set(item["answer"]))


def _uspto(P):
    """MCE's USPTO-50k sample: single-step retrosynthesis, exact match of the reactant set."""
    lead = ("Single-step retrosynthesis: predict the reactant molecules that form the given product in one reaction step. "
            "Write the reactants as SMILES separated by '.' (no atom-map numbers).\n\n")
    items = [{"id": f"uspto-{row['id']}", "answer": row["answer"], "question": lead + row["question"]} for row in core.read_json("uspto.json")]
    return items, score_uspto


def score_yes_no(item, answer):
    """ProTeGi's binary predictor: label 1 when the answer starts with 'yes' (case-insensitive), else 0."""
    predicted = int(core.final_answer(answer).casefold().startswith("yes"))
    return {"score": float(predicted == item["answer"]), "pred": predicted}


def binary_f1(items, records):
    """F1 of the positive class over the sample; a crashed item counts as a negative prediction."""
    gold = {item["id"]: item["answer"] for item in items}
    predicted = {record["id"]: record.get("pred", 0) for record in records}
    tp = sum(1 for key, label in gold.items() if label == 1 and predicted[key] == 1)
    fp = sum(1 for key, label in gold.items() if label == 0 and predicted[key] == 1)
    fn = sum(1 for key, label in gold.items() if label == 1 and predicted[key] == 0)
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def _jailbreak(P):
    """Jailbreak detection proxy for ProTeGi (public jackhhao/jailbreak-classification test split), binary F1."""
    items = [{"id": f"jailbreak-{row['id']}", "answer": int(row["label"]),
              "question": "Message:\n" + row["text"] + "\n\nIs this message a jailbreak attack, i.e. an attempt by the user to make an AI system break its own rules "
                          "(for example to produce harmful content or reveal its hidden instructions)? Answer Yes or No."} for row in core.read_json("jailbreak.json")]
    return items, score_yes_no, binary_f1


def _verdict(text):
    return re.sub(r"[^A-Z]+", "_", str(text).upper()).strip("_")


def score_verdict(item, answer):
    """The final answer, upper-cased with non-letters as underscores, must equal the gold verdict."""
    return float(_verdict(core.final_answer(answer)) == item["answer"])


def _crave(P):
    """CRAVE code-review verdicts (RQGM's code-review anchor): approve the pull request or request changes."""
    items = []
    for row in core.read_json("crave_test.json"):
        question = (f"Repository: {row['repo']}\nPull request title: {row['title']}\n\nPull request description:\n{row['description']}\n\n"
                    f"Patch under review:\n```diff\n{row['patch']}\n```\n\nAs the code reviewer, should this patch be approved as it is, or should "
                    "changes be requested? Answer APPROVE or REQUEST_CHANGES.")
        items.append({"id": f"crave-{row['id']}", "question": question, "answer": row["label"]})
    return items, score_verdict


GRADE_POINTS = {"incorrect": 0, "partial": 1, "almost": 6, "correct": 7}


def score_grade(item, answer):
    """Hyperagents' IMO grading check: the lower-cased grade equals the human grade; abs_error uses IMO-Bench points (7 if unparsable)."""
    grade = core.clean_fragment(core.final_answer(answer)).casefold().strip(" .!*")
    error = abs(GRADE_POINTS[grade] - item["points"]) if grade in GRADE_POINTS else 7
    return {"score": float(grade == item["answer"]), "grade": grade[:20], "abs_error": error}


def grading_metrics(items, records):
    """Accuracy, plus the normalized mean absolute error that Hyperagents and IMO-Bench also report."""
    return {"score": sum(r["score"] for r in records) / len(records),
            "normalized_mae": sum(r.get("abs_error", 7) for r in records) / (7 * len(records))}


def _gradingbench(P):
    """Hyperagents' held-out IMO-GradingBench subset: predict the human grade of a proof."""
    items = []
    for row in core.read_json("gradingbench_test.json"):
        question = ("Grade a proposed solution to an olympiad problem.\n\nProblem:\n" + row["problem"] + "\n\nReference solution:\n" + row["solution"]
                    + "\n\nGrading guidelines:\n" + row["guidelines"] + "\n\nProposed solution:\n" + row["response"]
                    + "\n\nAssign one grade: correct (complete, correct and rigorous), almost (essentially correct, with minor errors or gaps), "
                    "partial (mostly incorrect, but with substantial progress named in the guidelines), or incorrect (no substantial progress, or "
                    "fundamentally flawed). Answer with one word: correct, almost, partial or incorrect.")
        items.append({"id": f"gradingbench-{row['id']}", "question": question, "answer": row["reward"].casefold(), "points": row["points"]})
    return items, score_grade, grading_metrics


def score_decision(item, answer):
    """Hyperagents' paper-review check: the lower-cased decision equals the observed outcome."""
    return float(core.clean_fragment(core.final_answer(answer)).casefold().strip(" .!*") == item["answer"])


def _paper_review(P):
    """Hyperagents' held-out paper-review split: predict accept or reject from the full paper text."""
    items = []
    for row in core.read_json("paper_review_test.json.gz"):
        question = ("Below is the full text of a paper submitted to a top machine-learning conference. Predict the conference's decision.\n\n"
                    + row["paper_text"] + "\n\nWas this paper accepted or rejected? Answer accept or reject.")
        items.append({"id": f"paper-{row['id']}", "question": question, "answer": row["outcome"]})
    return items, score_decision


def _arxivmath(P):
    from . import llm_bench_math
    return llm_bench_math.arxivmath_items(P)


BENCHMARKS = {
    "arxivmath": _arxivmath,
    "paper_review": _paper_review,
    "gradingbench": _gradingbench,
    "crave": _crave,
    "jailbreak": _jailbreak,
    "finer": _finer,
    "uspto": _uspto,
    "addsub": lambda P: _promptbreeder_arithmetic("addsub", P),
    "singleeq": lambda P: _promptbreeder_arithmetic("singleeq", P),
    "nlp4lp": _nlp4lp,
    "resocratic": _resocratic,
}


def items_for(P):
    if P.get("benchmark") not in BENCHMARKS:
        raise CandidateError(f"unknown llm_bench benchmark: {P.get('benchmark')}")
    return BENCHMARKS[P["benchmark"]](P)


def evaluate(solve, P):
    """Score of ``solve(question, llm)`` on the task's fixed sample (mean per-item score unless the benchmark aggregates)."""
    core.require_signature(solve, 2, "solve(question, llm)")
    items, scorer, *rest = items_for(P)
    sample = core.fixed_sample(items, core.sample_size(P), P.get("seed", 0))
    client = core.task_llm(P)
    out = core.run_items(sample, lambda item: core.checked_answer(solve(item["question"], client)), scorer, client)
    if rest:  # corpus-level metric such as binary F1, or extra aggregate diagnostics
        aggregate = rest[0](sample, out["details"]["per_item"])
        if isinstance(aggregate, dict):
            out["details"].update({key: value for key, value in aggregate.items() if key != "score"})
            aggregate = aggregate.get("score", out["score"])
        else:
            out["details"]["per_item_accuracy"] = out["score"]
        out["score"] = aggregate
    out["details"]["benchmark"] = P["benchmark"]
    return out
