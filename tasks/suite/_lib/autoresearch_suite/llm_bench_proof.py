"""IMO-ProofBench proof writing, graded by a judge model in the manner of IMO-Bench's ProofAutoGrader.

Contract: ``solve(question, llm) -> proof``. The judge receives the problem,
the reference solution, the problem's grading guidelines and the proof, with
a rubric paraphrasing ProofAutoGrader's (7 complete and rigorous, 6 almost
correct with minor errors, 1 substantial progress named in the guidelines,
0 otherwise) and answers with ``<points>N out of 7</points>``; as in
ProofAutoGrader, the first such block is read (if the judge omits the tags,
its last "N out of 7" phrase is used and the item is marked ``parsed: phrase``). The item score is points / 7,
so the mean is IMO-ProofBench's "percentage of total points".
"""
from __future__ import annotations

import re

from . import llm_bench_core as core

GRADER = """You are an expert grader for olympiad mathematics who evaluates a proposed proof strictly: only complete, logically sound and precise arguments earn full credit.

Scoring (0 to 7 points):
- 7 points: complete, correct and rigorous. Discarded failed attempts do not cost points if a complete proof is eventually given.
- 6 points: the core argument is sound but there are minor calculation errors or small gaps. Missing proofs of major steps or unjustified claims do not qualify.
- 1 point: substantial progress of the kind named in the grading guidelines. Restating the problem or making easy observations does not qualify.
- 0 points: no substantial progress, or a fundamentally flawed argument.
The problem-specific grading guidelines take precedence over this general scale, in particular for partial credit.

Work through the reference solution and the guidelines, check every step of the proposed proof (beware of arguments that only pretend to be correct), and decide the score.

PROBLEM:
{problem}

REFERENCE SOLUTION (correct, one possible approach):
{solution}

GRADING GUIDELINES:
{guidelines}

PROPOSED PROOF:
{proof}

Explain your assessment, then give the score exactly once in the form <points>N out of 7</points>."""


def parse_points(reply):
    """(points, how) from the first <points>N out of 7</points> block, else from the last 'N out of 7' phrase."""
    match = re.search(r"<points>\s*(\d+)\s*out of 7\s*</points>", str(reply))
    how = "tag"
    if not match:
        found = list(re.finditer(r"\b(\d+)\s*(?:points?\s*)?out of 7\b", str(reply)))
        match, how = (found[-1], "phrase") if found else (None, None)
    if not match or not 0 <= int(match.group(1)) <= 7:
        return None, None
    return int(match.group(1)), how


def grade(judge, row, proof, max_tokens=8192):
    if not proof.strip():
        return {"score": 0.0, "points": 0, "parsed": True}
    prompt = GRADER.format(problem=row["problem"], solution=row["solution"], guidelines=row["guidelines"], proof=proof[:60000])
    points, how = parse_points(judge(prompt, max_tokens=max_tokens))
    return {"score": (points or 0) / 7.0, "points": points, "parsed": how or False}


def items(P):
    rows = core.read_json("proofbench.json")
    return [{"id": f"proofbench-{row['id']}", "question": row["problem"].strip() + "\n\nWrite a complete and rigorous proof.", "row": row} for row in rows]


def evaluate(solve, P):
    core.require_signature(solve, 2, "solve(question, llm)")
    sample = core.fixed_sample(items(P), core.sample_size(P, default=20), P.get("seed", 0))
    client, judge = core.task_llm(P), core.judge_llm(P)
    out = core.run_items(sample, lambda item: core.checked_answer(solve(item["question"], client), "proof"),
                         lambda item, proof: grade(judge, item["row"], proof, int(P.get("judge_max_tokens", 8192))), client, judge=judge)
    records = out["details"]["per_item"]
    out["details"].update(benchmark="imo_proofbench", pass_at_7=sum(1 for r in records if r.get("points") == 7) / len(records),
                          unparsed_grades=sum(1 for r in records if r.get("parsed") is False),
                          untagged_grades=sum(1 for r in records if r.get("parsed") == "phrase"))
    return out
