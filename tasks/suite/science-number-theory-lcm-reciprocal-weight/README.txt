# Large reciprocal weight under lcm separation

**Source collection:** [Problem-bank constructions: arithmetic and extremal sets: source-based research tasks](../../papers/science-prize_b.md)\
**Task id:** `science-number-theory-lcm-reciprocal-weight`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Return nonempty subset of [2,256] with lcm(a,b)>256 for all distinct pairs. Maximize exact reciprocal sum, not cardinality. Uses the reciprocal-sum objective in section 14(1), with n fixed; pairwise lcm threshold is strict and the score is rational. The unit denominator 1 is excluded locally to remove its isolated trivial certificate.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> candidate. Return nonempty subset of [2,256] with lcm(a,b)>256 for all distinct pairs. Maximize exact reciprocal sum, not cardinality.

## State of the art

The maximum of Σ 1/a over subsets of [2,256] with all pairwise least common multiples above 256 is 0.969774168765093 to double precision, attained by a set of 85 integers (exact maximum-weight independent set search with the integer weights lcm(2, ..., 256)/a, in agreement with an integer program). Schinzel and Szekeres proved Σ 1/a ≤ 31/30 for every n (Erdős 1973, p. 135).

Reference value: proven optimum `0.969774168765093` (reviewed 2026-11-24).

- [Problems and results on combinatorial number theory](https://users.renyi.hu/~p_erdos/1973-21.pdf), Paul Erdős (1973)

Parameters: `{"n": 256}`

## Labels

Scientific contribution: `rediscovery` (tier 2); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


The original mathematical question is verified in the cited primary research monograph/survey, but no dedicated paper with this exact problem as a primary research goal was independently established in this review. Conservative metadata does not equate a prize listing or historical open status with important_to_science.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Uses the reciprocal-sum objective in section 14(1), with n fixed; pairwise lcm threshold is strict and the score is rational. The unit denominator 1 is excluded locally to remove its isolated trivial certificate.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Sources

- paper: https://users.renyi.hu/~p_erdos/1973-21.pdf (Paul Erdős (1973), Problems and results on combinatorial number theory. Printed p. 134. Uses the reciprocal-sum objective in section 14(1), with n fixed; pairwise lcm threshold is strict and the score is rational. The unit denominator 1 is excluded locally to remove its isolated trivial certificate. Primary full PDF downloaded and its extracted full text inspected. No contemporary open/solved status is inferred from this historical publication.)
- benchmark: https://github.com/TheJustinSunPrize/awards/blob/bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2/problems/catalog-0401-0500.md#JSP-000433 (JSP-000433; pinned Justin Sun Prize bank revision bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2. Discovery source only; mathematical authorship remains with the cited researchers. Independent paraphrase and implementation.)

## Limitations

- Independent finite adaptation, not a Lean proof, prize submission, full solution, optimality proof or claim of mathematical novelty.
- Published source supplies the mathematical question; all verifier and starter code was written locally, with no upstream implementation imported.
- Finite bounds and the public starter are development instances. An accepted candidate establishes only the reported local feasible value; no asymptotic inference follows.
- Candidate Python code is trusted; this runner is not a malicious-code sandbox.
- Uses the reciprocal-sum objective in section 14(1), with n fixed; pairwise lcm threshold is strict and the score is rational. The unit denominator 1 is excluded locally to remove its isolated trivial certificate.
- Feasibility and rational objective are computed exactly; the standard score envelope converts the final rational to binary64, so exact numerator/denominator details should be used to resolve nearly tied scores.
