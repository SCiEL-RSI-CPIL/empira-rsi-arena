# Twelve rational points with few ordinary lines

**Source collection:** [Problem-bank constructions: discrete geometry and analysis: source-based research tasks](../../papers/science-prize_g.md)\
**Task id:** `science-ordinary-lines-twelve-rational-points`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

construct() -> exactly 12 distinct integer points [x, y] with |x|, |y| <= 10^9, not all collinear. Minimize the number of lines containing exactly two of the points (exact). Over the reals the minimum for 12 points is 6 and integer coordinates reach it, so the optimum is 6.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> exactly 12 distinct integer points [x, y] with |x|, |y| <= 10^9, not all collinear. Minimize the number of lines containing exactly two of the points (exact).

## State of the art

The minimum is 6. Twelve points that are not all collinear determine at least 6 ordinary lines (OEIS A003034; the Kelly-Moser bound 3n/7 already gives 6), and a Böröczky configuration (six points of a conic and six points of a line) has a copy with integer coordinates and exactly 6 ordinary lines, which the package's tests verify with the evaluator.

Reference value: proven optimum `6` (reviewed 2026-11-24).

Underlying problem: [Erdős Problem #210](https://www.erdosproblems.com/210), status: proved.

- [OEIS A003034](https://oeis.org/A003034)
- [On Sets Defining Few Ordinary Lines](https://doi.org/10.1007/s00454-013-9518-9), Ben Green, Terence Tao (2013)

Parameters: `{"n": 12, "bound": 1000000000}`

## Labels

Scientific contribution: `rediscovery` (tier 2); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Determine the minimum number of ordinary lines of n non-collinear points.

Why it matters: The Sylvester-Gallai and Dirac-Motzkin problems (Erdős Problem #210) are central in incidence geometry; the extremal Böröczky configurations come from regular polygons, and the 12-point one has a projective copy with rational coordinates.

- [On Sets Defining Few Ordinary Lines](https://doi.org/10.1007/s00454-013-9518-9): Main research objective: Determine the minimum number of ordinary lines of n non-collinear points. Task connection: Over the reals the minimum for 12 points is 6 and integer coordinates reach it, so the optimum is 6.

Primary research directly studies this mathematical objective; the finite scope, reference value and rediscovery status are explicit.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Over the reals the minimum for 12 points is 6 and integer coordinates reach it, so the optimum is 6. Exact local certificates are not Lean proofs; floating-point scores are stated as such. Consult the independent oracle and malformed-input tests.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Sources

- paper: https://doi.org/10.1007/s00454-013-9518-9 (Ben Green and Terence Tao, On Sets Defining Few Ordinary Lines (2013). Crossref record inspected; OEIS A003034 inspected: the minimal number of ordinary lines of n real points is 6 for n = 12; Erdős Problem #210 page inspected (proved; Green and Tao determine the minimum for large n). Over the reals the minimum for 12 points is 6 and integer coordinates reach it, so the optimum is 6.)
- archive: https://www.erdosproblems.com/210 (Erdős Problem #210 (proved); statement, history and current bounds reviewed 2026-09-24.)
- benchmark: https://github.com/TheJustinSunPrize/awards/blob/bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2/problems/catalog-0101-0200.md#JSP-000194 (JSP-000194; pinned discovery source, independently paraphrased. Primary mathematical attribution follows the papers above.)

## Limitations

- Solved finite case: the real minimum 6 is attained with integer coordinates.
- Independent Python implementation; no paper text, upstream evaluator, Lean proof or prize-bank content is redistributed. Candidate Python executes as trusted research code.
