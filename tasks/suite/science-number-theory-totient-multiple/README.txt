# Small integer whose totient is divisible by 120

**Source collection:** [Problem-bank constructions: arithmetic and extremal sets: source-based research tasks](../../papers/science-prize_b.md)\
**Task id:** `science-number-theory-totient-multiple`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Return integer n in [1,1000000] with phi(n) divisible by 120. Minimize n; the evaluator independently factors n and computes the exact totient. Constructs m_n from the source by fixing its divisor parameter to 120; does not compare asymptotic ratios with least primes in progressions.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> candidate. Return integer n in [1,1000000] with phi(n) divisible by 120. Minimize n; the evaluator independently factors n and computes the exact totient.

## State of the art

The least m with phi(m) divisible by 120 is 143 = 11 * 13, with phi(143) = 120 (OEIS A061026, a(120) in the b-file).

Reference value: proven optimum `143` (reviewed 2026-11-20).

- [Old and New Problems and Results in Combinatorial Number Theory](https://mathweb.ucsd.edu/~ronspubs/80_11_number_theory.pdf), Paul Erdős and Ronald L. Graham (1980)
- [OEIS A061026: Smallest number m such that phi(m) is divisible by n, where phi = Euler totient function A000010](https://oeis.org/A061026)

Parameters: `{"modulus": 120, "maximum": 1000000}`

## Labels

Scientific contribution: `rediscovery` (tier 2); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


The original mathematical question is verified in the cited primary research monograph/survey, but no dedicated paper with this exact problem as a primary research goal was independently established in this review. Conservative metadata does not equate a prize listing or historical open status with important_to_science.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Constructs m_n from the source by fixing its divisor parameter to 120; does not compare asymptotic ratios with least primes in progressions.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Sources

- paper: https://mathweb.ucsd.edu/~ronspubs/80_11_number_theory.pdf (Paul Erdős and Ronald L. Graham (1980), Old and New Problems and Results in Combinatorial Number Theory. Printed p. 91. Constructs m_n from the source by fixing its divisor parameter to 120; does not compare asymptotic ratios with least primes in progressions. Primary full text downloaded and inspected; the scanned monograph was OCRed locally and equation-sensitive passages checked against page images. No contemporary open/solved status is inferred from this historical publication.)
- benchmark: https://github.com/TheJustinSunPrize/awards/blob/bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2/problems/catalog-0301-0400.md#JSP-000370 (JSP-000370; pinned Justin Sun Prize bank revision bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2. Discovery source only; mathematical authorship remains with the cited researchers. Independent paraphrase and implementation.)

## Limitations

- Independent finite adaptation, not a Lean proof, prize submission, full solution, optimality proof or claim of mathematical novelty.
- Published source supplies the mathematical question; all verifier and starter code was written locally, with no upstream implementation imported.
- Finite bounds and the public starter are development instances. An accepted candidate establishes only the reported local feasible value; no asymptotic inference follows.
- Candidate Python code is trusted; this runner is not a malicious-code sandbox.
- Constructs m_n from the source by fixing its divisor parameter to 120; does not compare asymptotic ratios with least primes in progressions.
