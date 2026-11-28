# Small largest prime factor of consecutive-integer products

**Source collection:** [Problem-bank constructions: arithmetic and extremal sets: source-based research tasks](../../papers/science-prize_b.md)\
**Task id:** `science-number-theory-smooth-adjacent`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Return integer n in [10000,1000000]. Minimize the largest prime factor of n(n+1), independently computed by exact trial division. Searches a bounded window for smooth consecutive integers. The local lower cutoff prevents the trivial pair 1,2; it proves no asymptotic bound.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> candidate. Return integer n in [10000,1000000]. Minimize the largest prime factor of n(n+1), independently computed by exact trial division.

## State of the art

The minimum of P(n(n+1)) over n in [10000, 1000000] is 13. The largest n for which n(n+1) has no prime factor above 11 is 9800 (Lehmer 1964; OEIS A002072), which lies below the window, and a full scan of the window finds two n with P(n(n+1)) = 13.

Reference value: proven optimum `13` (reviewed 2026-11-24).

- [Old and New Problems and Results in Combinatorial Number Theory](https://mathweb.ucsd.edu/~ronspubs/80_11_number_theory.pdf), Paul Erdős and Ronald L. Graham (1980)
- [OEIS A002072: a(n) = smallest number m such that for all k > m, either k or k+1 has a prime factor > prime(n)](https://oeis.org/A002072)

Parameters: `{"lower": 10000, "upper": 1000000}`

## Labels

Scientific contribution: `rediscovery` (tier 2); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


The original mathematical question is verified in the cited primary research monograph/survey, but no dedicated paper with this exact problem as a primary research goal was independently established in this review. Conservative metadata does not equate a prize listing or historical open status with important_to_science.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Searches a bounded window for smooth consecutive integers. The local lower cutoff prevents the trivial pair 1,2; it proves no asymptotic bound.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Sources

- paper: https://mathweb.ucsd.edu/~ronspubs/80_11_number_theory.pdf (Paul Erdős and Ronald L. Graham (1980), Old and New Problems and Results in Combinatorial Number Theory. Printed p. 69. Searches a bounded window for smooth consecutive integers. The local lower cutoff prevents the trivial pair 1,2; it proves no asymptotic bound. Primary full text downloaded and inspected; the scanned monograph was OCRed locally and equation-sensitive passages checked against page images. No contemporary open/solved status is inferred from this historical publication.)
- benchmark: https://github.com/TheJustinSunPrize/awards/blob/bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2/problems/catalog-0301-0400.md#JSP-000304 (JSP-000304; pinned Justin Sun Prize bank revision bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2. Discovery source only; mathematical authorship remains with the cited researchers. Independent paraphrase and implementation.)

## Limitations

- Independent finite adaptation, not a Lean proof, prize submission, full solution, optimality proof or claim of mathematical novelty.
- Published source supplies the mathematical question; all verifier and starter code was written locally, with no upstream implementation imported.
- Finite bounds and the public starter are development instances. An accepted candidate establishes only the reported local feasible value; no asymptotic inference follows.
- Candidate Python code is trusted; this runner is not a malicious-code sandbox.
- Searches a bounded window for smooth consecutive integers. The local lower cutoff prevents the trivial pair 1,2; it proves no asymptotic bound.
