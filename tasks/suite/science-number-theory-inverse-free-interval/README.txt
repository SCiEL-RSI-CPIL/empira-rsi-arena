# Long intervals containing no modular inverse pair

**Source collection:** [Problem-bank constructions: arithmetic and extremal sets: source-based research tasks](../../papers/science-prize_b.md)\
**Task id:** `science-number-theory-inverse-free-interval`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Return [start,length] for a nonempty interval contained in [1,1008]. For EVERY x,y in the interval, including x=y, require x*y %1009 !=1. Maximize interval length. Finite obstruction search for the modular inverse-pair interval question. Diagonal pairs are checked, so the self-inverse residues 1 and 1008 cannot be present; this is stricter than the preliminary distinct-pair proposal.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> candidate. Return [start,length] for a nonempty interval contained in [1,1008]. For EVERY x,y in the interval, including x=y, require x*y %1009 !=1. Maximize interval length.

## State of the art

A scan of all intervals in [1,1008] shows that the longest interval containing no x, y (x = y allowed) with x·y ≡ 1 (mod 1009) has length 85: two intervals attain it and none of length 86 exists. Erdős and Graham (1980, p. 89) ask whether every interval of length p^c, c > 1/2, contains such a pair for large p.

Reference value: proven optimum `85` (reviewed 2026-11-24).

- [Old and New Problems and Results in Combinatorial Number Theory](https://mathweb.ucsd.edu/~ronspubs/80_11_number_theory.pdf), Paul Erdős and Ronald L. Graham (1980)

Parameters: `{"p": 1009}`

## Labels

Scientific contribution: `rediscovery` (tier 2); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


The original mathematical question is verified in the cited primary research monograph/survey, but no dedicated paper with this exact problem as a primary research goal was independently established in this review. Conservative metadata does not equate a prize listing or historical open status with important_to_science.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Finite obstruction search for the modular inverse-pair interval question. Diagonal pairs are checked, so the self-inverse residues 1 and 1008 cannot be present; this is stricter than the preliminary distinct-pair proposal.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Sources

- paper: https://mathweb.ucsd.edu/~ronspubs/80_11_number_theory.pdf (Paul Erdős and Ronald L. Graham (1980), Old and New Problems and Results in Combinatorial Number Theory. Printed p. 89. Finite obstruction search for the modular inverse-pair interval question. Diagonal pairs are checked, so the self-inverse residues 1 and 1008 cannot be present; this is stricter than the preliminary distinct-pair proposal. Primary full text downloaded and inspected; the scanned monograph was OCRed locally and equation-sensitive passages checked against page images. No contemporary open/solved status is inferred from this historical publication.)
- benchmark: https://github.com/TheJustinSunPrize/awards/blob/bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2/problems/catalog-0301-0400.md#JSP-000362 (JSP-000362; pinned Justin Sun Prize bank revision bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2. Discovery source only; mathematical authorship remains with the cited researchers. Independent paraphrase and implementation.)

## Limitations

- Independent finite adaptation, not a Lean proof, prize submission, full solution, optimality proof or claim of mathematical novelty.
- Published source supplies the mathematical question; all verifier and starter code was written locally, with no upstream implementation imported.
- Finite bounds and the public starter are development instances. An accepted candidate establishes only the reported local feasible value; no asymptotic inference follows.
- Candidate Python code is trusted; this runner is not a malicious-code sandbox.
- Finite obstruction search for the modular inverse-pair interval question. Diagonal pairs are checked, so the self-inverse residues 1 and 1008 cannot be present; this is stricter than the preliminary distinct-pair proposal.
