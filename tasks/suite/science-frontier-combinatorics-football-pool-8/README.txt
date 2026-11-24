# Football pool problem for 8 matches: ternary covering code K_3(8,1)

**Source collection:** [Open combinatorial design, coding and assignment problems with published records: source-based research tasks](../../papers/science-frontier_combinatorics.md)\
**Task id:** `science-frontier-combinatorics-football-pool-8`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Choose as few words of length 8 over the alphabet {0,1,2} as possible so that every one of the 6561 words differs from some chosen word in at most one position. The best known code has 486 words (van Laarhoven, Aarts, van Lint and Wille, 1989) and the best lower bound is 403.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct(q, n, radius) -> list of codewords; the evaluator passes this task's alphabet size q, length n and covering radius. Each codeword is a list of n integers in 0..q-1 and codewords must be distinct. Valid when every word of {0..q-1}^n is within Hamming distance radius of some codeword (checked over the whole space). Score: the number of codewords, minimized; it is an upper bound on K_q(n, radius).

## State of the art

K_3(8,1) <= 486 by the simulated-annealing code of van Laarhoven, Aarts, van Lint and Wille (1989), still the best known in G. Keri's tables and in the 2026 post-Keri table of the covering-codes-lean project; the best lower bound is 403 (D. Gijswijt, S. Polak, 2025).

Reference value: best known value `486` (reviewed 2026-10-12).

- [New upper bounds for the football pool problem for 6, 7, and 8 matches](https://doi.org/10.1016/0097-3165(89)90036-8), P. J. M. van Laarhoven, E. H. L. Aarts, J. H. van Lint, L. T. Wille (1989)
- [Semidefinite lower bounds for covering codes](https://arxiv.org/abs/2504.01932), D. Gijswijt, S. Polak (2025)
- [Tables for bounds on covering codes](https://old.sztaki.hu/~keri/codes/index.htm), G. Keri (2011)

Parameters: `{"q": 3, "n": 8, "radius": 1, "published": 486}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Determine K_3(8,1), the least number of ternary words of length 8 such that every ternary word differs from one of them in at most one position (the football pool problem for 8 matches).

Why it matters: Covering codes are a basic object of coding theory (covering radius, data compression, write-once memories, lossy coding). The football pool problem is the classic open case: exact values are known only up to five matches and for 13 matches, where the perfect ternary Hamming code gives K_3(13,1) = 3^10 = 59049, and for six to eight matches the gaps between the 1989 constructions and the lower bounds are still open.

- [New upper bounds for the football pool problem for 6, 7, and 8 matches](https://doi.org/10.1016/0097-3165(89)90036-8): Main research objective: Improve the upper bounds on the football pool problem K_3(n,1) for n = 6, 7 and 8 by finding covering codes with simulated annealing. Task connection: The task asks for a ternary covering code of length 8 and radius 1 with fewer than 486 words, the size of this paper's code.
- [Semidefinite lower bounds for covering codes](https://arxiv.org/abs/2504.01932): Main research objective: Derive semidefinite-programming lower bounds on the sizes of covering codes, including ternary codes of covering radius one. Task connection: Supplies the lower bound K_3(8,1) >= 403, so the optimum lies between 403 and 486.

Both papers have the football pool problem as their main objective, and the evaluator checks exactly the radius-one covering property over all ternary words of length n.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Independent exact verifier: the union of the Hamming balls is built over all q^n words. The published record construction for every task was scored with this evaluator and reproduces the published value exactly (validation/human-results/frontier_combinatorics.json). The record codes of van Laarhoven, Aarts, van Lint and Wille (1989) were read from the kernel-checked Lean transcription in the covering-codes-lean repository (BSD-3-Clause). Unit tests check perfect Hamming codes and reject codes that leave a word uncovered.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://doi.org/10.1016/0097-3165(89)90036-8 (van Laarhoven, Aarts, van Lint and Wille (1989): simulated-annealing codes giving K_3(6,1) <= 73, K_3(7,1) <= 186 and K_3(8,1) <= 486, printed in the paper.)
- paper: https://arxiv.org/abs/2504.01932 (D. Gijswijt, S. Polak (2025), Semidefinite lower bounds for covering codes: lower bound K_3(8,1) >= 403.)
- repository: https://github.com/florath/covering-codes-lean/blob/460df105545c2d6b04ba71f29de6b56dbda92825/CoveringCodes/Database/Sources/VanLaarhoven1989.lean (Kernel-checked Lean transcription (BSD-3-Clause) of the three published codes; the post-Keri bound table of the same repository lists the current bounds.)
- benchmark: https://old.sztaki.hu/~keri/codes/index.htm (G. Keri's tables of bounds on covering codes: K_3(8,1) between the lower bound of its time and 486.)

## Limitations

- Covering radius 1 over the ternary alphabet only; other alphabets and radii are separate problems.
- The tetracode-based seed has 729 words.
- The record construction is public, so a candidate can reproduce the record by embedding it; the score measures the returned object, not the search that produced it, and only a strictly smaller score than the best known value is a new result.
- Candidate Python runs inside the evaluator process as trusted research code; the evaluator recomputes every property of the returned object exactly and trusts nothing the candidate reports.
