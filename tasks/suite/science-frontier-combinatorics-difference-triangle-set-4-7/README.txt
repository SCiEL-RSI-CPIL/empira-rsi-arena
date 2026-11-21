# Difference triangle set (4,7)

**Source collection:** [Open combinatorial design, coding and assignment problems with published records: source-based research tasks](../../papers/science-frontier_combinatorics.md)\
**Task id:** `science-frontier-combinatorics-difference-triangle-set-4-7`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Choose 4 rows of 8 increasing integers starting at 0 so that all differences between two entries of the same row are distinct over all rows, with the largest entry as small as possible. The best known set has largest entry 134 (Shehadeh, Kingsford and Kschischang, 2025); the simple counting bound is 112.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct(n_rows, k) -> list of n_rows rows; the evaluator passes this task's n_rows and k. Each row is a list of k + 1 integers that starts at 0 and increases strictly. Valid when all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows. Score: the scope, the largest entry, minimized; it is an upper bound on m(n_rows, k).

## State of the art

The best known (4,7) difference triangle set has scope 134, found by a search implemented in FPGA hardware (Shehadeh, Kingsford and Kschischang, 2025); the previous best was 135. Optimality is not claimed for it; the counting bound is 112.

Reference value: best known value `134` (reviewed 2026-10-12).

- [New difference triangle sets by a field-programmable gate array-based search technique](https://doi.org/10.1002/jcd.22009), M. Shehadeh, W. Kingsford, F. R. Kschischang (2025)
- [Bounds and construction for difference triangle sets](https://doi.org/10.1109/18.32165), T. Klove (1989)

Parameters: `{"rows": 4, "k": 7, "published": 134}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Determine m(4,7), the least scope of a difference triangle set with 4 rows of 8 marks whose within-row differences are all distinct.

Why it matters: Difference triangle sets yield convolutional self-orthogonal codes for threshold decoding and are used in radar and communication waveform design and in higher-order staircase codes; their minimal scopes are known only for small parameters, and a smaller scope gives shorter code memory and lower decoding complexity.

- [New difference triangle sets by a field-programmable gate array-based search technique](https://doi.org/10.1002/jcd.22009): Main research objective: Find difference triangle sets with smaller scopes than the best known ones, using a search implemented in FPGA hardware. Task connection: The task asks for a (4,7) difference triangle set with scope below 134, the record of this paper.
- [Bounds and construction for difference triangle sets](https://doi.org/10.1109/18.32165): Main research objective: Give lower bounds and constructions for difference triangle sets, motivated by convolutional self-orthogonal codes. Task connection: Defines the minimal-scope problem and its bounds that the task measures against.

Both papers have minimal-scope difference triangle sets as their main objective, and the evaluator checks exactly the distinct-difference property and scores the scope.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Independent exact verifier of all within-row differences. The published record construction for every task was scored with this evaluator and reproduces the published value exactly (validation/human-results/frontier_combinatorics.json). The record sets of Shehadeh, Kingsford and Kschischang (2025) were read from their GitHub repository (MIT). Unit tests check optimal Golomb rulers (one-row sets) and reject repeated differences.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://doi.org/10.1002/jcd.22009 (Shehadeh, Kingsford and Kschischang, J. Combin. Des. 34(1) (2026) 37-50 (CC BY 4.0; arXiv:2502.19517), Table I: m(4,7) <= 134, previous best 135; only the (14,4) and (15,4) sets are claimed optimal.)
- repository: https://github.com/applecoffeecake/dts-search-hdl/blob/2ccff8e767c3fe6390fd3af056dfda11847011fc/new-dts-db-fpga/4-7-134-523.txt (The (4,7) set of scope 134 found by the FPGA search (MIT).)
- paper: https://doi.org/10.1109/18.32165 (Klove (1989), Bounds and construction for difference triangle sets: the problem, its use for convolutional self-orthogonal codes and lower bounds.)

## Limitations

- Differs from the catalog's Sidon-set tasks: those ask for one or two Sidon sets in a fixed interval, whereas this asks for 4 rulers with jointly distinct differences and minimizes the largest mark.
- Only the counting bound n k (k+1) / 2 = 112 is reported by the evaluator; stronger linear-programming lower bounds exist in the literature.
- The greedy seed has scope 358.
- The record construction is public, so a candidate can reproduce the record by embedding it; the score measures the returned object, not the search that produced it, and only a strictly smaller score than the best known value is a new result.
- Candidate Python runs inside the evaluator process as trusted research code; the evaluator recomputes every property of the returned object exactly and trusts nothing the candidate reports.
