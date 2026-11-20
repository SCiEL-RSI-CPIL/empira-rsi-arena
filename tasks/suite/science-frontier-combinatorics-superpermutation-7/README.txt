# Shortest superpermutation on seven symbols

**Source collection:** [Open combinatorial design, coding and assignment problems with published records: source-based research tasks](../../papers/science-frontier_combinatorics.md)\
**Task id:** `science-frontier-combinatorics-superpermutation-7`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Write the shortest word over the symbols 1 to 7 that contains every one of the 5040 permutations of 1..7 as seven consecutive symbols. The shortest known word has 5905 symbols (A. Teodorescu, August 2026) and the best published lower bound is 5884.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct(n) -> a word over the symbols 1..n, given as a string of the digits 1..n or a list of integers; the evaluator passes this task's n. Valid when every permutation of 1..n occurs as n consecutive symbols (every window is checked). Score: the length of the word, minimized; at most n * n! symbols.

## State of the art

The shortest known superpermutation on 7 symbols has length 5905, posted by A. Teodorescu on 18 August 2026 and found with G. Egan's PermutationChains.c over millions of kernels from KernelFinder.c; it improved 5906 (G. Egan and R. Houston, 2019) and 5907 (B. Coanda, 2019). The best published lower bound is 5884 (Houston, Pantone and Vatter, 2018).

Reference value: best known value `5905` (reviewed 2026-10-12).

- [n=7 superpermutation of length 5905 (superpermutators mailing list)](https://groups.google.com/g/superpermutators/c/D5sKkV2jBuc), A. Teodorescu (2026)
- [Superpermutations](https://www.gregegan.net/SCIENCE/Superpermutations/Superpermutations.html), G. Egan (2019)
- [A lower bound on the length of the shortest superpattern](https://oeis.org/A180632/a180632.pdf), Anonymous 4chan poster, R. Houston, J. Pantone, V. Vatter (2018)

Parameters: `{"n": 7, "published": 5905}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Determine the minimal length of a superpermutation on seven symbols: a word containing each of the 5040 permutations of 1..7 as a contiguous block.

Why it matters: The minimal superpermutation problem is a well-known open problem in combinatorics on words. The conjecture that 1! + 2! + ... + n! is minimal failed in 2014, exact values are known only up to n = 5, and the lower-bound question is tied to Hamiltonian paths in Cayley graphs of the symmetric group; for seven symbols the gap between the lower bound 5884 and the best constructions is open.

- [Tackling the minimal superpermutation problem](https://arxiv.org/abs/1408.5108): Main research objective: Find shorter superpermutations and study the minimal superpermutation problem through travelling-salesman formulations. Task connection: The task asks for a shorter superpermutation on seven symbols than the best known one (5905).
- [Containing all permutations](https://doi.org/10.1080/00029890.2021.1835384): Main research objective: Survey and advance the problems of words and permutations that contain all permutations, including minimal superpermutations. Task connection: Describes the superpermutation problem, its bounds and the constructions this task measures against.

Both papers have the minimal superpermutation problem as their main objective, and the evaluator checks exactly that every permutation of 1..7 occurs as a block of the returned word.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Independent exact verifier over all windows of the word. The published record construction for every task was scored with this evaluator and reproduces the published value exactly (validation/human-results/frontier_combinatorics.json). The seed is the classic recursive construction of length 1! + ... + n!; unit tests check the known minimal lengths 9, 33 and 153 for n = 3, 4, 5.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- archive: https://groups.google.com/g/superpermutators/c/D5sKkV2jBuc (Superpermutators mailing list, 18 August 2026: A. Teodorescu posts four length-5905 superpermutations for n = 7 (two distinct up to symmetry) found with G. Egan's PermutationChains.c and KernelFinder.c; A. Balint and S. Street confirm that they contain all 5040 permutations.)
- archive: https://www.gregegan.net/SCIENCE/Superpermutations/Superpermutations.html (G. Egan's page: the previous record 5906 (Egan and Houston, 27 February 2019), its word, and the search programs.)
- paper: https://arxiv.org/abs/1408.5108 (R. Houston (2014), Tackling the minimal superpermutation problem.)
- paper: https://oeis.org/A180632/a180632.pdf (Anonymous 4chan poster, R. Houston, J. Pantone and V. Vatter (2018): lower bound n! + (n-1)! + (n-2)! + n - 3, which is 5884 for n = 7.)
- benchmark: https://oeis.org/A180632 (OEIS A180632 (minimal superpermutation lengths): a(7) <= 5906 recorded in 2019.)

## Limitations

- The 5905 record was announced on a mailing list and checked there by several members; it has no peer-reviewed publication yet.
- A lower bound of 5898 was also claimed on the same list in August 2026 without review; the published lower bound remains 5884.
- The recursive seed has length 5913 = 1! + 2! + ... + 7!.
- The record construction is public, so a candidate can reproduce the record by embedding it; the score measures the returned object, not the search that produced it, and only a strictly smaller score than the best known value is a new result.
- Candidate Python runs inside the evaluator process as trusted research code; the evaluator recomputes every property of the returned object exactly and trusts nothing the candidate reports.
