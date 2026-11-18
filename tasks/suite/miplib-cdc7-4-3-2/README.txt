# MIPLIB 2017 open instance cdc7-4-3-2

**Task id:** `miplib-cdc7-4-3-2`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization, mathematics

## Task

Find a feasible solution of the MIPLIB 2017 instance cdc7-4-3-2 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Search for a large constant-dimension subspace code in GF(2)^7 (dimension 3, minimum distance 4), the open case related to a binary q-analog of the Fano plane, submitted by Sascha Kurz.

## Why it matters

**Problem:** Find a maximum set of 3-dimensional subspaces of GF(2)^7 whose pairwise intersections have dimension at most 1 (a constant-dimension code with parameters n=7, d=4, k=3, q=2), in the integer programming model cdc7-4-3-2.

Subspace codes are used in random network coding; A_2(7,4;3) is only known to lie between 333 and 381, and a code of size 381 would be a binary q-analog of the Fano plane, whose existence is a well known open question.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [A subspace code of size $333$ in the setting of a binary $q$-analog of the Fano plane](https://arxiv.org/abs/1708.06224): Construct a binary constant-dimension subspace code of length 7, dimension 3 and minimum distance 4 with 333 words, improving the previous lower bound of 329. The task is the integer programming (clique) model of the same code problem; a feasible solution larger than 333 would improve the bound reported in the paper.

## Record

Listed objective -333.0 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists -333.0; published solution file header =obj= -333.0; the published solution, checked by our verifier: feasible within 1e-6, objective -333.0.

## Data and requirements

https://miplib.zib.de/WebData/instances/cdc7-4-3-2.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/miplib/cdc7-4-3-2.mps.gz`. Run `python3 scripts/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The objective listed on the open-instances page may have been improved since 2026-11-11 (the list did not change between 2026-11-02 and 2026-11-11).
- Candidates may call HiGHS through highspy; a stronger solver is a method, not a discovery by the workflow.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
