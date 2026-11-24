# MIPLIB 2017 open instance rmine11

**Task id:** `miplib-rmine11`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance rmine11 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Multi-period open-pit mining model over a cube of blocks with two knapsack constraints per period, submitted by Daniel Espinoza to MIPLIB 2010.

## Why it matters

**Problem:** Choose the blocks to extract and the period of extraction in each period of an open-pit mine cube, respecting block precedences and two knapsack (capacity) constraints per period, to maximise the objective of the model rmine11.

Open-pit mine planning decides what to mine and when, and the multi-period model is a large precedence-constrained knapsack problem whose size makes exact solution hard.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Large-scale multi-period precedence constrained knapsack problem: A mining application](https://doi.org/10.1016/j.endm.2010.05.052): Study and solve large-scale multi-period precedence constrained knapsack problems that arise in mine production scheduling. The MIPLIB page describes the instances as open pit mining over a cube with several periods and two knapsack constraints per period; the link to this paper is by problem description and co-authorship of the submitter, not an explicit citation on the page.

## Record

Listed objective -2508.404144 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists -2508.404144; published solution file header =obj= -2508.404144; the published solution, checked by our verifier: feasible within 1e-6, objective -2508.404144.

## Data and requirements

https://miplib.zib.de/WebData/instances/rmine11.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/arnew/miplib/rmine11.mps.gz`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The objective listed on the open-instances page may have been improved since 2026-11-11 (the list did not change between 2026-11-02 and 2026-11-11).
- Candidates may call HiGHS through highspy; a stronger solver is a method, not a discovery by the workflow.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
