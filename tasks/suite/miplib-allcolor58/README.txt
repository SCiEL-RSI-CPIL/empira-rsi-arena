# MIPLIB 2017 open instance allcolor58

**Task id:** `miplib-allcolor58`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance allcolor58 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Prepack optimization model (assortment packing for retail), submitted by Domenico Salvagnin, who co-authored the prepack MILP-heuristics paper.

## Why it matters

**Problem:** Solve the prepack optimization problem instance allcolor58: choose prepack configurations of items so that retail assortments are met at least cost.

Prepacks are fixed bundles of items shipped to retail stores; choosing them well reduces shipping and handling effort.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Mixed-integer linear programming heuristics for the prepack optimization problem](https://doi.org/10.1016/j.disopt.2015.03.004): Design MILP-based heuristics for the prepack optimization problem. The page calls the instance a prepack optimization model submitted by a co-author; whether it is an instance used in the paper is not stated, and the paper's objective is heuristics for this problem.

## Record

Listed objective 42.0 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists 42.0; published solution file header =obj= 42.0; the published solution, checked by our verifier: feasible within 1e-6, objective 42.0.

## Data and requirements

https://miplib.zib.de/WebData/instances/allcolor58.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/miplib/allcolor58.mps.gz`. Run `python3 scripts/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The objective listed on the open-instances page may have been improved since 2026-11-11 (the list did not change between 2026-11-02 and 2026-11-11).
- Candidates may call HiGHS through highspy; a stronger solver is a method, not a discovery by the workflow.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
