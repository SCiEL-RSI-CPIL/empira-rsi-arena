# MIPLIB 2017 open instance gsvm2rl11

**Task id:** `miplib-gsvm2rl11`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization, machine learning

## Task

Find a feasible solution of the MIPLIB 2017 instance gsvm2rl11 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Ramp-loss support vector machine in the GSVM2-RL mixed-integer formulation of Hess and Brooks (2015), submitted by Toni Sorrell.

## Why it matters

**Problem:** Train a ramp-loss support vector machine with L1-norm regularisation by solving the GSVM2-RL mixed-integer linear programme for the training set encoded in gsvm2rl11.

Ramp loss makes SVM classification robust to outliers but is non-convex, so exact training needs mixed-integer programming; these models are hard for current solvers.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [The Support Vector Machine and Mixed Integer Linear Programming: Ramp Loss SVM with L1-Norm Regularization](https://doi.org/10.1287/ics.2015.0017): Formulate ramp-loss SVM with L1-norm regularisation as mixed-integer linear programmes and study their solution. The task is the GSVM2-RL formulation of this paper for one data set; the evaluator scores the MILP objective, not classification accuracy.

## Record

Listed objective 18121.6376441351 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists 18121.6376441351; published solution file header =obj= 18121.6376441351; the published solution, checked by our verifier: feasible within 1e-6, objective 18121.637644135124.

## Data and requirements

https://miplib.zib.de/WebData/instances/gsvm2rl11.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/arnew/miplib/gsvm2rl11.mps.gz`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The objective listed on the open-instances page may have been improved since 2026-11-11 (the list did not change between 2026-11-02 and 2026-11-11).
- Candidates may call HiGHS through highspy; a stronger solver is a method, not a discovery by the workflow.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
