# MIPLIB 2017 open instance shs1042

**Task id:** `miplib-shs1042`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance shs1042 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Joint online truck scheduling and inventory management for multiple warehouses, submitted by Christoph Helmberg and imported from the MIPLIB 2010 submissions.

## Why it matters

**Problem:** Schedule truck transports of pallets between warehouses so that supply is sufficient for known and stochastic demand, as an integer program (instance shs1042).

Warehouse replenishment by truck is a real logistics problem that has to be solved online within minutes; the integer programs are very large (here several hundred thousand variables).

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [A Case Study of Joint Online Truck Scheduling and Inventory Management for Multiple Warehouses](https://doi.org/10.1287/opre.1060.0374): Solve a real-world problem of transporting pallets between warehouses to guarantee supply, via convex relaxation of an integer programming formulation suitable for online optimisation. The MIPLIB instances are integer programs of the same joint scheduling and inventory problem by the same submitter; it was not checked which data snapshot they use.

## Record

Listed objective 11070.701580240244 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists 11070.7015802402; published solution file header =obj= 11070.701580240244; the published solution, checked by our verifier: feasible within 1e-6, objective 11070.701580240177.

## Data and requirements

https://miplib.zib.de/WebData/instances/shs1042.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/arnew/miplib/shs1042.mps.gz`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The objective listed on the open-instances page may have been improved since 2026-11-11 (the list did not change between 2026-11-02 and 2026-11-11).
- Candidates may call HiGHS through highspy; a stronger solver is a method, not a discovery by the workflow.
- The starting program finds no feasible solution of this instance (HiGHS with a 30 s limit, run of 2026-11-11), so a search starts from an invalid candidate.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
