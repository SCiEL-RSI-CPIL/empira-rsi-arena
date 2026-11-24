# MIPLIB 2017 open instance t1717

**Task id:** `miplib-t1717`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance t1717 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Vehicle scheduling set partitioning problem from Berlin's Telebus transportation system for handicapped people, submitted by R. Borndoerfer.

## Why it matters

**Problem:** Assign requests to vehicles for the Berlin Telebus dial-a-ride system by solving the set partitioning model t1717.

Telebus serves mobility-impaired people in Berlin; good vehicle schedules save fleet cost and improve service, and the set partitioning model is large and hard.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Telebus Berlin: Vehicle Scheduling in a Dial-a-Ride System](https://doi.org/10.1007/978-3-642-85970-0_19): Describe and solve vehicle scheduling for the Berlin Telebus dial-a-ride system. The task is a set partitioning instance from this application; the paper's exact instance and the formulation used are not confirmed from the page.

## Record

Listed objective 158260.0 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists 158260; published solution file header =obj= 158260.000000003; the published solution, checked by our verifier: feasible within 1e-6, objective 158259.9999999995.

## Data and requirements

https://miplib.zib.de/WebData/instances/t1717.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/arnew/miplib/t1717.mps.gz`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The objective listed on the open-instances page may have been improved since 2026-11-11 (the list did not change between 2026-11-02 and 2026-11-11).
- Candidates may call HiGHS through highspy; a stronger solver is a method, not a discovery by the workflow.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
