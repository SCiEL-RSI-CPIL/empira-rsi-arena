# MIPLIB 2017 open instance r4l4-02-tree-bounds-50

**Task id:** `miplib-r4l4-02-tree-bounds-50`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance r4l4-02-tree-bounds-50 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Periodic event scheduling (PESP) model for public transport timetable optimisation, submitted by Christian Liebchen, with data from LinTim and PESPlib and variables as in Sec. 9 of Liebchen (2006).

## Why it matters

**Problem:** Find a periodic timetable of minimum weighted slack for the PESP instance modelled in r4l4-02-tree-bounds-50 (public transport network R4L4 from PESPlib).

PESP underlies periodic timetabling of railway and bus networks; PESPlib instances are large and remain unsolved to optimality.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [A Mathematical Model for Periodic Scheduling Problems](https://doi.org/10.1137/0402049): Introduce the periodic event scheduling model for periodic scheduling problems. The task is a MIP formulation (Liebchen 2006) of a PESP instance derived from the LinTim and PESPlib data; the objective is the PESP slack cost, not the original paper's setting.

## Record

Listed objective 499061405.0 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists 499061405.0; published solution file header =obj= 499061405.0; the published solution, checked by our verifier: feasible within 1e-6, objective 499061405.0.

## Data and requirements

https://miplib.zib.de/WebData/instances/r4l4-02-tree-bounds-50.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/miplib/r4l4-02-tree-bounds-50.mps.gz`. Run `python3 scripts/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The objective listed on the open-instances page may have been improved since 2026-11-11 (the list did not change between 2026-11-02 and 2026-11-11).
- Candidates may call HiGHS through highspy; a stronger solver is a method, not a discovery by the workflow.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
