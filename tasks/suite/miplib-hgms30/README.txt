# MIPLIB 2017 open instance hgms30

**Task id:** `miplib-hgms30`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance hgms30 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Generator maintenance scheduling model for hydropower systems, submitted by Jesus Rodriguez (a co-author of the MILP formulations paper on this problem).

## Why it matters

**Problem:** Schedule the maintenance of the generators of a hydropower system, as in the MILP model hgms30, at minimum cost.

Maintenance prevents costly generator breakdowns but takes generators out of service, so schedules affect power output and cost; the hydropower setting makes the MILP large.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [MILP Formulations for Generator Maintenance Scheduling in Hydropower Systems](https://doi.org/10.1109/tpwrs.2018.2833061): Propose and compare MILP formulations for scheduling generator maintenance in hydropower systems. The task is an instance of a maintenance scheduling MILP for hydropower generators submitted by the paper's first author; which exact formulation and data the instance uses is not stated on the page.

## Record

Listed objective -44338.74480308597 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists -44338.74480308597; published solution file header =obj= -44338.74480308597; the published solution, checked by our verifier: feasible within 1e-6, objective -44338.74480308597.

## Data and requirements

https://miplib.zib.de/WebData/instances/hgms30.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/miplib/hgms30.mps.gz`. Run `python3 scripts/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

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
