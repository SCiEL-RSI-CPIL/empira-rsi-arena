# MIPLIB 2017 open instance zeil

**Task id:** `miplib-zeil`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance zeil with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Model computing a slight shift of planned train departure times for a small part of the German railway network so that the maximum 15-minute average power consumption is minimal, submitted by Andreas Baermann.

## Why it matters

**Problem:** Shift the planned departure times of trains in a small part of the German railway network slightly so that the maximum power consumption averaged over 15-minute intervals is minimal.

Trains are time-flexible consumers of the railway power supply; limiting simultaneous departures reduces peak load and can stabilise the supply.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Efficient Formulations and Decomposition Approaches for Power Peak Reduction in Railway Traffic via Timetabling](https://doi.org/10.1287/trsc.2020.1021): Derive efficient formulations and decomposition approaches for optimal timetable adjustment that limits peak power consumption. The instance is a MILP of this timetable adjustment problem on a small German network; whether it appears among the paper's test instances is not stated on the page.

## Record

Listed objective 1081.358500000008 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists 1081.358500000008; published solution file header =obj= 1081.358500000008; the published solution, checked by our verifier: feasible within 1e-6, objective 1081.358500000008.

## Data and requirements

https://miplib.zib.de/WebData/instances/zeil.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/miplib/zeil.mps.gz`. Run `python3 scripts/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

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
