# MIPLIB 2017 open instance dws012-01

**Task id:** `miplib-dws012-01`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance dws012-01 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Design of a decentralised drinking-water supply system for a skyscraper (pump layout and continuous pump speeds over load scenarios, pump curves modelled by an aggregated convex combination), submitted by Philipp Leise; the instances differ in the number of floors and load scenarios.

## Why it matters

**Problem:** Design a decentralised drinking-water supply system for a skyscraper (pump layout and pump operation per load scenario) with minimum cost, with the pump characteristics linearised by an aggregated convex combination (instance dws012-01).

Pumping water in high-rise buildings consumes a lot of energy, and the pump system is designed once for many load scenarios; the MILP size grows with floors and scenarios.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Energy-Efficient Design of a Water Supply System for Skyscrapers by Mixed-Integer Nonlinear Programming](https://doi.org/10.1007/978-3-319-89920-6_63): Design an energy-efficient decentralised water supply system for skyscrapers by mixed-integer nonlinear programming. The MIPLIB instances are the MILP version of the same design problem by the same first author; the paper states an MINLP formulation, so the pump curve modelling differs.

## Record

Listed objective 82030.53014261136 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists 82030.53014261134; published solution file header =obj= 82030.53014261136; the published solution, checked by our verifier: feasible within 1e-6, objective 82030.53014261136.

## Data and requirements

https://miplib.zib.de/WebData/instances/dws012-01.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/miplib/dws012-01.mps.gz`. Run `python3 scripts/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

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
