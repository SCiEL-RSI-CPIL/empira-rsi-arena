# MIPLIB 2017 open instance rocII-10-11

**Task id:** `miplib-rocii-10-11`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance rocII-10-11 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Optimal control of a bounded-confidence opinion dynamics system (steering the opinions of a group of agents), submitted by Joerg Rambau and imported from the MIPLIB 2010 submissions.

## Why it matters

**Problem:** Choose controls for a strategic agent that steers the opinions of the other agents in a deterministic bounded-confidence (Hegselmann-Krause type) opinion dynamics over a finite horizon, as a mixed-integer program (instance rocII-10-11).

Opinion dynamics models how opinions form in societies; optimal control of them asks how a campaign can best influence a group, and the resulting mixed-integer programs are hard for general solvers.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Optimal Opinion Control: The Campaign Problem](https://doi.org/10.18564/jasss.2847): Formulate and study the problem of a strategic agent that optimally influences the opinions of normal agents following bounded-confidence dynamics. The MIPLIB page names the same optimal control problem (Rambau opinion control); it was not checked that the paper contains these exact instances or model variant.

## Record

Listed objective -11.7943245021343 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists -11.7943245021343; published solution file header =obj= -11.7943245021343; the published solution, checked by our verifier: feasible within 1e-6, objective -11.794324502134275.

## Data and requirements

https://miplib.zib.de/WebData/instances/rocII-10-11.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/miplib/rocII-10-11.mps.gz`. Run `python3 scripts/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

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
