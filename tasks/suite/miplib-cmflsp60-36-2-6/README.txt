# MIPLIB 2017 open instance cmflsp60-36-2-6

**Task id:** `miplib-cmflsp60-36-2-6`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance cmflsp60-36-2-6 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Capacitated multi-family lot-sizing problem (production planning of items grouped into families over a multi-period horizon), submitted by Andrea Arias.

## Why it matters

**Problem:** Plan production of items grouped in families with capacities over the horizon at minimum cost in the capacitated multi-family lot-sizing problem (instance cmflsp60-36-2-6).

Lot-sizing with family set-ups is a standard production planning problem; the formulation strongly affects solvability.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [A comparative analysis of mixed-integer linear formulations for the multi-family capacitated lot-sizing problem](https://doi.org/10.1080/21681015.2018.1446461): Compare mixed-integer linear formulations of the multi-family capacitated lot-sizing problem. The MIPLIB submitter is the first author of this paper and the problem is the same; it was not checked that the instance sizes (items, periods) are those of the paper.

## Record

Listed objective 73891245.384587 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists 73891245.38458699; published solution file header =obj= 73891245.384587; the published solution, checked by our verifier: feasible within 1e-6, objective 73891245.38458702.

## Data and requirements

https://miplib.zib.de/WebData/instances/cmflsp60-36-2-6.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/miplib/cmflsp60-36-2-6.mps.gz`. Run `python3 scripts/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

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
