# MIPLIB 2017 open instance lr2-22dr3-333vc4v17a-t60

**Task id:** `miplib-lr2-22dr3-333vc4v17a-t60`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance lr2-22dr3-333vc4v17a-t60 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Group 1 instance (60-period horizon) of the Maritime Inventory Routing Problem Library MIRPLib, submitted by Dimitri Papageorgiou: routing a vessel fleet between loading and discharging ports with split pickups and deliveries.

## Why it matters

**Problem:** Route a fleet of vessels between loading and discharging ports and schedule the loads so that port inventories stay within bounds at minimum cost, for the MIRPLib instance lr2-22dr3-333vc4v17a-t60.

Maritime inventory routing is a real shipping problem (for example liquefied natural gas or ore); the library instances stay open for commercial MIP solvers, as the MIPLIB page states.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Two-Stage Decomposition Algorithms for Single Product Maritime Inventory Routing](https://doi.org/10.1287/ijoc.2014.0601): Solve single-product deep-sea maritime inventory routing problems with two-stage decomposition algorithms. The task instances are MIRPLib Group 1 or 2 models of the same problem class; the MIPLIB page and the MIRPLib page credit the instance definitions to the library paper, so it was not verified that this paper solves these exact instances.

## Record

Listed objective -27262.58917622233 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists -27262.58917622233; published solution file header =obj= -27262.58917622233; the published solution, checked by our verifier: feasible within 1e-6, objective -27262.589176222325.

## Data and requirements

https://miplib.zib.de/WebData/instances/lr2-22dr3-333vc4v17a-t60.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/arnew/miplib/lr2-22dr3-333vc4v17a-t60.mps.gz`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

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
