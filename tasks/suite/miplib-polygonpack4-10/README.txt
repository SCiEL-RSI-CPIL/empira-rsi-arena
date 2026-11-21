# MIPLIB 2017 open instance polygonpack4-10

**Task id:** `miplib-polygonpack4-10`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance polygonpack4-10 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Packing a maximum-area subset of non-convex polygons, each allowed four rotations, into a rectangle without overlap; submitted by Antonio Frangioni as a simplified version of a real application, with one of five polygon sets and 7, 10 or 15 polygons chosen from it.

## Why it matters

Scientific importance: not established (`unreviewed`). No qualifying paper whose main objective is the problem this instance models has been established for this task. MIPLIB 2017 is a benchmark library, and its paper is not such evidence. This label describes the available evidence, not proof that the task has no scientific value.

## Record

Listed objective -53594508.70758325 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists -53594508.70758325; published solution file header =obj= -53594508.70758325; the published solution, checked by our verifier: feasible within 1e-6, objective -53594508.707583256.

## Data and requirements

https://miplib.zib.de/WebData/instances/polygonpack4-10.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/miplib/polygonpack4-10.mps.gz`. Run `python3 scripts/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The objective listed on the open-instances page may have been improved since 2026-11-11 (the list did not change between 2026-11-02 and 2026-11-11).
- Candidates may call HiGHS through highspy; a stronger solver is a method, not a discovery by the workflow.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
