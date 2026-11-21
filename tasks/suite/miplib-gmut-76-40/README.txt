# MIPLIB 2017 open instance gmut-76-40

**Task id:** `miplib-gmut-76-40`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance gmut-76-40 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Harvest scheduling model of a hypothetical forest (maximise net timber revenue under single harvest, harvest-flow, final average age and maximum clearcut size constraints), submitted by Nora Konnyu and imported from the MIPLIB 2010 submissions.

## Why it matters

**Problem:** Choose which management units (and generalised units) to harvest in which period to maximise net timber revenue subject to single harvest, harvest-flow, minimum final average age and maximum clearcut size constraints (instance gmut-76-40).

Maximum clearcut size limits are a regulatory requirement in forestry; including them makes harvest planning a hard combinatorial problem.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [A cutting plane method for solving harvest scheduling models with area restrictions](https://doi.org/10.1016/j.ejor.2013.01.020): Solve harvest scheduling models with area restrictions by a cutting plane method. The MIPLIB submitter is the first author of this paper and the page describes the same area-restricted harvest model; the instances are hypothetical forests and it was not checked that they are those of the paper.
- [Harvest Scheduling Subject to Maximum Area Restrictions: Exploring Exact Approaches](https://doi.org/10.1287/opre.1040.0169): Solve the forest harvest scheduling problem with maximum area restrictions by an exact approach based on clique representations. Same model class (area restriction model with generalised units); the instances of the task are not those of the paper.

## Record

Listed objective -14169477.79550346 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists -14169477.79550346; published solution file header =obj= -14169477.795503462; the published solution, checked by our verifier: feasible within 1e-6, objective -14169477.79550346.

## Data and requirements

https://miplib.zib.de/WebData/instances/gmut-76-40.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/miplib/gmut-76-40.mps.gz`. Run `python3 scripts/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The objective listed on the open-instances page may have been improved since 2026-11-11 (the list did not change between 2026-11-02 and 2026-11-11).
- Candidates may call HiGHS through highspy; a stronger solver is a method, not a discovery by the workflow.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
