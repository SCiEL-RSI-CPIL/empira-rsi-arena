# MIPLIB 2017 open instance core4284-1064

**Task id:** `miplib-core4284-1064`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Find a feasible solution of the MIPLIB 2017 instance core4284-1064 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Set covering instance from Italian railway (crew scheduling) models, submitted by A. Caprara, M. Fischetti and P. Toth.

## Why it matters

**Problem:** Choose a minimum-cost set of columns covering all rows of the Italian railway set covering instance core4284-1064.

Set covering models crew scheduling of Italian State Railways; the core instances are large and hard for general MIP solvers.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Algorithms for railway crew management](https://doi.org/10.1007/bf02614314): Solve railway crew management (crew scheduling) problems for Italian railways with set-covering-based algorithms. The task is a set covering instance arising from this application; the paper does not state which MIPLIB instance it is.
- [A Heuristic Method for the Set Covering Problem](https://doi.org/10.1287/opre.47.5.730): Design a Lagrangian-based heuristic for large set covering problems, tested on railway instances. The task is a set covering instance of the type tested; identity with the paper's test instances is not confirmed on the page.

## Record

Listed objective 1062.0 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists 1062.0; published solution file header =obj= 1062.0; the published solution, checked by our verifier: feasible within 1e-6, objective 1062.0.

## Data and requirements

https://miplib.zib.de/WebData/instances/core4284-1064.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/arnew/miplib/core4284-1064.mps.gz`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The objective listed on the open-instances page may have been improved since 2026-11-11 (the list did not change between 2026-11-02 and 2026-11-11).
- Candidates may call HiGHS through highspy; a stronger solver is a method, not a discovery by the workflow.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
