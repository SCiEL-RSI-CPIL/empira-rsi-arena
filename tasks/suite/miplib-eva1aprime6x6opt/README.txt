# MIPLIB 2017 open instance eva1aprime6x6opt

**Task id:** `miplib-eva1aprime6x6opt`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization, materials science

## Task

Find a feasible solution of the MIPLIB 2017 instance eva1aprime6x6opt with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

MILP for generating structures with negative thermal expansion coefficients, submitted by Yoshihiro Kanno; the grid size in the name is the only size information.

## Why it matters

**Problem:** Find a periodic frame structure with a negative thermal expansion coefficient by solving the mixed-integer programme eva1aprime6x6opt.

Materials with negative thermal expansion are used to compensate thermal deformation in precision devices; designing them as discrete structures leads to hard mixed-integer programmes.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Optimal design of periodic frame structures with negative thermal expansion via mixed integer programming](https://doi.org/10.1007/s11081-015-9276-z): Design periodic frame structures with negative thermal expansion by mixed-integer programming. The instance is a MILP of this kind submitted by Kanno; the page does not state that the paper's exact model and grid sizes are the ones used, so the match is by topic and submitter.

## Record

Listed objective -18.101469789649965 (in the model's own sense).

**Record validation:** matches a pinned source. open-instances page lists -18.10146978964996; published solution file header =obj= -18.101469789649965. Note: Our verifier does not accept the published solution: a constraint is violated (largest violations: rows 5.3e-06, bounds 0.0e+00, integrality 0.0e+00). MIPLIB's own check is looser than this task's absolute 1e-6, so matching the record here needs a cleaner solution; a result that beats it here also beats it there.

## Data and requirements

https://miplib.zib.de/WebData/instances/eva1aprime6x6opt.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/arnew/miplib/eva1aprime6x6opt.mps.gz`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

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
