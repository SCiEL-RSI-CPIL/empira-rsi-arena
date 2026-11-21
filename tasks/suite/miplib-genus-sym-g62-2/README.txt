# MIPLIB 2017 open instance genus-sym-g62-2

**Task id:** `miplib-genus-sym-g62-2`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `miplib` (suite/_lib/arnew)\
**Areas:** optimization, mathematics

## Task

Find a feasible solution of the MIPLIB 2017 instance genus-sym-g62-2 with as low an objective value as possible (a maximisation model is negated, so that lower is better for every instance). A solution is feasible when every variable bound, every row bound and the integrality of every integer variable hold within an absolute tolerance of 1e-6 on the values as they are returned; integer variables are not rounded first.

## Candidate interface

solve(model) -> numpy array of num_cols values, one per column. model has name, mps_path (the uncompressed MPS file), num_cols, num_rows, A (scipy CSR matrix of the rows), col_cost, col_lower, col_upper, row_lower, row_upper, integrality (bool array), offset, maximise and indicators (a list of (row, column, value): such a row has to hold only when that column equals the value; empty for most instances). Score: col_cost @ x + offset in minimisation form (negated when maximise is true), minimized; an infeasible vector is invalid.

## The instance

Minimum-genus integer programming model of a graph from the AT&T (North) or Rome graph collections, submitted by Stephan Beyer (the model is the one of Beyer, Chimani, Hedtke and Kotrbcik); the -sym variants add symmetry-breaking constraints.

## Why it matters

**Problem:** Find an embedding of the graph genus-sym-g62-2 (an AT&T/North or Rome graph) on an orientable surface of smallest genus, in the integer programming model of the minimum-genus problem.

Minimum genus is NP-hard and a basic parameter of topological graph theory with no standard practical exact method before ILP models; open instances here mean the optimum is not proven or not found.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [A Practical Method for the Minimum Genus of a Graph: Models and Experiments](https://doi.org/10.1007/978-3-319-38851-9_6): Develop and experimentally compare integer programming models and a practical exact method for computing the minimum genus of a graph. The task is one instance of an ILP model of this problem on a graph from the North or Rome collections (the page names these collections; the exact model variant is the submitter's and was not compared line by line with the paper).

## Record

Listed objective -38.0 (in the model's own sense).

**Record validation:** reproduced. open-instances page lists -38.0; published solution file header =obj= -38.0; the published solution, checked by our verifier: feasible within 1e-6, objective -38.0.

## Data and requirements

https://miplib.zib.de/WebData/instances/genus-sym-g62-2.mps.gz (size varies; some files are large)

Requires: `py:numpy`, `py:scipy`, `py:highspy`, `file:data/miplib/genus-sym-g62-2.mps.gz`. Run `python3 scripts/prepare_new_data.py --family miplib` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- The objective listed on the open-instances page may have been improved since 2026-11-11 (the list did not change between 2026-11-02 and 2026-11-11).
- Candidates may call HiGHS through highspy; a stronger solver is a method, not a discovery by the workflow.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
