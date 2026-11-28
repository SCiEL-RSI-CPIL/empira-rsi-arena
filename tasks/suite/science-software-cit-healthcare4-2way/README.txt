# Software engineering: pairwise covering array for an industrial healthcare model

**Source collection:** [Software engineering research: source-based research tasks](../../papers/science-software_engineering.md)\
**Task id:** `science-software-cit-healthcare4-2way`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Build a smallest pairwise covering array for the 35-parameter Healthcare4 model, one of the IBM industrial combinatorial test design models of Segall et al., under its forbidden value combinations. Every feasible pair of values must appear in some test; fewer tests is better.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

generate(model) -> list of rows. model = {'name': str, 'strength': t, 'values': [v_0, ..., v_{k-1}], 'forbidden': [[[p, level], ...], ...]}: parameter p takes a level in 0..v_p-1 and each forbidden entry lists parameter-level pairs that must not all occur in one row. Return a list of rows, each a list of k Python ints. Every row must avoid all forbidden combinations, and the rows together must contain every combination of levels of t distinct parameters that occurs in at least one valid configuration (the evaluator decides feasibility exactly). Score: number of rows, minimised; an array that misses a feasible combination or contains a forbidden one is invalid. generate() runs in a sandboxed subprocess (no files, network or subprocesses; CPU limit 60 s for pairwise and 120 s for 3-way tasks, 2 GB); it should be deterministic (seed any randomness, do not read clocks).

Parameters: `{"instance": "healthcare4-2way", "cpu_seconds": 60, "reference": 46, "lower_bound": 42}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `software_engineering`, data `public_benchmark`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `software_engineering`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Combinatorial interaction testing of configurable software: choose as few test configurations as possible such that every feasible combination of t option settings is exercised, while respecting constraints that forbid some option combinations.

Why it matters: Configurable systems such as compilers, web servers and model checkers have astronomically many configurations; interaction faults are triggered by a few options together, so smaller constrained covering arrays reduce testing cost while keeping the interaction coverage guarantee. Constraint handling is a central open difficulty because invalid configurations cannot be executed.

- [Using Binary Decision Diagrams for Combinatorial Test Design](https://doi.org/10.1145/2001420.2001451): Main research objective: Construct combinatorial test designs for industrial systems with complex restrictions between parameters. Task connection: The task's Healthcare4 model is one of the industrial models collected in this work and redistributed in the ICSE 2015 archive.
- [Learning Combinatorial Interaction Test Generation Strategies Using Hyperheuristic Search](https://doi.org/10.1109/ICSE.2015.71): Main research objective: Construct smaller constrained covering arrays for configurable systems by learning search strategies with a hyperheuristic. Task connection: The task uses this paper's published benchmark archive and its objective (array size under constraints).

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Models and constraints are the CASA-format files of the ICSE 2015 CIT benchmark archive (Zenodo 581657); their 1-based level numbers are converted to 0-based. Feasible t-way combinations are decided exactly at preparation time (backtracking search); an independent Z3 recomputation agreed on every required combination of all five models (validation/software-checks.json), hence recreated_verified. The candidate gets a fixed CPU budget in a Python sandbox, unlike the published C++ solvers (1000 s to 5 h per run), so sizes are comparable with the published values but run times are not.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/software/cit/healthcare4-2way.json`

## Sources

- paper: https://doi.org/10.1145/2001420.2001451 (Segall, Tzoref-Brill and Farchi (ISSTA 2011): combinatorial test design with binary decision diagrams; source of the 20 IBM industrial models, including Healthcare4.)
- paper: https://doi.org/10.1109/ICSE.2015.71 (Jia, Cohen, Harman and Petke (ICSE 2015): hyperheuristic search for constrained covering arrays; publishes the benchmark archive used here.)
- archive: https://zenodo.org/records/581657 (ICSE15_HHSA_Benchmarks.zip (CC BY 4.0): CASA-format model and constraint files; levels are numbered from 1.)
- repository: https://github.com/jkunlin/fastca/blob/master/additional_table.pdf (Supplementary table of the FastCA repository: HHSA and CASA sizes (2-way and 3-way, 5-hour cutoff) on the same HHSA-format instance files.)

## Limitations

- Published sizes on these exact files: HHSA and CASA 46 rows (FastCA repository supplementary table, 5-hour cutoff); the counting lower bound is 42.
- The array size is the only objective; generation time is bounded by the sandbox CPU limit and is not scored, whereas the published solvers ran for 1000 s to 5 h in C++.
- Feasible combinations are decided by an exact search at preparation time and cross-checked with Z3 in the tests; a Z3 disagreement would change the required set.
- The models are public; a candidate could embed a precomputed array. Scores are comparable with published sizes only for arrays generated within the contract.
