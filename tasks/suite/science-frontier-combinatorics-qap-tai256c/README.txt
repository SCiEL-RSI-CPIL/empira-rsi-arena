# Quadratic assignment: QAPLIB tai256c

**Source collection:** [Open combinatorial design, coding and assignment problems with published records: source-based research tasks](../../papers/science-frontier_combinatorics.md)\
**Task id:** `science-frontier-combinatorics-qap-tai256c`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Assign the 256 facilities of the QAPLIB instance tai256c to 256 locations so that the sum of flow times distance is as small as possible. The best known assignment costs 44,759,294 and no optimality proof exists.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

solve(a, b) -> permutation p of 0..n-1 (list of n integers); a and b are this task's QAPLIB matrices (int64 NumPy arrays of shape (n, n), first and second matrix of the instance file, passed as copies). Facility i is placed at location p[i]. Score: the exact integer cost sum_{i,j} a[i, j] * b[p[i], p[j]], minimized. The instance file is read from the prepared data directory and its SHA-256 is checked before every evaluation.

## State of the art

The best known solution of Taillard's grey-density instance tai256c (the largest unsolved QAPLIB instance) costs 44,759,294; it was found by an ant-colony algorithm (QAPLIB code ANT, T. Stuetzle's MAX-MIN ant system, 1997) and is recorded in QAPLIB. Branch-and-bound lower bounds are within about 1.25% (Fujii, Kim, Kojima, Mittelmann and Shinano, 2024); QAPLIB's bound is 43,849,646. No later improvement was found in the literature reviewed on 2026-10-12.

Reference value: best known value `44759294` (reviewed 2026-10-12).

- [QAPLIB - A Quadratic Assignment Problem Library - Problem instances and solutions (Edinburgh DataShare)](https://doi.org/10.7488/ds/3428), R. E. Burkard, E. Cela, S. E. Karisch, F. Rendl, M. Anjos, P. Hahn (2022)
- [An exceptionally difficult binary quadratic optimization problem with symmetry: a challenge for the largest unsolved QAP instance Tai256c](https://doi.org/10.1007/s11590-024-02157-2), K. Fujii, S. Kim, M. Kojima, H. D. Mittelmann, Y. Shinano (2024)

Parameters: `{"instance": "tai256c", "n": 256, "sha256": "e7d72083d49f4768e5a15a02ee33fde038a794e7407e5e485c3c110f1509fafd", "published": 44759294}`

## Labels

Scientific contribution: `method` (tier 2); artifact `construction`, domain `combinatorial_optimisation`, data `public_benchmark`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `optimization`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find a minimum-cost Koopmans-Beckmann quadratic assignment of facilities to locations for the QAPLIB instance tai256c.

Why it matters: The quadratic assignment problem models facility and hospital layout, keyboard design, backboard wiring and turbine balancing. It is NP-hard and notoriously difficult in practice: QAPLIB instances above about 40 facilities remain unsolved, and the best known solutions of the large instances have resisted improvement for decades, so a cheaper permutation is a new best known solution of a standard open instance.

- [Assignment problems and the location of economic activities](https://doi.org/10.2307/1907742): Main research objective: Formulate the assignment of indivisible economic activities to locations, introducing the quadratic assignment problem. Task connection: The task's objective is the Koopmans-Beckmann quadratic assignment cost on the QAPLIB instance tai256c.
- [An exceptionally difficult binary quadratic optimization problem with symmetry: a challenge for the largest unsolved QAP instance Tai256c](https://doi.org/10.1007/s11590-024-02157-2): Main research objective: Solve or tightly bound tai256c, the largest unsolved QAPLIB instance, by reformulating it as a binary quadratic problem and exploiting its symmetry in branch and bound. Task connection: The paper's object is exactly this instance; its lower bound leaves a gap of about 1.25% to the best known cost 44,759,294 that the task asks to reduce from above.

Both papers have the quadratic assignment problem (and, for the second, this instance or the search for its best solutions) as their main objective; the evaluator scores exactly the QAP cost of a permutation on the pinned instance.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Independent exact objective in the QAPLIB convention (first matrix indexed by facilities, second by locations). The published record construction for every task was scored with this evaluator and reproduces the published value exactly (validation/human-results/frontier_combinatorics.json). Instances and best known solutions come from the QAPLIB deposit on Edinburgh DataShare (CC BY 4.0); the instance files are fetched by scripts/prepare_frontier_combinatorics.py and pinned by SHA-256. Unit tests compare the objective with brute force on small instances.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/frontier_combinatorics/tai256c.dat`

## Sources

- benchmark: https://doi.org/10.7488/ds/3428 (QAPLIB deposit (CC BY 4.0): tai256c.dat in qapdata.tar.gz (SHA-256 e7d72083d49f4768...) and the best known permutation tai256c.sln in qapsoln.tar.gz; the README table lists 44759294 for tai256c.)
- paper: https://doi.org/10.1023/A:1008293323270 (Burkard, Karisch and Rendl (1997), QAPLIB - a quadratic assignment problem library: instance definitions and conventions.)
- paper: https://doi.org/10.2307/1907742 (Koopmans and Beckmann (1957), Assignment problems and the location of economic activities: the quadratic assignment model.)
- paper: https://doi.org/10.1007/s11590-024-02157-2 (K. Fujii, S. Kim, M. Kojima, H. D. Mittelmann, Y. Shinano (2024), An exceptionally difficult binary quadratic optimization problem with symmetry: a challenge for the largest unsolved QAP instance Tai256c.)

## Limitations

- Branch-and-bound lower bounds are within about 1.25% (Fujii, Kim, Kojima, Mittelmann and Shinano, 2024); QAPLIB's bound is 43,849,646.
- The instance file is fetched by scripts/prepare_frontier_combinatorics.py and pinned by SHA-256; the evaluator refuses a file with another digest.
- The rank-matching seed costs 98,685,678.
- The record construction is public, so a candidate can reproduce the record by embedding it; the score measures the returned object, not the search that produced it, and only a strictly smaller score than the best known value is a new result.
- Candidate Python runs inside the evaluator process as trusted research code; the evaluator recomputes every property of the returned object exactly and trusts nothing the candidate reports.
