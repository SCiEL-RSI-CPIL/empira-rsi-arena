# Two hundred equal circles in a square

**Source collection:** [Open packing, covering and energy problems with published records: source-based research tasks](../../papers/science-frontier_physics.md)\
**Task id:** `science-frontier-physics-circles-in-square-200`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Place the centres of 200 equal, non-overlapping circles inside a unit square so that their common radius is as large as possible. The best known packing, with radius 0.036612798904, is credited to Huang, Ye and Cantrell (2010) and is not proven optimal.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> array of shape (N, 2): centres of N equal circles inside the square [-1/2, 1/2]^2 (side 1). Score: the largest common radius r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|)) for which the circles are disjoint and inside the square, computed in double precision; maximize. A centre on or outside the square, coincident centres, a wrong shape or non-finite values make the candidate invalid.

## State of the art

The best known packing of 200 equal circles in a unit square has circle radius 0.036612798904 (Packomania, last update 2026-09-10), credited to W. Huang, T. Ye and D. W. Cantrell (2010). Many neighbouring sizes were improved in 2022; no optimality proof is known.

Reference value: best known value `0.036612798904` (reviewed 2026-10-11).

- [The best known packings of equal circles in a square (Packomania)](https://www.packomania.com/csq/csq.html), Eckard Specht (2026)
- [Greedy vacancy search algorithm for packing equal circles in a square](https://doi.org/10.1016/j.orl.2010.07.004), W. Huang, T. Ye (2010)

Parameters: `{"n": 200, "published": 0.036612798904}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find the largest common radius of 200 non-overlapping equal circles inside a square (equivalently, spread 200 points in a square).

Why it matters: Dense packings of equal discs and spheres in containers are classical problems of discrete geometry and models of confined hard-particle systems; optima are proven only for small N, and record tables kept for decades are still being improved by new global-optimisation methods.

- [Greedy vacancy search algorithm for packing equal circles in a square](https://doi.org/10.1016/j.orl.2010.07.004): Main research objective: Develop a heuristic for packing equal circles in a square and improve the best known packings up to N = 200. Task connection: The task's N = 200 record is credited to this method (with D. W. Cantrell's refinements).
- [Improving dense packings of equal disks in a square](https://doi.org/10.37236/1524): Main research objective: Find denser packings of N equal disks in a square and report new record packings. Task connection: It reports the N = 50 record packing that the task's score is compared with.

The cited papers' main objective is the underlying problem the task scores; the record table, instance size and precision limits are stated in the task's limitations.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Independent implementation; the published record constructions (Packomania, the Hardin-Sloane-Smith covering table, the Amore-Figueroa-Ramos log-energy minima) were scored with this evaluator and reproduce the published values (validation/human-results/frontier_physics.json). No optimality is claimed for any value; a score beyond the record by more than the stated margin is a candidate new record that needs independent high-precision confirmation.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://doi.org/10.1016/j.orl.2010.07.004 (Huang and Ye (2010), Greedy vacancy search algorithm for packing equal circles in a square: the search method credited, with D. W. Cantrell's refinements, for the N = 200 record.)
- benchmark: https://www.packomania.com/csq/csq.html (E. Specht, Packomania table of equal circles in a square (last update 2026-09-10), row N = 200: radius 0.036612798904, references [25,24] (Huang and Ye 2010; D. W. Cantrell 2010); coordinates in csq/txt/csq200.txt.)

## Limitations

- Optimality is proven only for N <= 30 and N = 36 (bold entries of the Packomania table). Packomania publishes these records with 12 decimals, so ties are judged with the 1e-11 margin.
- The record coordinates are public (Packomania), so a candidate can reproduce the record from memory or by downloading it beforehand; scores measure the returned construction, not the search that produced it.
- Scores are double-precision radii of the returned centres. A score above the record by at most 1e-11 is reported as a tie; a larger gain is a candidate new record that should be confirmed in higher precision and submitted to the record table before it is claimed.
- Candidate Python runs inside the evaluator process and is trusted research code; the evaluator recomputes the radius from the returned array but does not sandbox the program.
