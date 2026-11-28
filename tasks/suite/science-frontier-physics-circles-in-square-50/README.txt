# Fifty equal circles in a square

**Source collection:** [Open packing, covering and energy problems with published records: source-based research tasks](../../papers/science-frontier_physics.md)\
**Task id:** `science-frontier-physics-circles-in-square-50`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Place the centres of 50 equal, non-overlapping circles inside a unit square so that their common radius is as large as possible. The best known packing, with radius 0.071377103865, was found by Boll, Donovan, Graham and Lubachevsky in 2000 and is not proven optimal.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> array of shape (N, 2): centres of N equal circles inside the square [-1/2, 1/2]^2 (side 1). Score: the largest common radius r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|)) for which the circles are disjoint and inside the square, computed in double precision; maximize. A centre on or outside the square, coincident centres, a wrong shape or non-finite values make the candidate invalid.

## State of the art

The best known packing of 50 equal circles in a unit square has circle radius 0.071377103865 (Packomania, last update 2026-09-10), credited to Boll, Donovan, Graham and Lubachevsky (2000). No optimality proof is known (proofs exist for N <= 33 and N = 36).

Reference value: best known value `0.071377103865` (reviewed 2026-11-24).

- [The best known packings of equal circles in a square (Packomania)](https://www.packomania.com/csq/csq.html), Eckard Specht (2026)
- [Improving dense packings of equal disks in a square](https://doi.org/10.37236/1524), D. W. Boll, J. Donovan, R. L. Graham, B. D. Lubachevsky (2000)

Parameters: `{"n": 50, "published": 0.071377103865}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find the largest common radius of 50 non-overlapping equal circles inside a square (equivalently, spread 50 points in a square).

Why it matters: Dense packings of equal discs and spheres in containers are classical problems of discrete geometry and models of confined hard-particle systems; optima are proven only for small N, and record tables kept for decades are still being improved by new global-optimisation methods.

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

- paper: https://doi.org/10.37236/1524 (Boll, Donovan, Graham and Lubachevsky (2000), Improving dense packings of equal disks in a square: new record packings including N = 50.)
- benchmark: https://www.packomania.com/csq/csq.html (E. Specht, Packomania table of equal circles in a square (last update 2026-09-10), row N = 50: radius 0.071377103865, reference [14]; coordinates in csq/txt/csq50.txt.)

## Limitations

- Optimality is proven for N <= 33 and N = 36: the Packomania table marks N <= 30 and N = 36 in bold, and computer-assisted proofs for N = 31, 32 and 33 were published by Markot (J. Global Optim. 81 (2021) 773-803). Packomania publishes these records with 12 decimals, so ties are judged with the 1e-11 margin.
- The record coordinates are public (Packomania), so a candidate can reproduce the record from memory or by downloading it beforehand; scores measure the returned construction, not the search that produced it.
- Scores are double-precision radii of the returned centres. A score above the record by at most 1e-11 is reported as a tie; a larger gain is a candidate new record that should be confirmed in higher precision and submitted to the record table before it is claimed.
- Candidate Python runs inside the evaluator process and is trusted research code; the evaluator recomputes the radius from the returned array but does not sandbox the program.
