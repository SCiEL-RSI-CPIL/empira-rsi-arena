# One hundred equal spheres in a cube

**Source collection:** [Open packing, covering and energy problems with published records: source-based research tasks](../../papers/science-frontier_physics.md)\
**Task id:** `science-frontier-physics-spheres-in-cube-100`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Place the centres of 100 equal, non-overlapping spheres inside a unit cube so that their common radius is as large as possible. The best known packing, with radius 0.1113823475125, was found by Eckard Specht's search program and is not proven optimal.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> array of shape (N, 3): centres of N equal spheres inside the cube [-1/2, 1/2]^3 (side 1). Score: the largest common radius r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|)) for which the spheres are disjoint and inside the cube, computed in double precision; maximize. A centre on or outside the cube, coincident centres, a wrong shape or non-finite values make the candidate invalid.

## State of the art

The best known packing of 100 equal spheres in a unit cube has sphere radius 0.111382347512479750227357863499 (Packomania, last update 2026-10-08), credited to E. Specht's program scu (2010-2012). Several neighbouring sizes were improved between 2012 and October 2026; no optimality proof is known.

Reference value: best known value `0.11138234751247975` (reviewed 2026-10-11).

- [The best known packings of equal spheres in a cube (Packomania)](https://www.packomania.com/scu/scu.html), Eckard Specht (2026)
- [Dense packings of equal spheres in a cube](https://doi.org/10.37236/1786), T. Gensane (2004)

Parameters: `{"n": 100, "published": 0.11138234751247975}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find the largest common radius of 100 non-overlapping equal spheres inside a cube (equivalently, spread 100 points in a cube).

Why it matters: Dense packings of equal discs and spheres in containers are classical problems of discrete geometry and models of confined hard-particle systems; optima are proven only for small N, and record tables kept for decades are still being improved by new global-optimisation methods.

- [Dense packings of equal spheres in a cube](https://doi.org/10.37236/1786): Main research objective: Find dense packings of N equal spheres in a cube (equivalently, spread N points in a cube) and analyse their structure. Task connection: It defines the problem and the record conventions that the task scores for N = 50 and 100.

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

- paper: https://doi.org/10.37236/1786 (Gensane (2004), Dense packings of equal spheres in a cube: the problem and its equivalence with spreading points in a cube.)
- benchmark: https://www.packomania.com/scu/scu.html (E. Specht, Packomania table of equal spheres in a cube (last update 2026-10-08), row N = 100: radius 0.111382347512479750227357863499, reference [31] (program scu, 2010-2012); coordinates in scu/txt/scu100.txt.)

## Limitations

- Optimality is proven only for a few small N (for example N = 14, Joós 2009); the Packomania table was being revised on 2026-10-04 to 2026-10-08 with new submissions for other sizes, so it may change.
- The record coordinates are public (Packomania), so a candidate can reproduce the record from memory or by downloading it beforehand; scores measure the returned construction, not the search that produced it.
- Scores are double-precision radii of the returned centres. A score above the record by at most 1e-11 is reported as a tie; a larger gain is a candidate new record that should be confirmed in higher precision and submitted to the record table before it is claimed.
- Candidate Python runs inside the evaluator process and is trusted research code; the evaluator recomputes the radius from the returned array but does not sandbox the program.
