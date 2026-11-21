# Covering the sphere with 50 equal caps

**Source collection:** [Open packing, covering and energy problems with published records: source-based research tasks](../../papers/science-frontier_physics.md)\
**Task id:** `science-frontier-physics-sphere-covering-50`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Place 50 points on the unit sphere so that the largest angular distance from any point of the sphere to its nearest chosen point (the covering radius) is as small as possible. The best known covering, with radius 18.3000226 degrees, was found by Hardin, Sloane and Smith in 1994 and is not proven optimal.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> array of shape (N, 3) of nonzero vectors; each is normalised to the unit sphere. Score: the covering radius in degrees, max over points y of the sphere of the angular distance from y to the nearest configuration point (the largest circumradius of the facets of the convex hull, each facet checked to have an empty cap); minimize. Coincident points, configurations inside a closed hemisphere, numerically inconsistent hulls, a wrong shape or non-finite values make the candidate invalid.

## State of the art

The best known covering of the sphere by 50 equal caps has angular covering radius 18.3000226 degrees (rounded up), from the table of R. H. Hardin, N. J. A. Sloane and W. D. Smith (1994), now maintained by H. Cohn, who invites improvements. No optimality proof is known for N = 50.

Reference value: best known value `18.3000226` (reviewed 2026-10-11).

- [Sloane's tables of point configurations on spheres (coverings by N spherical caps in 3 dimensions)](https://cohn.mit.edu/sloane/), R. H. Hardin, N. J. A. Sloane, W. D. Smith; maintained by H. Cohn (1994)
- [Covering a sphere by equal circles, and the rigidity of its graph](https://doi.org/10.1017/S0305004100070134), T. Tarnai, Zs. Gáspár (1991)

Parameters: `{"n": 50, "published": 18.3000226}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find 50 points on the sphere whose equal caps of smallest possible angular radius cover the whole sphere.

Why it matters: Optimal spherical coverings bound the worst-case error of placing sites, sensors or quadrature points on the sphere; they are proven only for small N, and the 1994 records for larger N remain open to improvement.

- [Covering a sphere by equal circles, and the rigidity of its graph](https://doi.org/10.1017/S0305004100070134): Main research objective: Find coverings of the sphere by N equal circles with the smallest angular radius and analyse their rigidity. Task connection: The task scores the same covering radius for N = 50 and 130 against the Hardin-Sloane-Smith records.

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
- `py:scipy`

## Sources

- paper: https://doi.org/10.1017/S0305004100070134 (Tarnai and Gáspár (1991), Covering a sphere by equal circles, and the rigidity of its graph: the covering problem, conjectured coverings for N = 15-20 and sporadic larger N.)
- benchmark: https://cohn.mit.edu/sloane/ (Hardin, Sloane and Smith table of coverings by N spherical caps in 3 dimensions (calculations of 1994, maintained by H. Cohn; radii rounded up), row N = 50: 18.3000226 degrees; coordinates in cov3-50.txt (also neilsloane.com/coverings/dim3/cover.3.50.txt).)

## Limitations

- The table gives covering radii rounded up to 1e-7 degrees; a covering counts as a new record only when it is below the table value by more than 1e-7 degrees. Optimal coverings are proven only for small N; the table value for N = 50 dates from 1994, and the value for N = 130 is Henry Cohn's 2026 improvement of the 1994 value.
- Points are normalised to the unit sphere before scoring, so only directions matter; the record coordinates are public, so scores measure the returned construction, not the search that produced it.
- Candidate Python runs inside the evaluator process and is trusted research code; the evaluator recomputes the score from the returned array but does not sandbox the program.
