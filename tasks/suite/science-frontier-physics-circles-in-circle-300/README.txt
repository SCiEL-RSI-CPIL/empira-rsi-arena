# Three hundred equal circles in a circle

**Source collection:** [Open packing, covering and energy problems with published records: source-based research tasks](../../papers/science-frontier_physics.md)\
**Task id:** `science-frontier-physics-circles-in-circle-300`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Place the centres of 300 equal, non-overlapping circles inside a circle of radius 1 so that their common radius is as large as possible. The best known packing, with radius 0.0531542991469, was found by Lai, Hao, Yue, Lü and Fu in 2021 and is not proven optimal.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> array of shape (N, 2): centres of N equal circles inside the unit circle (radius 1, centred at the origin). Score: the largest common radius r = min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|)) for which the circles are disjoint and inside the container, computed in double precision; maximize. A centre on or outside the container, coincident centres, a wrong shape or non-finite values make the candidate invalid.

## State of the art

The best known packing of 300 equal circles in a circle of radius 1 has circle radius 0.053154299146851187500993025786 (Packomania, last update 2024-12-25), credited to X. Lai, J.-K. Hao, D. Yue, Z. Lü and Z.-H. Fu (September 2021; method published in 2022). Records near this size were improved again in 2023 and 2024; no optimality proof is known.

Reference value: best known value `0.053154299146851186` (reviewed 2026-10-11).

- [The best known packings of equal circles in a circle (Packomania)](https://www.packomania.com/cci/cci.html), Eckard Specht (2024)
- [Iterated dynamic thresholding search for packing equal circles into a circular container](https://doi.org/10.1016/j.ejor.2021.08.044), X. Lai, J.-K. Hao, D. Yue, Z. Lü, Z.-H. Fu (2022)

Parameters: `{"n": 300, "published": 0.053154299146851186}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find the largest common radius of 300 non-overlapping equal circles inside a circle (densest packing of equal circles in a circle).

Why it matters: Dense packings of equal discs and spheres in containers are classical problems of discrete geometry and models of confined hard-particle systems; optima are proven only for small N, and record tables kept for decades are still being improved by new global-optimisation methods.

- [Iterated dynamic thresholding search for packing equal circles into a circular container](https://doi.org/10.1016/j.ejor.2021.08.044): Main research objective: Design a search algorithm for packing equal circles in a circular container and improve the best known packings. Task connection: Its 2021 search produced the N = 300 record used by the task.
- [Dense packings of congruent circles in a circle](https://doi.org/10.1016/S0012-365X(97)00050-2): Main research objective: Find the densest known packings of N congruent circles in a circle for N up to 65 and describe their structure. Task connection: The N = 50 packing of that paper is still the record that the task's score is compared with.

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

- paper: https://doi.org/10.1016/j.ejor.2021.08.044 (Lai, Hao, Yue, Lü and Fu (2022), Iterated dynamic thresholding search for packing equal circles into a circular container: the search method behind the 2021 records for N = 126 to 319.)
- benchmark: https://www.packomania.com/cci/cci.html (E. Specht, Packomania table of equal circles in a circle (last update 2024-12-25), row N = 300: radius 0.053154299146851187500993025786, reference [21] (Lai, Hao, Yue, Lu and Fu, September 2021); coordinates in cci/txt/cci300.txt.)

## Limitations

- Optimality is proven for N <= 14 and N = 19: the Packomania table marks N <= 13 and N = 19 in bold, and a proof for N = 14 was published by Ekanayake and LaFountain (Ital. J. Pure Appl. Math. 51 (2024) 115-136); computer-assisted proofs for N = 15 were announced on Packomania on 2026-10-11. The record table also notes a queue of unprocessed submissions, so it may change.
- The record coordinates are public (Packomania), so a candidate can reproduce the record from memory or by downloading it beforehand; scores measure the returned construction, not the search that produced it.
- Scores are double-precision radii of the returned centres. A score above the record by at most 1e-11 is reported as a tie; a larger gain is a candidate new record that should be confirmed in higher precision and submitted to the record table before it is claimed.
- Candidate Python runs inside the evaluator process and is trusted research code; the evaluator recomputes the radius from the returned array but does not sandbox the program.
