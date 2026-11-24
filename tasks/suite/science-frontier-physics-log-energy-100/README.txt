# Minimal logarithmic energy of 100 points on the sphere

**Source collection:** [Open packing, covering and energy problems with published records: source-based research tasks](../../papers/science-frontier_physics.md)\
**Task id:** `science-frontier-physics-log-energy-100`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Place 100 points on the unit sphere to minimize the logarithmic energy, the sum over all pairs of log(1/distance), the energy in Smale's 7th problem. The lowest published energy, -1083.377140539, is the putative global minimum: it is not proven minimal, but random-restart local optimisation reproduces it within seconds, so the task measures whether an optimiser reaches it.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct() -> array of shape (N, 3) of nonzero vectors; each is normalised to the unit sphere. Score: the logarithmic energy sum_{i<j} log(1 / |x_i - x_j|) (natural logarithm, chordal distances, pair terms summed with math.fsum); minimize. Points closer than 1e-9, a wrong shape or non-finite values make the candidate invalid.

## State of the art

The lowest published logarithmic energy we found for 100 points on the unit sphere is E = -1083.377140539 (standardized pair energy v_0 = -0.21886406879576 in the data tables of Nerattini, Brauchart and Kiessling 2014, compiled from Rakhmanov, Saff and Zhou 1995 and Calef 2009). The lowest of the 39 local minima published by Amore, Figueroa and Ramos (2025) attains it to 2.9e-10. No minimizer is proven for this N; rigorous results exist only for a few small N such as N = 6 and N = 12.

Reference value: best known value `-1083.377140539012` (reviewed 2026-10-11).

- [Optimal N-point configurations on the sphere: "magic" numbers and Smale's 7th problem](https://doi.org/10.1007/s10955-014-1107-7), R. Nerattini, J. S. Brauchart, M. K.-H. Kiessling (2014)
- [Minimal discrete energy on the sphere](https://doi.org/10.4310/MRL.1994.v1.n6.a3), E. A. Rakhmanov, E. B. Saff, Y. M. Zhou (1994)
- [Exploring the energy landscape of the logarithmic potential: local minima and stationary states](https://arxiv.org/abs/2512.12416), P. Amore, V. Figueroa, R. Ramos (2025)
- [Mathematical problems for the next century](https://doi.org/10.1007/BF03025291), S. Smale (1998)

Parameters: `{"n": 100, "published": -1083.377140539012}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find 100 points on the unit sphere with minimal logarithmic energy (Smale's 7th problem; elliptic Fekete points).

Why it matters: Smale's 7th problem asks for an efficient construction of near-minimal logarithmic-energy points, which are linked to well-conditioned polynomial systems and to charged-particle models; even putative minima for moderate N are known only from extensive numerical searches.

- [Minimal discrete energy on the sphere](https://doi.org/10.4310/MRL.1994.v1.n6.a3): Main research objective: Study and compute N-point configurations on the sphere with minimal logarithmic and Riesz energy. Task connection: The task's objective is the logarithmic energy of that paper; its numerical minima underlie the record.
- [Optimal N-point configurations on the sphere: "magic" numbers and Smale's 7th problem](https://doi.org/10.1007/s10955-014-1107-7): Main research objective: Analyse putatively minimal Riesz and logarithmic energies of N points on the sphere in relation to Smale's 7th problem. Task connection: Its s = 0 data table gives the record energies used for N = 100 and 150.
- [Mathematical problems for the next century](https://doi.org/10.1007/BF03025291): Main research objective: State major open problems for the 21st century, including the 7th: computing near-minimal logarithmic-energy points on the sphere. Task connection: The task's objective is the energy of Smale's 7th problem.

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

- paper: https://doi.org/10.4310/MRL.1994.v1.n6.a3 (Rakhmanov, Saff and Zhou (1994), Minimal discrete energy on the sphere: the minimal logarithmic and Riesz energy problems and numerical minima.)
- paper: https://doi.org/10.1007/s10955-014-1107-7 (Nerattini, Brauchart and Kiessling (2014), supplement table 6.2 (s = 0): putatively minimal standardized pair energy v_0(100) = -0.21886406879576, i.e. E = -1083.377140539.)
- archive: https://doi.org/10.5281/zenodo.17923310 (Amore, Figueroa and Ramos (2025), CC BY 4.0 dataset of local minima of the logarithmic energy; LocalMinima.zip, member 100.zip (39 minima for N = 100), whose lowest minimum reproduces the tabulated value to 2.9e-10.)

## Limitations

- The record is the lowest published value we could check: the tables of Nerattini, Brauchart and Kiessling (2014), compiled from Rakhmanov, Saff and Zhou (1995) and Calef (2009), and the 2025 landscape survey of Amore, Figueroa and Ramos. R. Womersley's published point sets could not be accessed when the task was prepared, so a lower value in sources we could not read cannot be excluded.
- Points are normalised to the unit sphere before scoring, so only directions matter; the record coordinates are public, so scores measure the returned construction, not the search that produced it.
- Candidate Python runs inside the evaluator process and is trusted research code; the evaluator recomputes the score from the returned array but does not sandbox the program.
- The reference is easy to reach: L-BFGS on the normalised coordinates from random starting points ties it within the 1e-9 margin in about a second (55 of 129 random starts in 35 s on one core for N = 100), no start went below it, and the 39 local minima published by Amore, Figueroa and Ramos (2025) contain no lower value. The headroom of the task is therefore the gap between the seed and this value; a score below it by more than the margin would be a new record and is not expected.
