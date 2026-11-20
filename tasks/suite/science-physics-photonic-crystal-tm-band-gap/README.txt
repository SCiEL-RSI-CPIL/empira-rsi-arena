# Physics: photonic-crystal unit cell with a large higher-order TM band gap

**Source collection:** [Physics research: source-based research tasks](../../papers/science-physics_research.md)\
**Task id:** `science-physics-photonic-crystal-tm-band-gap`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Design a symmetric 32x32-pixel unit cell of air and dielectric (epsilon 12) for a two-dimensional square-lattice photonic crystal that opens the largest gap between the fourth and fifth TM bands. The evaluator solves the band structure with an exact-coefficient plane-wave method over the irreducible Brillouin zone and scores the gap-midgap ratio.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

design(problem) -> 32 x 32 nested list of 0/1 pixels of one square unit cell (1 = dielectric with epsilon 12, 0 = air), invariant under the square symmetry group about the cell centre (cell == cell[::-1] row-wise and column-wise and cell == its transpose). problem gives pixels 32, eps_low 1, eps_high 12, lower_band 4 and polarization TM. design runs in a sandboxed subprocess with a 120 s CPU budget. The evaluator computes TM bands with a 289-plane-wave Galerkin method using the exact Fourier coefficients of the pixel map, on 30 wave vectors covering the boundary and interior of the irreducible Brillouin zone. Score (maximize): gap-midgap ratio 2 (min omega_5 - max omega_4) / (min omega_5 + max omega_4) between TM bands 4 and 5; negative when the bands overlap

## State of the art

No published value under this exact discretization and gap measure. For context, the textbook rod crystal (the seed) gives 0.043 here, close to the 3.84 % reported by the MPB tutorial for the same nominal structure, and a 150-second greedy pixel-flip search run during task construction reached 0.311.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [MPB Python tutorial](https://mpb.readthedocs.io/en/latest/Python_Tutorial/)

Parameters: `{"lower_band": 4, "cpu_seconds": 120}`

## Labels

Scientific contribution: `method` (tier 2); artifact `construction`, domain `natural_science`, data `none`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `physics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find the arrangement of dielectric and air in a periodic unit cell that opens the largest photonic band gap at a chosen pair of bands.

Why it matters: Band gaps are what make photonic crystals useful for waveguiding, lasers and cavities; maximizing them over the material distribution is a well-posed inverse-design problem that produced new crystal geometries, and higher-order gaps are harder than the first.

- [Maximizing band gaps in two-dimensional photonic crystals](https://doi.org/10.1137/S0036139998338455): Main research objective: Formulate and solve the problem of maximizing photonic band gaps over the dielectric distribution in a unit cell. Task connection: The task is the same optimization, restricted to a two-material 32x32 symmetric cell and the TM 4-5 gap, with the band structure computed by a plane-wave method.
- [Bandgap optimization of two-dimensional photonic crystals using semidefinite programming and subspace methods](https://doi.org/10.1016/j.jcp.2010.01.023): Main research objective: Compute optimized photonic-crystal structures with large band gaps, including higher TM and TE gaps. Task connection: Uses the same square-lattice symmetric-cell setting and higher-gap targets that the task adopts.

The cited papers' main goal is maximizing photonic band gaps by designing the unit cell, exactly what the evaluator scores.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- The plane-wave model is converged to about 0.1 percentage points of gap-midgap ratio on test structures (289 versus 625 plane waves), and it reproduces the MPB tutorial's TM gap of the rod crystal (epsilon 12, radius 0.2) within 0.2 percentage points, the difference coming from the pixelated circle. Band extrema are searched on 30 wave vectors only, which is standard for symmetric cells but can miss extrema between samples. The symmetry requirement and the 32-pixel grid follow the setting of Men et al. (2010); the score is not comparable with their eigenvalue-based gap measure.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:scipy`

## Sources

- paper: https://doi.org/10.1137/S0036139998338455 (Cox and Dobson (1999), Maximizing band gaps in two-dimensional photonic crystals: the gap-maximization problem for TM and TE polarizations on a square lattice.)
- paper: https://doi.org/10.1016/j.jcp.2010.01.023 (Men, Nguyen, Freund, Parrilo and Peraire (2010): band-gap optimization of symmetric square-lattice cells on a 1/32 grid with air and a dielectric, including higher TM gaps such as the fourth.)
- paper: https://doi.org/10.1103/PhysRevLett.100.153904 (Sigmund and Hougaard (2008), Geometric properties of optimal photonic crystals: topology-optimized band-gap structures.)
- repository: https://mpb.readthedocs.io/en/latest/Python_Tutorial/ (MPB Python tutorial: the rod crystal (epsilon 12, radius 0.2a) has TM gaps of 38.95 % (bands 1-2) and 3.84 % (bands 4-5); used to check the local solver.)

## Limitations

- The evaluator is a 289-plane-wave Galerkin solver on a 32-pixel grid: designs are scored under this discretization, which is converged to about 0.1 percentage points on test structures but is not a full-wave simulation of a fabricated device.
- Band extrema are searched on 30 wave vectors of the irreducible zone (boundary plus interior samples), standard for symmetric cells but able to miss extrema between samples.
- Only one polarization and one gap (TM bands 4-5) are optimized; complete band gaps and fabrication constraints such as minimum feature size are not part of the score.
- Published optimized gaps (Men et al. 2010) use an eigenvalue-based gap ratio and a finite-element discretization, so they are not directly comparable with this score.
