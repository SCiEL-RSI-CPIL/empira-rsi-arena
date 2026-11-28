# Sutton-Chen transition-metal cluster global minima

**Source collection:** [Molecular and materials design research: source-based research tasks](../../papers/science-chemistry_research.md)\
**Task id:** `science-materials-sutton-chen-cluster-minima`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Find the lowest-energy structures of 38- to 80-atom clusters of the Sutton-Chen many-body metal potential with the silver, nickel and gold parameter sets (six instances). Energies are recomputed exactly and compared with the putative global minima of Doye and Wales.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct(instance) -> array of shape (n, 3): Cartesian coordinates of n atoms in reduced units (lattice constant 1). instance = {'potential': 'sutton_chen', 'kind': '12-6' | '9-6' | '10-8', 'n': atoms, 'n_exp', 'm_exp', 'c'}; called once for each of six instances (kind, n) = (12-6, 38), (12-6, 75), (9-6, 59), (9-6, 80), (10-8, 55), (10-8, 79); all calls together within 300 s (enforced) and should use at most 4 threads (not enforced by the evaluator). The evaluator recomputes E = sum_i [1/2 sum_{j!=i} r_ij^-n_exp - c sqrt(sum_{j!=i} r_ij^-m_exp)] over all pairs (no cutoff) with (n_exp, m_exp, c) = (12, 6, 144.41), (9, 6, 39.432), (10, 8, 34.408); atoms must be at least 0.35 apart. Score: mean over instances of 100*(E - E_pub)/|E_pub| against the putative global minima tabulated by Doye and Wales (1998) (minimize; 0 matches all six, negative is lower). chem_clusters.relax(positions, 'sutton_chen', kind) and chem_clusters.sutton_chen_energy_gradient(positions, kind) may be called

## State of the art

Matching the six putative global minima of Doye and Wales (1998) gives 0. The 2024 GrowPAL search reports lower energies than the Cambridge Cluster Database only for sizes not used here; the database entries for the six sizes used here equal the 1998 table (read on 2026-11-24).

Reference value: best known value `0.0` (reviewed 2026-11-24).

- [Global minima for transition metal clusters described by Sutton-Chen potentials](https://doi.org/10.1039/a709249k)
- [An Efficient Growth Pattern Algorithm (GrowPAL) for Cluster Structure Prediction](https://doi.org/10.1021/acs.jctc.4c00365)

## Labels

Scientific contribution: `method` (tier 2); artifact `construction`, domain `natural_science`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `materials_science`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find the lowest-energy structure of an atomic cluster for a given interatomic potential (global optimisation on a rugged energy landscape). Here: transition-metal clusters described by the many-body Sutton-Chen potential.

Why it matters: The structures of small metal clusters control catalytic, optical and magnetic properties of nanoparticles; many-body metal potentials give energy landscapes in which global optimisation still finds improvements decades later.

- [Global minima for transition metal clusters described by Sutton-Chen potentials](https://doi.org/10.1039/a709249k): Main research objective: Find the global minima of Sutton-Chen 12-6, 9-6 and 10-8 clusters with up to 80 atoms and analyse their structures. Task connection: The task scores six of the tabulated sizes against those putative minima.
- [An Efficient Growth Pattern Algorithm (GrowPAL) for Cluster Structure Prediction](https://doi.org/10.1021/acs.jctc.4c00365): Main research objective: Develop an efficient algorithm for predicting cluster structures and apply it to Lennard-Jones, Sutton-Chen and boron clusters. Task connection: Its Sutton-Chen search reproduces the six scored minima and identifies the sizes with later improvements that the task excludes.

The cited papers' main objective is the underlying problem the task scores; the local data, splits and limits are stated in the task's limitations.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Many-body Sutton-Chen energy recomputed without cutoff. Reference check (tests/test_chem_clusters.py): relaxed icosahedra and truncated octahedra reproduce nine published minima of all three parameter sets (n = 13, 38, 55, 79) within the four published decimals. Sizes for which Lopez-Castro et al. (2024) report lower energies than the database (12-6 n = 68, 10-8 n = 58) or failed to reproduce it (10-8 n = 60) are not used.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:scipy`

## Sources

- paper: https://doi.org/10.1039/a709249k (Doye and Wales (1998): Sutton-Chen 12-6, 9-6 and 10-8 clusters, table of putative global minima for N <= 80 (arXiv cond-mat/9711038).)
- paper: https://doi.org/10.1080/09500839008206493 (Sutton and Chen (1990): the long-range Finnis-Sinclair potential and its (n, m, c) parameters.)
- paper: https://doi.org/10.1021/acs.jctc.4c00365 (Lopez-Castro, Ortiz-Chi and Merino (2024): GrowPAL search over Sutton-Chen clusters with 5-80 atoms; reports new lower minima for 12-6 N = 68 and 10-8 N = 58 (not used here).)
- archive: https://arxiv.org/abs/cond-mat/9711038 (LaTeX source of the global-minimum table (four decimals).)

## Limitations

- References are the 1998 putative minima (four decimals); the Cambridge Cluster Database listed the same six energies on 2026-11-24, and its coordinate files score them with this evaluator within 5e-5. Lopez-Castro et al. (2024) report database-level minima for all sizes from 5 to 80 except three (lower energies for 12-6 N = 68 and 10-8 N = 58, not reproduced for 10-8 N = 60); those three are excluded.
- Putative minima are public, so remembered structures score well without search; the energy is computed without a cutoff, as in the source, and differs from implementations that truncate the sums.
- The score averages relative energy excesses over six instances within a shared 300 s budget.
