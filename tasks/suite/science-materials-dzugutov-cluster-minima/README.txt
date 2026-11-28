# Dzugutov cluster global minima (polytetrahedral model metal)

**Source collection:** [Molecular and materials design research: source-based research tasks](../../papers/science-chemistry_research.md)\
**Task id:** `science-materials-dzugutov-cluster-minima`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Find the lowest-energy structures of 38- to 60-atom clusters bound by the Dzugutov potential, a model of glass- and quasicrystal-forming metals whose energy maximum favours polytetrahedral order (six instances). Energies are recomputed exactly and compared with the putative global minima of Doye, Wales and Simdyankin.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct(instance) -> array of shape (n, 3): Cartesian coordinates of n atoms in reduced units. instance = {'potential': 'dzugutov', 'n': atoms, 'A', 'B', 'a', 'b', 'c', 'd', 'm'}; called once for each n in (38, 45, 50, 55, 57, 60); all calls together within 300 s (enforced) and should use at most 4 threads (not enforced by the evaluator). The evaluator recomputes E = sum_{i<j} V(r_ij) with V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b], A = 5.82, B = 1.28, a = 1.87, b = 1.94, c = 1.10, d = 0.27, m = 16; atoms must be at least 0.5 apart. Score: mean over instances of 100*(E - E_pub)/|E_pub| against the putative global minima of Doye, Wales and Simdyankin (2001) (minimize; 0 matches all six, negative is lower). chem_clusters.relax(positions, 'dzugutov') and chem_clusters.dzugutov_energy_gradient(positions) may be called

## State of the art

Matching the six putative global minima of Doye, Wales and Simdyankin (2001) gives 0.

Reference value: best known value `0.0` (reviewed 2026-10-10).

- [Global optimization and the energy landscapes of Dzugutov clusters](https://doi.org/10.1039/b008881l)

## Labels

Scientific contribution: `method` (tier 2); artifact `construction`, domain `natural_science`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `materials_science`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find the lowest-energy structure of an atomic cluster for a given interatomic potential (global optimisation on a rugged energy landscape). Here: clusters of the Dzugutov potential, a model of glass- and quasicrystal-forming metals.

Why it matters: Polytetrahedral order underlies metallic glasses, Frank-Kasper phases and quasicrystals; the global minima of Dzugutov clusters show which polytetrahedral structures such systems favour.

- [Global optimization and the energy landscapes of Dzugutov clusters](https://doi.org/10.1039/b008881l): Main research objective: Determine the global minima and energy landscapes of Dzugutov clusters up to 100 atoms and their polytetrahedral structures. Task connection: The task scores six sizes up to 60 atoms against the published putative minima.

The cited papers' main objective is the underlying problem the task scores; the local data, splits and limits are stated in the task's limitations.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Pair potential recomputed without further cutoff beyond its own smooth cutoffs at a and b. Reference check (tests/test_chem_clusters.py): the relaxed 7-atom pentagonal bipyramid and 13-atom icosahedron reproduce the published minima within six decimals. The authors note that minima above about 60 atoms may be improvable, so only sizes up to 60 are scored.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:scipy`

## Sources

- paper: https://doi.org/10.1039/b008881l (Doye, Wales and Simdyankin (2001): Dzugutov potential parameters and table of putative global minima for N = 3-100 (arXiv cond-mat/0011018).)
- paper: https://doi.org/10.1103/physreva.46.r2984 (Dzugutov (1992): the pair potential designed to suppress crystallisation in a monatomic liquid.)
- archive: https://arxiv.org/abs/cond-mat/0011018 (LaTeX source of the global-minimum table (six decimals).)

## Limitations

- The references are 2001 putative minima; the authors note that sizes above about 60 may have lower minima, so only sizes up to 60 are used, and the Cambridge Cluster Database listed the same six energies on 2026-11-24 (its coordinate files score them with this evaluator within 5e-7).
- Putative minima are public, so remembered structures score well without search.
- The score averages relative energy excesses over six instances within a shared 300 s budget.
