# Physics: three-dimensional spin-glass ground states on fixed instances

**Source collection:** [Physics research: source-based research tasks](../../papers/science-physics_research.md)\
**Task id:** `science-physics-spin-glass-torus-3d`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Find ground states of nine Edwards-Anderson spin glasses on 5x5x5, 6x6x6 and 7x7x7 periodic cubic lattices with Gaussian couplings. The evaluator recomputes the cut weight of every spin configuration and compares it with the exact optimum computed for these instances by branch-and-cut.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

ground_state(instance) -> list of instance['nodes'] spins, each -1 or +1. instance has keys name, nodes, side, dimension and edges (list of [i, j, weight] with integer weights and 0-based node indices) of a 3-dimensional periodic grid of side 5, 6 and 7 with Gaussian couplings. ground_state is called once per instance in a sandboxed subprocess with a 240 s CPU budget for all nine instances together. Score (maximize): mean over the instances of cut_weight / published_optimal_cut, where the evaluator recomputes the cut weight from the edge list; 1.0 means reaching the exact optimum everywhere. The spin-glass energy of the couplings J = -weight is sum(weight) - 2 * cut_weight, so a larger cut is a lower energy

## State of the art

The optimal cut weights of these nine instances are known exactly (10933215, 11582216, 11552046 for 5x5x5; 17434469, 20217380, 19475011 for 6x6x6; 28302918, 33611981, 29118445 for 7x7x7), so a score of 1.0 is attainable and cannot be exceeded.

Reference value: proven optimum `1.0` (reviewed 2026-10-10).

- [Biq Mac Library documentation, Table 10](https://biqmac.aau.at/biqmaclib.pdf)
- [Computing exact ground states of hard Ising spin glass problems by branch-and-cut](https://doi.org/10.1002/3527603794.ch4)

Parameters: `{"dimension": 3, "cpu_seconds": 240}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `natural_science`, data `public_benchmark`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `physics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find the ground states of three-dimensional Edwards-Anderson Ising spin glasses on finite cubic lattices with Gaussian couplings, an NP-hard problem.

Why it matters: Ground-state energies and their finite-size scaling are the basis for studying the spin-glass phase, stiffness exponents and the computational complexity boundary between two and three dimensions; exact ground states had to be computed with branch-and-cut, and heuristics are still measured against them.

- [Computing exact ground states of hard Ising spin glass problems by branch-and-cut](https://doi.org/10.1002/3527603794.ch4): Main research objective: Compute exact ground states of Ising spin glasses on two- and three-dimensional toroidal lattices. Task connection: The task uses those instances and their exact optima as the reference and asks a candidate to approach them heuristically.
- [On the computational complexity of Ising spin glass models](https://doi.org/10.1088/0305-4470/15/10/028): Main research objective: Determine the computational complexity of finding spin-glass ground states on planar and three-dimensional lattices. Task connection: Explains why the two-dimensional instances are tractable in principle and the three-dimensional ones are not.

The cited papers' main goal is the spin-glass ground-state problem itself, which the evaluator scores exactly against published optima.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Instances and optimal values come from the Biq Mac library (toroidal grid graphs with Gaussian couplings of Liers et al. 2004); the optima were computed there by branch-and-cut and are used only as reference values, never shown to the candidate. The evaluator recomputes every cut weight from the edge list. Scores depend on the CPU budget, which is far below the branch-and-cut effort behind the reference values, and the nine instances of one task share that budget.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/bio_physics/spinglass/torus_3d.json.gz`

## Sources

- paper: https://doi.org/10.1002/3527603794.ch4 (Liers, Juenger, Reinelt and Rinaldi (2004) compute exact ground states of these three-dimensional toroidal Ising spin glasses by branch-and-cut.)
- paper: https://doi.org/10.1088/0305-4470/15/10/028 (Barahona (1982) proves that finding spin-glass ground states on three-dimensional lattices is NP-hard, which is why these instances are hard for heuristics.)
- benchmark: https://biqmac.aau.at/biqmaclib.html (Biq Mac library: the t3g torus instances (ising_all.tar.gz) and their optimal cut values in Table 10 of biqmaclib.pdf; no licence stated, so the archive is downloaded by the user and not redistributed here.)

## Limitations

- The reference values come from branch-and-cut runs without a time limit; this task gives a few minutes of CPU for nine instances, so the score is budget-dependent and not comparable with exact solvers.
- Only these nine fixed instances are scored, so a better score is a better heuristic on them rather than a physical result about the three-dimensional Edwards-Anderson model.
- Couplings are the integer-scaled Gaussian weights of the published instances; the evaluator reports the corresponding energy but scores the cut weight.
- Ground-state degeneracy is not examined: the task only measures how close the recomputed cut weight comes to the published optimum.
- NP-hardness is a worst-case statement and these instances have at most 343 spins: simulated annealing with a few hundred replicas reaches the published optimum of eight of the nine instances within seconds each (the ninth to within 3e-4), so a score close to 1.0 is attainable inside the budget and the headroom of the task is the gap between the seed (0.885) and 1.0.
