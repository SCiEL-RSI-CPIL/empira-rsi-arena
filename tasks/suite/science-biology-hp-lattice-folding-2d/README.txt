# Biology: HP lattice protein folding on the square lattice

**Source collection:** [Biology research: source-based research tasks](../../papers/science-biology_research.md)\
**Task id:** `science-biology-hp-lattice-folding-2d`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Fold the ten standard two-dimensional HP benchmark sequences (20 to 100 residues) into low-energy self-avoiding conformations on the square lattice. The evaluator checks self-avoidance, recomputes the hydrophobic contact energy of every conformation and scores the mean ratio to the lowest energies reported in the literature.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fold(sequence, dimension) -> list of len(sequence) integer lattice points (lists of length dimension). sequence is a string of H and P residues; dimension is 2 (square lattice). The returned walk must be self-avoiding with unit steps between consecutive residues. Energy = minus the number of H-H pairs on neighbouring lattice sites that are not adjacent along the chain; the evaluator recomputes it. fold is called once per benchmark sequence in a sandboxed subprocess with a 120 s CPU budget for all sequences together. Score (maximize): mean over the 10 benchmark sequences of energy / best_known_energy (both energies are negative), so 0 is an unfolded chain, 1.0 reproduces every best-known energy and a value above 1.0 beats one

## State of the art

A score of 1.0 means matching the lowest energies reported for all ten sequences: -9, -8, -14, -23, -21, -36, -42, -53, -48 and -50 for the 20-, 25-, 36-, 48-, 50-, 60-, 64-, 85- and two 100-residue chains. Fragment-regrowth Monte Carlo (FRESS) and Wang-Landau sampling reach every one of these values, each with searches far longer than this task's CPU budget.

Reference value: best known value `1.0` (reviewed 2026-10-10).

- [Biopolymer structure simulation and optimization via fragment regrowth Monte Carlo](https://doi.org/10.1063/1.2736681)
- [Optimized Wang-Landau sampling of lattice polymers](https://arxiv.org/abs/1207.3974)

Parameters: `{"dimension": 2, "cpu_seconds": 120}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `program`, domain `natural_science`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `biology`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find the minimum-energy conformations of hydrophobic-polar lattice chains, the standard minimal model of how a protein sequence determines a compact native structure.

Why it matters: The HP model is the canonical statistical-mechanics abstraction of protein folding: its ground states and their degeneracies are used to study designability, folding thermodynamics and the sequence-to-structure map, and finding them is NP-hard, which is why algorithmic progress on the benchmark chains is itself a research result.

- [A lattice statistical mechanics model of the conformational and sequence spaces of proteins](https://doi.org/10.1021/ma00200a030): Main research objective: Introduce and analyse the HP lattice model of protein conformational and sequence space. Task connection: The task searches for minimum-energy conformations of exactly this model on the square lattice.
- [A study of density of states and ground states in hydrophobic-hydrophilic protein folding models by equi-energy sampling](https://doi.org/10.1063/1.2208607): Main research objective: Find ground states and estimate densities of states of HP model proteins with a new Monte Carlo sampler. Task connection: Supplies the benchmark sequences and the best-known energies that the task uses as reference values.

The cited papers' main goal is the HP model and the search for its ground states, which is exactly what the evaluator scores.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- The benchmark sequences and the best-known energies are transcribed from the cited papers (2D: Kou, Oh and Wong 2006 and Zhang, Kou and Liu 2007; 3D 48-mers: Bachmann and Janke 2004 for the Yue et al. 1995 set; longer 3D chains: Zhang, Kou and Liu 2007 and Wuest and Landau 2012). Only the ten 48-mers have ground states proven by exhaustive hydrophobic-core construction; the other values are the lowest energies reported in that literature, so a lower score is possible in principle. The evaluator checks self-avoidance and recomputes every energy, but it does not prove optimality, and the single CPU budget for all sequences makes the score budget-dependent.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://doi.org/10.1021/ma00200a030 (Lau and Dill (1989) define the HP lattice model of protein folding whose minimum-energy conformations this task searches for.)
- paper: https://doi.org/10.1063/1.2208607 (Kou, Oh and Wong (2006), Tables IV and V: the nine widely used 2D benchmark sequences and the lowest energies reached by equi-energy sampling, genetic algorithms, evolutionary Monte Carlo, PERM, SISPER and human-guided search.)
- paper: https://doi.org/10.1063/1.2736681 (Zhang, Kou and Liu (2007), Tables II and III: the benchmark sequences longer than 50 residues and the fragment-regrowth energies used here as best-known values.)
- paper: https://arxiv.org/abs/1207.3974 (Wuest and Landau (2012), Table II: Wang-Landau sampling reaches the same putative ground-state energies for the long 2D chains (-42, -53, -48, -50).)

## Limitations

- The HP model is a coarse two-letter abstraction of a protein: its ground states are not real protein structures, and a better HP energy is a better solution of the stated lattice optimisation problem rather than a structural-biology result.
- Best-known energies for these 2D sequences are the lowest values reported in the cited literature, not proven optima; the evaluator cannot certify optimality, only recompute energies.
- All ten sequences share one CPU budget inside the sandboxed subprocess, so the score depends on that budget and is not comparable with the multi-hour searches of the cited papers.
- The published energies are embedded in the evaluator as reference values only; the candidate never receives them.
