# Biology: HP lattice protein folding on the cubic lattice

**Source collection:** [Biology research: source-based research tasks](../../papers/science-biology_research.md)\
**Task id:** `science-biology-hp-lattice-folding-3d`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Fold the sixteen standard three-dimensional HP benchmark chains (the ten 48-residue sequences with proven ground states plus six longer chains of 64 to 136 residues) on the simple cubic lattice. The evaluator checks self-avoidance, recomputes every hydrophobic contact energy and scores the mean ratio to the published reference energies.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fold(sequence, dimension) -> list of len(sequence) integer lattice points (lists of length dimension). sequence is a string of H and P residues; dimension is 3 (simple cubic lattice). The returned walk must be self-avoiding with unit steps between consecutive residues. Energy = minus the number of H-H pairs on neighbouring lattice sites that are not adjacent along the chain; the evaluator recomputes it. fold is called once per benchmark sequence in a sandboxed subprocess with a 240 s CPU budget for all sequences together. Score (maximize): mean over the 16 benchmark sequences of energy / best_known_energy (both energies are negative), so 0 is an unfolded chain, 1.0 reproduces every best-known energy and a value above 1.0 beats one

## State of the art

A score of 1.0 means matching the reference energy of every chain: -32, -34, -34, -33, -32, -32, -32, -31, -34 and -33 for the ten 48-mers (proven ground states), and -56, -56, -72, -58, -75 and -83 for the 64-, 67-, 88-, 103-, 124- and 136-residue chains. Fragment-regrowth Monte Carlo and Wang-Landau sampling reach all of these, the longest ones after hours to days of computation.

Reference value: best known value `1.0` (reviewed 2026-10-10).

- [Thermodynamics of lattice heteropolymers](https://arxiv.org/abs/cond-mat/0310707)
- [Biopolymer structure simulation and optimization via fragment regrowth Monte Carlo](https://doi.org/10.1063/1.2736681)
- [Optimized Wang-Landau sampling of lattice polymers](https://arxiv.org/abs/1207.3974)

Parameters: `{"dimension": 3, "cpu_seconds": 240}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `natural_science`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `biology`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find the ground states of hydrophobic-polar lattice chains on the three-dimensional cubic lattice, including the ten 48-residue sequences whose native states were established by exhaustive hydrophobic-core construction.

Why it matters: The HP model is the canonical statistical-mechanics abstraction of protein folding: its ground states and their degeneracies are used to study designability, folding thermodynamics and the sequence-to-structure map, and finding them is NP-hard, which is why algorithmic progress on the benchmark chains is itself a research result.

- [A lattice statistical mechanics model of the conformational and sequence spaces of proteins](https://doi.org/10.1021/ma00200a030): Main research objective: Introduce and analyse the HP lattice model of protein conformational and sequence space. Task connection: The task searches for minimum-energy conformations of this model on the cubic lattice.
- [A test of lattice protein folding algorithms](https://doi.org/10.1073/pnas.92.1.325): Main research objective: Determine the native structures of ten designed 48-residue cubic-lattice HP sequences and use them to test folding algorithms. Task connection: Provides the ten 48-mers and their proven ground-state energies, which are the task's reference values.

The cited papers' main goal is the HP model and the search for its ground states, which is exactly what the evaluator scores.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- The benchmark sequences and the best-known energies are transcribed from the cited papers (2D: Kou, Oh and Wong 2006 and Zhang, Kou and Liu 2007; 3D 48-mers: Bachmann and Janke 2004 for the Yue et al. 1995 set; longer 3D chains: Zhang, Kou and Liu 2007 and Wuest and Landau 2012). The ten 48-mers have ground states proven by exhaustive hydrophobic-core construction, and Hsu et al. (2003) report that the energies of the 64- and 67-residue chains of Yue and Dill (1995) were determined exactly by the same method; the other values are the lowest energies reported in that literature, so only for them is a lower energy (a score above 1.0) possible in principle. The evaluator checks self-avoidance and recomputes every energy, but it does not prove optimality, and the single CPU budget for all sequences makes the score budget-dependent.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://doi.org/10.1021/ma00200a030 (Lau and Dill (1989) define the HP lattice model whose cubic-lattice ground states this task searches for.)
- paper: https://doi.org/10.1073/pnas.92.1.325 (Yue, Fiebig, Thomas, Chan, Shakhnovich and Dill (1995) introduce the ten 48-residue cubic-lattice sequences and determine their ground states by constraint-based hydrophobic-core construction.)
- paper: https://arxiv.org/abs/cond-mat/0310707 (Bachmann and Janke (2004), Table II: the ten 48-mer sequences in run-length notation with their ground-state energies (-32, -34, -34, -33, -32, -32, -32, -31, -34, -33) and degeneracies.)
- paper: https://doi.org/10.1063/1.2736681 (Zhang, Kou and Liu (2007), Tables II and IV: the longer 3D benchmark sequences and the lowest energies found for them (-56, -56, -72, -57, -75, -83).)
- paper: https://arxiv.org/abs/1207.3974 (Wuest and Landau (2012), Table II: Wang-Landau sampling reaches -44, -56, -56, -72, -58, -75 and -83 on the long 3D chains, improving the 103-residue reference to -58.)

## Limitations

- The HP model is a two-letter lattice abstraction: its ground states are not real protein structures, so a better energy is a better solution of a lattice optimisation problem, not a structural-biology finding.
- The ten 48-mers have ground states established by exhaustive hydrophobic-core construction, and the energies of the 64- and 67-residue chains are reported as exact results of the same method (Hsu et al. 2003, citing Yue and Dill 1995); the references for the 88-, 103-, 124- and 136-residue chains are the lowest energies reported in the cited literature and can in principle be beaten.
- All sixteen chains share one CPU budget in the sandboxed subprocess, so scores are budget-dependent and not comparable with the multi-hour or multi-day searches of the cited papers.
- The reference energies are used only for scoring; the candidate never receives them.
