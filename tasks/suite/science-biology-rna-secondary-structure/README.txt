# Biology: RNA secondary-structure prediction against Rfam consensus structures

**Source collection:** [Biology research: source-based research tasks](../../papers/science-biology_research.md)\
**Task id:** `science-biology-rna-secondary-structure`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Predict the secondary structure of 90 non-coding RNA sequences from 47 Rfam families that have experimentally determined three-dimensional structures. The evaluator compares the predicted base pairs with the family's consensus pairing projected onto each sequence and scores the mean F1.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

predict_structure(sequence) -> dot-bracket string of the same length as sequence, containing only '.', '(' and ')' with balanced brackets. sequence is an ACGU string of 40 to 150 nucleotides from an Rfam family that has a three-dimensional structure. predict_structure is called once per sequence in a sandboxed subprocess with a 180 s CPU budget for all 90 sequences. Score (maximize): mean F1 of the predicted base-pair set against the reference pairs, i.e. the harmonic mean of sensitivity and precision; sensitivity and precision are reported separately

## State of the art

No published value for this exact set. Measured during task construction: ViennaRNA 2.6.4 minimum-free-energy folding reaches mean F1 0.609, and the Nussinov-style seed 0.401.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [ViennaRNA Package 2.0](https://doi.org/10.1186/1748-7188-6-26)

Parameters: `{"cpu_seconds": 180}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `natural_science`, data `public_benchmark`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `biology`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict which bases pair with which in a non-coding RNA molecule from its sequence alone.

Why it matters: Secondary structure determines how structured RNAs function, and single-sequence prediction accuracy has plateaued around 60-70 % of pairs for thermodynamic models, so improvements are an active research goal.

- [Incorporating chemical modification constraints into a dynamic programming algorithm for prediction of RNA secondary structure](https://doi.org/10.1073/pnas.0401799101): Main research objective: Improve the accuracy of RNA secondary-structure prediction and measure it as sensitivity and precision of predicted base pairs. Task connection: The task scores predicted base-pair sets against reference structures with the same sensitivity and precision combined into F1.
- [ViennaRNA Package 2.0](https://doi.org/10.1186/1748-7188-6-26): Main research objective: Provide and improve algorithms for predicting RNA secondary structure from sequence. Task connection: Defines the state of practice the task measures against; the evaluator used it as an independent baseline (mean F1 0.609).

The cited papers' main goal is RNA secondary-structure prediction from sequence, exactly what the evaluator scores.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Reference pairings are the Rfam covariance-model consensus structures projected onto each sequence, keeping the canonical pairs whose columns are both present; they are comparative-sequence-analysis structures, not per-sequence experimental structures, and pseudoknots and non-canonical pairs are dropped. Two sequences per family at most, families describing viral elements or CRISPR loci excluded. ViennaRNA is installed in the suite environment, so a candidate may call a thermodynamic folding engine: the research content is improving on it (the evaluator measured ViennaRNA 2.6.4 minimum-free-energy folding at mean F1 0.609 on this set, against 0.401 for the seed).

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/bio_physics/rna/rfam_consensus_structures.json.gz`

## Sources

- paper: https://doi.org/10.1073/pnas.0401799101 (Mathews et al. (2004): incorporating chemical modification constraints into a dynamic programming algorithm for RNA secondary structure prediction, the problem and accuracy measures used here.)
- paper: https://doi.org/10.1186/1748-7188-6-26 (Lorenz et al. (2011), ViennaRNA Package 2.0: the thermodynamic folding algorithms that define the state of practice for single-sequence prediction.)
- archive: https://doi.org/10.1093/nar/gkaa1047 (Rfam (CC0): seed alignments and consensus secondary structures; this task uses Rfam.3d.seed.gz, the families with three-dimensional structures.)

## Limitations

- Reference structures are consensus structures from comparative sequence analysis projected onto individual sequences, not per-sequence experimental structures; pseudoknots, non-canonical pairs and pairs involving gapped columns are dropped.
- Only canonical nested pairs with a span of at least four nucleotides are scored, and at most two sequences per family are used, so the set is small (90 sequences, 47 families).
- ViennaRNA is available in the environment, so a candidate can call a thermodynamic folding engine; an improvement over that engine on this set is a method result on 47 families, not a general advance in RNA folding.
- Families describing viral elements or CRISPR loci are excluded by name, so the set is biased towards riboswitches, ribozymes, small RNAs and spliceosomal RNAs.
- Rfam's CURRENT directory tracks the latest release; the derived file's hash pins the exact sequences and structures used.
