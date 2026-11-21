# Biology: phylogenetic topology recovery from simulated sequences

**Source collection:** [Biology research: source-based research tasks](../../papers/science-biology_research.md)\
**Task id:** `science-biology-phylogeny-topology-recovery`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Reconstruct the unrooted tree topology of twenty simulated instances with 8 to 32 taxa and 250 to 2000 sites, generated inside the evaluator under the Jukes-Cantor model with hidden random topologies and branch lengths. The score is the mean normalized Robinson-Foulds distance to the generating trees.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

reconstruct(sequences) -> Newick string over the labels t0..t(n-1) (branch lengths optional, trailing semicolon optional, every taxon exactly once). sequences maps those labels to equal-length DNA strings over ACGT simulated along an unknown tree. reconstruct is called once per instance in a sandboxed subprocess with a 180 s CPU budget for all 20 instances (8 to 32 taxa, 250 to 2000 sites). Score (minimize): mean normalized Robinson-Foulds distance between the non-trivial unrooted splits of the returned tree and those of the generating tree; 0 means recovering every topology exactly

## State of the art

No published value for this instance set. The neighbour-joining seed reaches mean normalized Robinson-Foulds 0.177 and recovers 5 of the 20 topologies exactly; maximum-likelihood search is expected to do better, which is the intended research content.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Performance of phylogenetic methods in simulation](https://doi.org/10.2307/2413481)

Parameters: `{"cpu_seconds": 180}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `natural_science`, data `synthetic`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `biology`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Recover the evolutionary tree relating a set of sequences from the sequences alone.

Why it matters: Tree reconstruction underpins comparative biology, and its accuracy under known generating trees is the standard way methods such as neighbour joining and maximum likelihood have been compared since the 1990s.

- [The neighbor-joining method: a new method for reconstructing phylogenetic trees](https://doi.org/10.1093/oxfordjournals.molbev.a040454): Main research objective: Develop a method that reconstructs the correct phylogenetic tree from sequence data. Task connection: The task scores exactly this reconstruction problem; the seed program is a neighbour-joining implementation.
- [Performance of phylogenetic methods in simulation](https://doi.org/10.2307/2413481): Main research objective: Measure how often phylogenetic methods recover the true tree from sequences simulated along known trees. Task connection: The task follows this protocol: simulated sequences, hidden true topology, distance to it as the score.

Both papers' main research goal is phylogenetic tree reconstruction and its accuracy, which the evaluator measures with the Robinson-Foulds distance to the generating tree.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Instances are generated inside the evaluator from uniform draws of a seeded PCG64 bit generator (random unrooted topology, exponential branch lengths, Jukes-Cantor substitutions), so they are identical on every run and the true topology is never shown to the candidate. Jukes-Cantor is a deliberately simple model: no rate variation, no indels, no among-site heterogeneity, and the branch-length regimes are a local choice. Exact recovery is not always possible from finite sequence data, so the attainable minimum of this score is unknown.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://doi.org/10.1093/oxfordjournals.molbev.a040454 (Saitou and Nei (1987): the neighbour-joining method for reconstructing phylogenetic trees, the problem this task scores and the basis of the seed program.)
- paper: https://doi.org/10.2307/2413481 (Huelsenbeck (1995), Performance of phylogenetic methods in simulation: evaluating tree-reconstruction accuracy on sequences simulated along known trees, the protocol followed here.)
- paper: https://doi.org/10.1016/0025-5564(81)90043-2 (Robinson and Foulds (1981): the split-based tree distance used as the score.)
- paper: https://doi.org/10.1093/sysbio/syq010 (Guindon et al. (2010), PhyML 3.0: maximum-likelihood tree search, the state of practice a candidate would have to match or beat.)

## Limitations

- Sequences are simulated under Jukes-Cantor with no rate heterogeneity, no indels and no model misspecification, so the task measures reconstruction under an idealized model rather than on real data.
- Instances, topologies and branch-length regimes are a local choice (20 instances, 8 to 32 taxa); results do not transfer automatically to empirical alignments.
- Exact recovery is not guaranteed to be attainable from finite data, so the minimum achievable score is unknown and the seed's value is only a baseline.
- Branch lengths of the returned tree are ignored: only the unrooted topology is scored.
