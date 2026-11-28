# Biology: protein multiple-alignment accuracy on BAliBASE reference alignments

**Source collection:** [Biology research: source-based research tasks](../../papers/science-biology_research.md)\
**Task id:** `science-biology-balibase-protein-alignment`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Align the unaligned protein sequences of 26 BAliBASE reference-1 test cases, spanning very divergent and medium-divergence families. The evaluator checks that each alignment is a valid gapped version of the input sequences and scores the sum-of-pairs agreement with the structure-based reference alignment on its annotated core blocks.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

align(sequences) -> list of aligned rows, one per input sequence, all of the same length (at most 6 times the longest input sequence plus 10 columns), each row equal to its input sequence with '-' gap characters inserted (upper-case amino acids). sequences is a list of 4 to 10 unaligned protein sequences of a BAliBASE reference-1 test case. align is called once per test case in a sandboxed subprocess with a 240 s CPU budget for all 26 cases. Score (maximize): mean sum-of-pairs score over the cases, i.e. the fraction of residue pairs that the structure-based reference alignment places in the same core-block column and the candidate also aligns; the total-column score is reported alongside

## State of the art

No published value for this 26-case subset and local scorer. For context, the progressive-alignment seed reaches mean SP 0.640 (0.400 on the very divergent RV11 cases, 0.816 on RV12) and mean TC 0.480; published aligners report substantially higher SP on the full reference-1 sets, so there is room to improve.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [A comprehensive comparison of multiple sequence alignment programs](https://doi.org/10.1093/nar/27.13.2682)

Parameters: `{"cpu_seconds": 240}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `natural_science`, data `public_benchmark`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `biology`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Align homologous protein sequences so that structurally equivalent residues occupy the same column, especially for families below 20 % identity.

Why it matters: Multiple sequence alignment underlies homology modelling, phylogenetics and functional-site prediction; alignment accuracy on structure-based reference alignments is the standard way the field measures progress, and divergent families remain unsolved.

- [A comprehensive comparison of multiple sequence alignment programs](https://doi.org/10.1093/nar/27.13.2682): Main research objective: Measure how accurately alignment programs reproduce structure-based reference alignments, and define the SP and TC scores for that purpose. Task connection: The task scores candidate aligners with the same SP and TC definitions on BAliBASE reference-1 cases.
- [MAFFT: a novel method for rapid multiple sequence alignment based on fast Fourier transform](https://doi.org/10.1093/nar/gkf436): Main research objective: Develop a more accurate and faster multiple sequence alignment method. Task connection: The task asks for exactly this kind of method and scores it on reference alignments.

The cited papers' main goal is multiple sequence alignment accuracy, which the evaluator measures directly against structure-based references.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Reference alignments and core-block annotations come from the BAliBASE 3.0 archive (reference set 1: RV11 very divergent, RV12 medium divergence); 26 cases with at most 10 sequences, at most 300 reference columns and at least 43 core-block columns are used (ten further cases within the size limits have fewer than 40 core-block columns and are left out) so that a CPU-only aligner finishes inside the budget. The SP and TC scores are an independent implementation of the definitions of Thompson et al. (1999) restricted to core blocks, not a port of the bundled bali_score program, and the reference alignments score exactly 1.0 under it (checked in the tests). Scores on this subset are not comparable with published whole-benchmark averages.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/bio_physics/align/balibase_rv11_rv12.json.gz`

## Sources

- paper: https://doi.org/10.1093/nar/27.13.2682 (Thompson, Plewniak and Poch (1999): comparison of multiple alignment programs on reference alignments; defines the sum-of-pairs and total-column scores on core blocks implemented here.)
- paper: https://doi.org/10.1002/prot.20527 (Thompson, Koehl, Ripp and Poch (2005): BAliBASE 3.0, the structure-based reference alignments and core-block annotations used as ground truth.)
- paper: https://doi.org/10.1093/nar/gkf436 (Katoh et al. (2002), MAFFT: an alignment method whose main goal is accurate and fast multiple sequence alignment, the problem this task scores.)
- benchmark: https://www.lbgi.fr/balibase/ (BAliBASE download page; the archive BAliBASE_R1-5.tar.gz (reference sets RV11 and RV12 of this task) states no licence and is downloaded by the prepare script rather than redistributed.)

## Limitations

- Only 26 of the 82 reference-1 cases are used (at most 10 sequences and 300 reference columns each, and at least 43 core-block columns; ten further cases within the size limits, each with fewer than 40 core-block columns, are not used) so that a CPU-only aligner finishes within the budget; averages are therefore not comparable with published whole-benchmark numbers.
- SP and TC are an independent implementation of the published definitions restricted to core blocks; it scores the reference alignments at exactly 1.0 but is not byte-compatible with the bundled bali_score program.
- Reference alignments are themselves structure-based estimates of homology, and core blocks cover only part of each alignment.
- The references are public, so search feedback can overfit this fixed subset; a candidate aligner should be checked on held-out families before any claim of improved accuracy.
