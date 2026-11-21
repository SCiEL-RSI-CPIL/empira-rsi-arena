# Software engineering: spectrum-based fault localisation of seeded faults in three CPython modules

**Source collection:** [Software engineering research: source-based research tasks](../../papers/science-software_engineering.md)\
**Task id:** `science-software-fault-localization-seeded`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Rank the code blocks of a faulty module by suspiciousness using only its program spectrum: for each block, how many failing and passing tests execute it. The 532 faults are first-order mutants of textwrap, urllib.parse and fractions; the score is the mean EXAM (rank of the faulty block divided by the number of blocks).

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

localize(spectrum) -> list of float. spectrum = {'subject': 'textwrap', 'urllib.parse' or 'fractions', 'blocks': n, 'failed': F, 'passed': P, 'ef': [n ints], 'ep': [n ints]} describes one faulty version of the module: F of its pool tests fail (their outcome differs from the correct module's) and P pass, and ef[b] (ep[b]) is the number of failing (passing) tests that execute block b, where blocks are the evaluator's block probes (function entries, branch outcomes, loop bodies and exits, handlers) in a fixed pseudo-random order per fault, so an index does not denote the same block in different faults. Return n finite numbers, a suspiciousness per block (higher means more likely faulty). Score: mean EXAM over the 532 faults, i.e. the faulty block's rank (ties count as their average rank) divided by n, minimised. localize() is called once per fault, all faults in one sandboxed subprocess (no files, network or subprocesses; 120 s CPU in total, 2 GB); it should be deterministic (seed any randomness, do not read clocks).

## State of the art

No published value exists for these faults. Computed with this evaluator: Tarantula 0.0506, Ochiai 0.0268 and DStar(2) 0.0248 mean EXAM.

Reference value: no documented value for this finite instance (reviewed 2026-10-11).

- [The DStar Method for Effective Software Fault Localization](https://doi.org/10.1109/TR.2013.2285319)

Parameters: `{"cpu_seconds": 120}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `software_engineering`, data `public_benchmark`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `software_engineering`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Fault localisation: given a faulty program and the outcomes and coverage of its tests, point developers to the code most likely responsible for the failures.

Why it matters: Locating faults is one of the most expensive debugging activities; spectrum-based techniques that rank code by how strongly its execution correlates with failing tests are among the most studied approaches in software engineering and are used by automated repair tools.

- [Empirical evaluation of the tarantula automatic fault-localization technique](https://doi.org/10.1145/1101908.1101949): Main research objective: Locate faulty statements by ranking them with a suspiciousness computed from passing and failing test coverage. Task connection: The task asks for such a ranking from test coverage spectra; Tarantula is the seed.
- [On the Accuracy of Spectrum-based Fault Localization](https://doi.org/10.1109/TAIC.PART.2007.13): Main research objective: Improve and assess the accuracy of spectrum-based fault localisation with block hit spectra. Task connection: The task uses block hit spectra and the same rank-based accuracy measure.
- [Evolving Human Competitive Spectra-Based Fault Localisation Techniques](https://doi.org/10.1007/978-3-642-33119-0_18): Main research objective: Find better risk-evaluation formulas for spectrum-based fault localisation by genetic programming. Task connection: The task is the same search for a ranking function, scored on seeded faults.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Faults are the evaluator's first-order mutants of CPython 3.12.14's textwrap, urllib.parse and fractions (PSF licence), one fault per version, detected by a seeded 240-call test pool; spectra are block hit counts from the evaluator's own probes, and the faulty block is the statement-level block that contains the mutated statement. The literature mostly uses statement-level spectra of real or hand-seeded faults (Siemens, Defects4J), so absolute EXAM values are not comparable; Tarantula, Ochiai and DStar(2) are computed on the same data and reported in the details.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/software/sbfl/faults.json`

## Sources

- paper: https://doi.org/10.1145/1101908.1101949 (Jones and Harrold (ASE 2005): the Tarantula fault-localisation technique and its evaluation by the share of code examined (the seed).)
- paper: https://doi.org/10.1109/TAIC.PART.2007.13 (Abreu, Zoeteweij and van Gemund (TAIC PART 2007): accuracy of spectrum-based fault localisation with block hit spectra; Ochiai coefficient.)
- paper: https://doi.org/10.1109/TR.2013.2285319 (Wong, Debroy, Gao and Li (IEEE Trans. Reliability 2014): the DStar formula.)
- paper: https://doi.org/10.1007/978-3-642-33119-0_18 (Yoo (SSBSE 2012): evolving risk-evaluation formulas with genetic programming, the program-search setting of this task.)
- paper: https://doi.org/10.1109/ICSE.2017.62 (Pearson et al. (ICSE 2017): results on artificial faults do not carry over to real faults; the main caveat of this task.)
- archive: https://www.python.org/downloads/release/python-31214/ (CPython 3.12.14 source release (PSF licence): textwrap, urllib.parse and fractions.)

## Limitations

- Faults are artificial first-order mutants, one per version; Pearson et al. (ICSE 2017) found that rankings of fault-localisation techniques on artificial faults do not predict their rankings on real faults.
- Spectra are block hit counts from this evaluator's probes and a seeded 240-call test pool per module, not statement coverage of a project's own test suite; the faulty block is the statement-level block containing the mutated statement (faults on one-line compound statements are left out), so EXAM values are not comparable with published ones.
- Undetected mutants and mutants that break the import are excluded; textwrap contributes 110 faults, urllib.parse 183 and fractions 239.
- Reference values computed here: Tarantula (the seed) 0.0506, Ochiai 0.0268 and DStar(2) 0.0248 mean EXAM; ranking each faulty block first among blocks with an identical spectrum would give 0.0087, a bound for formulas of the counts that no single formula need reach.
- Blocks are shown in a fixed pseudo-random order per fault, so position carries no information, but the faults are public and a search can still overfit this fault set.
