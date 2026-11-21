# Software engineering: loop-invariant inference for the Code2Inv programs

**Source collection:** [Software engineering research: source-based research tasks](../../papers/science-software_engineering.md)\
**Task id:** `science-software-loop-invariants-code2inv`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Infer a loop invariant for each of the 124 valid Code2Inv C programs so that Z3 proves initiation, consecution and the post-condition. The procedure may query a verifier that returns counterexamples (50 queries per program); the score is the fraction of programs with a proven invariant.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

synthesize(problem, oracle) -> str. problem = {'name': str, 'c_source': str (declarations, pre-condition assignments, one while loop and a post-condition assert), 'variables': [names of the integer program variables], 'definitions': str (the Code2Inv SMT-LIB definitions of pre-f, trans-f and post-f over the variables, their primed next-state copies and SSA auxiliaries), 'max_queries': 50}. Return a loop invariant as one SMT-LIB Boolean term over the variables using integer literals (write negative numbers as (- n)), true, false and the operators and, or, not, =>, ite, =, <, <=, >, >=, +, -, * (at most one non-constant factor), div and mod (positive literal divisor), i.e. linear integer arithmetic. It is correct when Z3 proves initiation (pre-f implies it), consecution (it and trans-f imply it for the primed variables) and safety (it implies post-f), each within a fixed resource limit. oracle(invariant) performs the same check and returns {'valid': bool, 'failed': 'initiation', 'consecution', 'safety', 'syntax' or a budget message, 'state': a counterexample assignment of the variables or None, 'next': the successor assignment for consecution failures or None, 'queries_left': int, 'error': parse error if any}; each problem allows 50 queries and all queries share a fixed Z3 resource budget. Score: fraction of the 124 valid programs whose returned invariant is correct, maximised; a returned str that does not parse counts as not proven, a non-str answer makes the evaluation invalid. synthesize() is called once per program, all programs in one sandboxed subprocess (no files, network or subprocesses, so Z3 is not available there; 180 s CPU in total, 2 GB); it should be deterministic (seed any randomness, do not read clocks).

## State of the art

CLN2INV proves all 124 valid programs (LoopInvGen 104, Code2Inv 90) with one hour per program and unbounded solver calls (Ryan et al. 2020, Table 1); their invariants were not re-checked with this evaluator. The score is the fraction of programs proven, so 1.0 is its maximum: the optimum is attained and this instance cannot show a new state of the art.

Reference value: proven optimum `1.0` (reviewed 2026-10-10).

- [CLN2INV: Learning Loop Invariants with Continuous Logic Networks](https://arxiv.org/abs/1909.11542)

Parameters: `{"cpu_seconds": 180, "max_queries": 50}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `software_engineering`, data `public_benchmark`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `software_engineering`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Loop-invariant inference for program verification: find an inductive invariant that proves a loop program's post-condition, the main obstacle to automating deductive verification of programs with loops.

Why it matters: Verifying real programs requires invariants for every loop, and writing them by hand is the main manual cost of deductive verification; automatic invariant inference is a long-standing central problem with data-driven, learning-based and solver-based approaches compared on shared benchmarks.

- [Learning Loop Invariants for Program Verification](https://proceedings.neurips.cc/paper/2018/hash/65b1e92c585fd4c2159d5f33b5030ff2-Abstract.html): Main research objective: Infer loop invariants that let an SMT solver verify programs, learning from verifier feedback. Task connection: The task uses the same programs and verification conditions and gives the candidate a counterexample-returning verifier.
- [CLN2INV: Learning Loop Invariants with Continuous Logic Networks](https://arxiv.org/abs/1909.11542): Main research objective: Infer loop invariants from execution traces with continuous logic networks and verify them with an SMT solver. Task connection: The task scores the same 124 valid Code2Inv programs.
- [ICE: A Robust Framework for Learning Invariants](https://doi.org/10.1007/978-3-319-08867-9_5): Main research objective: Learn inductive invariants from positive, negative and implication counterexamples supplied by a verifier. Task connection: The task's oracle returns the initiation, consecution and safety counterexamples that ICE learners use.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Programs and verification conditions are those of the Code2Inv repository (PL-ML/code2inv at commit 336a7063; no licence stated, fetched and never redistributed). Bounded model checking (up to 8 loop steps, Z3) finds a post-condition violation in exactly 9 of the 133 programs (26, 27, 31, 32, 61, 62, 72, 75, 106), matching the 9 invalid programs reported by Ryan et al. (ICLR 2020); the remaining 124 are scored, hence recreated_verified. Candidate terms are parsed with a whitelist and restricted to linear arithmetic; the published tools had one hour per problem and unbounded solver calls, while here the candidate gets 50 verification queries per problem and a shared CPU budget.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:z3`
- `file:data/software/code2inv/problems.json`

## Sources

- paper: https://proceedings.neurips.cc/paper/2018/hash/65b1e92c585fd4c2159d5f33b5030ff2-Abstract.html (Si, Dai, Raghothaman, Naik and Song (NeurIPS 2018): Code2Inv, learning loop invariants with a verifier in the loop; introduces the 133-program benchmark.)
- repository: https://github.com/PL-ML/code2inv/tree/336a7063ae0d6313a6993dfb897766d01d5ba41a/benchmarks (Pinned Code2Inv benchmark directory: C programs (C_instances/c) and SMT-LIB verification conditions (C_instances/c_smt2); no licence stated.)
- paper: https://arxiv.org/abs/1909.11542 (Ryan, Wong, Yao, Gu and Jana (ICLR 2020): CLN2INV; removes 9 invalid programs and solves all 124 others (LoopInvGen 104, Code2Inv 90), one hour per problem.)
- paper: https://doi.org/10.1007/978-3-319-08867-9_5 (Garg, Loding, Madhusudan and Neider (CAV 2014): ICE learning, invariant inference from a teacher's implication and counterexamples.)
- paper: https://doi.org/10.1007/3-540-45251-6_29 (Flanagan and Leino (FME 2001): Houdini, the template-based invariant pruning used by the seed.)

## Limitations

- Of the 133 programs, bounded model checking here finds 9 whose post-condition can fail (26, 27, 31, 32, 61, 62, 72, 75, 106), the count reported by Ryan et al. (2020); those 9 are excluded, so scores use 124 programs.
- Five of the 124 post-conditions hold at every loop exit, so the invariant 'true' already proves them; many others need only simple bounds.
- Correctness follows the benchmark's own SMT-LIB verification conditions, checked by Z3 with a fixed resource limit; the C programs themselves are not executed, so an error in an encoding would carry over.
- Invariants are restricted to linear integer arithmetic and the candidate has 50 verifier queries per program and 180 s of CPU in total, while published tools had one hour per program and unbounded solver calls (LoopInvGen averaged about 3,600 calls); published counts are therefore not directly comparable.
- The programs are public and solved invariants circulate; embedding them is outside the contract but cannot be checked automatically.
- Results depend on the Z3 version only where a check reaches the resource limit; this run used z3-solver 5.1.0.
