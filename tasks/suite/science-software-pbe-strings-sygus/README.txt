# Software engineering: programming by example for string transformations (SyGuS-Comp PBE strings)

**Source collection:** [Software engineering research: source-based research tasks](../../papers/science-software_engineering.md)\
**Task id:** `science-software-pbe-strings-sygus`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a synthesiser that, from a few input-output examples and a grammar, returns a string-manipulation program (phone numbers, names, initials, university addresses) that also reproduces held-out examples of the same task. The score is the fraction of 27 SyGuS-Comp tasks solved.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

synthesize(task) -> str. task = {'name': str, 'function': str, 'params': [[name, sort], ...], 'sort': 'String', 'start': nonterminal, 'grammar': [[nonterminal, sort, [productions]], ...], 'examples': [[[argument strings], output string], ...]} is a SyGuS PBE string problem: productions are parameter names, literals (string literals as {'str': value}, integers, true, false), other nonterminals or operator lists such as ['str.substr', 'ntString', 'ntInt', 'ntInt']. Return one SMT-LIB term (an S-expression string; string literals in double quotes with "" for a quote, negative numbers as -1 or (- 1)) that the grammar derives from 'start'. Operations follow SMT-LIB 2.6 string semantics (str.++, str.len, str.at, str.substr, str.indexof, str.replace, str.to.int, int.to.str, str.prefixof, str.suffixof, str.contains, +, -, =, ite, with the edge cases of the standard, e.g. out-of-range str.at and str.substr give "" and str.to.int of a non-numeral gives -1). A task is solved when the term reproduces the given examples and the held-out examples of the benchmark's '-long' variant. Score: fraction of the 27 tasks solved, maximised; a term that does not parse or leaves the grammar counts as unsolved, a non-str answer makes the evaluation invalid. synthesize() is called once per task, all tasks in one sandboxed subprocess (no files, network or subprocesses; 120 s CPU in total, 2 GB); it should be deterministic (seed any randomness, do not read clocks).

## State of the art

No published value measures generalisation to the held-out '-long' examples of these 27 tasks; the bottom-up enumeration seed solves 8 (computed here).

Reference value: no documented value for this finite instance (reviewed 2026-10-11).

- [Overfitting in Synthesis: Theory and Practice](https://doi.org/10.1007/978-3-030-25540-4_17)

Parameters: `{"cpu_seconds": 120}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `software_engineering`, data `public_benchmark`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `software_engineering`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Programming by example: synthesise a program in a given language that maps the example inputs to their outputs and generalises to the user's intended transformation.

Why it matters: Most spreadsheet and data-wrangling users cannot program; synthesis from examples (FlashFill and its successors) automates such transformations, and finding a program that generalises from a few examples, rather than one that merely fits them, is the central difficulty.

- [Automating string processing in spreadsheets using input-output examples](https://doi.org/10.1145/1926385.1926423): Main research objective: Synthesise string-manipulation programs from a few input-output examples for spreadsheet users. Task connection: The task asks for such programs for string transformations of the same kind.
- [Scaling Enumerative Program Synthesis via Divide and Conquer](https://doi.org/10.1007/978-3-662-54577-5_18): Main research objective: Make enumerative syntax-guided synthesis scale to larger programs and example sets. Task connection: The task's problems are SyGuS PBE string tasks of the kind EUSolver was evaluated on.
- [Overfitting in Synthesis: Theory and Practice](https://doi.org/10.1007/978-3-030-25540-4_17): Main research objective: Understand and reduce overfitting of synthesised programs to the given examples. Task connection: The task scores generalisation to held-out examples, i.e. the absence of overfitting.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Tasks are the 27 problems of SyGuS-Org/benchmarks lib/PBE_SLIA_Track/from_2018 (commit 13c8deb6; no licence stated, fetched and never redistributed) that have a '-long' variant: the base file's examples and grammar are given, the long variant's other examples are held out (8 to 94 per task; no example conflicts with the base file). The competition scored consistency with the given examples and solving time; generalisation to held-out examples is this dataset's criterion. The interpreter agrees with Z3's string theory on 400 random terms and the standard's edge cases (tests).

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/software/pbe/tasks.json`

## Sources

- paper: https://doi.org/10.1145/1926385.1926423 (Gulwani (POPL 2011): FlashFill, synthesising string transformations in spreadsheets from input-output examples.)
- paper: https://doi.org/10.1109/FMCAD.2013.6679385 (Alur et al. (FMCAD 2013): syntax-guided synthesis, the problem format (grammar plus specification) of these tasks.)
- paper: https://doi.org/10.1007/978-3-662-54577-5_18 (Alur, Radhakrishna and Udupa (TACAS 2017): EUSolver, enumerative synthesis with divide and conquer, evaluated on SyGuS PBE string tasks.)
- paper: https://doi.org/10.1007/978-3-030-25540-4_17 (Padhi, Millstein, Nori and Sharma (CAV 2019): overfitting in synthesis, programs that fit the examples but not the intended function.)
- benchmark: https://github.com/SyGuS-Org/benchmarks/tree/13c8deb68a873635879c9a69bc78caebd340f646/lib/PBE_SLIA_Track/from_2018 (Pinned SyGuS benchmarks (no licence stated): the base and '-long' files of the 27 tasks.)

## Limitations

- Only 27 tasks, from a few families (11 phone-number tasks, 4 name-combination tasks, 6 university-address tasks and 6 others), so scores move in steps of 1/27 and families are correlated.
- Correctness is judged on examples: the given ones plus 8 to 94 held-out examples of the '-long' variant per task; a program that agrees on all of them but differs elsewhere counts as solved.
- SyGuS-Comp scored consistency with the given examples and solving time, not generalisation, so published solved counts are not comparable; the benchmark repository notes that errors and duplicates may exist.
- The bottom-up enumeration seed solves 8 of 27 within its budget; the tasks are public and solutions could be remembered, which the contract excludes but cannot check.
- The interpreter implements the SMT-LIB 2.6 string semantics used by SyGuS-Comp 2018 and was cross-checked against Z3 on random terms; a solver with different edge-case semantics could disagree on unusual programs.
