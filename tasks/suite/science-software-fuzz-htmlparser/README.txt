# Software engineering: coverage-guided fuzzing of CPython's html.parser

**Source collection:** [Software engineering research: source-based research tasks](../../papers/science-software_engineering.md)\
**Task id:** `science-software-fuzz-htmlparser`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Generate at most 1,000 input strings that drive html.parser.HTMLParser with _markupbase (CPython 3.12.14) through as many code blocks as possible. The score is the number of distinct instrumented blocks (function entries, branch outcomes, loop exits, handlers) that the inputs reach.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fuzz(target) -> list of str. target = {'subject': str, 'entry': str (how one input is executed), 'files': {path: source} (the subject's CPython 3.12.14 files as the evaluator loads them), 'instrumented': [paths whose coverage counts], 'max_inputs': 1000, 'max_chars': 2000}. Return at most max_inputs strings of at most max_chars characters. The evaluator imports the files under a private package name, inserts a numbered probe at the start of every block inside the functions of the instrumented files (function entry; if and else branches, including an empty else; loop bodies and normal loop exits; try bodies, else and finally blocks; exception handlers; with bodies; match cases; both branches of conditional expressions), runs the entry on every input (exceptions are expected and ignored; an input is cut off after 1,000,000 steps, where a step is a function entry or a loop iteration inside a function of the instrumented files, so the cut does not depend on the machine, or after 5 s of CPU time, a safety net for time spent in C code; probes reached before the cut count) and counts the distinct probes reached. Entries: tomllib -> tomllib.loads(text); re -> re._compiler.compile(text, 0), i.e. regular-expression parsing and compilation; html.parser -> for convert_charrefs in (True, False): HTMLParser(convert_charrefs).feed(text) then close(); plistlib -> plistlib.loads(text.encode('latin-1')), so each character U+0000-U+00FF is one byte and the XML or binary format is detected from the bytes (characters above U+00FF only raise the encoding error). Score: number of distinct probes reached, maximised; replaying all inputs may take at most 120 s of CPU time, otherwise the evaluation is invalid. fuzz() runs in a sandboxed subprocess (no files, network or subprocesses; 60 s CPU, 2 GB) and may load and run the given files itself, for example to trace coverage; it should be deterministic (seed any randomness, do not read clocks).

## State of the art

No published value exists for this instrumentation. As a reference, the 525 distinct string constants of CPython's test_htmlparser.py reach 172 of the 372 probes (computed with this evaluator).

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Coverage-based Greybox Fuzzing as Markov Chain](https://doi.org/10.1145/2976749.2978428)

Parameters: `{"subject": "html.parser", "cpu_seconds": 60}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `software_engineering`, data `public_benchmark`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `software_engineering`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Automated test input generation (fuzzing) for parsers: produce inputs that exercise as much of a parser's logic as possible, since unexercised code cannot reveal its faults.

Why it matters: Parsers process untrusted input and are a main source of security and robustness defects; coverage-guided and grammar-aware fuzzing are among the most widely used and studied testing techniques.

- [Coverage-based Greybox Fuzzing as Markov Chain](https://doi.org/10.1145/2976749.2978428): Main research objective: Make coverage-guided greybox fuzzing reach more program paths with the same number of executions. Task connection: The task rewards inputs by the program coverage they reach.
- [Grammar-based whitebox fuzzing](https://doi.org/10.1145/1375581.1375607): Main research objective: Generate well-formed inputs that reach deep into programs with structured input such as parsers. Task connection: The targets are parsers, where input structure decides which code is reached.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Targets are pure-Python CPython 3.12.14 modules (PSF licence) imported from source under a private package name; html/parser.py is patched to import that copy of _markupbase. Coverage is counted with the evaluator's own block probes (no coverage.py), and the corpus is bounded by size instead of the wall-clock budgets of fuzzing benchmarks; probe counts depend on these sources, not on the interpreter's line events.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/software/pymod/fuzz-html.parser.json`

## Sources

- paper: https://doi.org/10.1145/2976749.2978428 (Bohme, Pham and Roychoudhury (CCS 2016): coverage-based greybox fuzzing, where reaching new coverage guides input generation.)
- paper: https://doi.org/10.1145/1375581.1375607 (Godefroid, Kiezun and Levin (PLDI 2008): grammar-based fuzzing of parsers to reach code beyond the lexer.)
- paper: https://doi.org/10.1145/3243734.3243804 (Klees, Ruef, Cooper, Wei and Hicks (CCS 2018): how fuzzers are evaluated, including coverage as a measure.)
- archive: https://www.python.org/downloads/release/python-31214/ (CPython 3.12.14 source release (PSF licence): the pure-Python target modules and the test inputs used as references.)

## Limitations

- Coverage is a proxy: no faults are seeded and crashes are not scored, while fuzzing research also measures bugs found (Klees et al. 2018).
- Block probes are this evaluator's own instrumentation of the pinned sources (no coverage.py or compiler edges), so counts are not comparable with other coverage tools; some probes are unreachable.
- The corpus is bounded by size (1,000 strings of at most 2,000 characters) and generation by 60 s of CPU time instead of the hours-long wall-clock budgets of fuzzing benchmarks.
- Reference computed here: the 525 distinct string constants of CPython's test_htmlparser.py reach 172 of the 372 probes; the random-string seed reaches fewer.
- The target is pure Python executed by CPython; results say nothing about memory-safety bugs in C code.
