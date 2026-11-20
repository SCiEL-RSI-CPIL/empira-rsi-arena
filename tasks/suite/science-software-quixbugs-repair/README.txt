# Software engineering: test-based repair of the QuixBugs Python programs

**Source collection:** [Software engineering research: source-based research tasks](../../papers/science-software_engineering.md)\
**Task id:** `science-software-quixbugs-repair`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a general repair procedure that receives a buggy QuixBugs Python program with a few visible tests and returns a patch of at most two statements. A patch counts only when it also passes held-out QuixBugs tests and generated tests whose expected outputs come from the reference fix.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

repair(task) -> str. task = {'name': str, 'entry': str, 'source': str, 'tests': [{'input': [arguments], 'expected': value, 'tolerance': float (only for sqrt)}, ...], 'max_statement_edits': 2}: 'source' is a buggy QuixBugs Python program defining the function 'entry'; 'tests' are its visible tests, at least one of which fails. Return the patched program as a str of at most 20000 characters that differs from 'source' in at most 2 statements (replaced, inserted or deleted statements of the parsed program; formatting and comments do not count). A test passes when entry(*input) returns the expected value after JSON normalisation (generators are consumed into lists, tuples become lists) within 1 s of CPU time, or a number within 'tolerance' of it. Score: fraction of the 31 programs whose patch passes its visible tests, its remaining QuixBugs tests and 10 to 40 generated held-out tests whose expected values come from the reference fix (higher is better); a patch passing only some of them counts as overfitting, not repaired. repair() runs once per program in a sandboxed subprocess (no files, network or subprocesses; 4 s CPU, 1 GB) and may execute the given program and tests itself; it should be deterministic. The procedure must be general: remembered program-specific fixes are outside the contract.

## State of the art

No published result uses this 31-program subset with these held-out tests. Ye et al. (2021) report repair tools on all 40 Java QuixBugs programs, and language-model studies report counts on all 40 Python programs under other correctness criteria.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [A comprehensive study of automatic program repair on the QuixBugs benchmark](https://doi.org/10.1016/j.jss.2020.110825)

Parameters: `{"cpu_seconds": 4}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `software_engineering`, data `public_benchmark`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `software_engineering`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Automated program repair: given a faulty program and tests that expose the fault, produce a small patch that makes the program correct, not merely one that passes the given tests.

Why it matters: Debugging consumes a large share of development effort; test-suite-based repair is one of the most studied routes to automating it, and patch overfitting to the available tests is its central open difficulty.

- [GenProg: A Generic Method for Automatic Software Repair](https://doi.org/10.1109/TSE.2011.104): Main research objective: Automatically repair defects in real programs by searching for patches that make the failing tests pass while keeping the passing ones. Task connection: The task asks for the same kind of procedure: a test-guided repair of a buggy program.
- [Is the cure worse than the disease? Overfitting in automated program repair](https://doi.org/10.1145/2786805.2786825): Main research objective: Measure how often automatically generated patches overfit the tests used for repair, using independent held-out tests. Task connection: The task separates repaired from overfitting patches with held-out tests in the same way.
- [A comprehensive study of automatic program repair on the QuixBugs benchmark](https://doi.org/10.1016/j.jss.2020.110825): Main research objective: Assess how well repair tools fix the QuixBugs programs and how many of their plausible patches are actually correct. Task connection: The task uses the QuixBugs programs and judges correctness beyond the original tests.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- QuixBugs at commit 4257f44b (MIT): the 31 Python programs with JSON test cases; the nine graph programs (pytest-only tests) and two very slow test cases are not used. Visible tests are the even-indexed QuixBugs tests plus the first failing one; the other QuixBugs tests and up to 40 generated tests per program (seeded inputs within the documented preconditions, expected values from the reference fix, computed at preparation time) are held out and never reach the candidate. Correctness is judged by these tests, not by the manual patch inspection used in the literature, and the 2-statement edit bound is local.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/software/quixbugs/repair.json`

## Sources

- paper: https://doi.org/10.1145/3135932.3135941 (Lin, Koppel, Chen and Solar-Lezama (SPLASH Companion 2017): the QuixBugs benchmark of 40 programs with one-line bugs in Python and Java.)
- repository: https://github.com/jkoppel/QuixBugs/tree/4257f44b0ff1181dedaedee6a447e133219fcebf (Pinned QuixBugs commit (MIT licence): python_programs, correct_python_programs (used only at preparation time) and json_testcases.)
- paper: https://doi.org/10.1109/TSE.2011.104 (Le Goues, Nguyen, Forrest and Weimer (TSE 2012): GenProg, test-suite-based generate-and-validate repair.)
- paper: https://doi.org/10.1145/2786805.2786825 (Smith, Barr, Le Goues and Brun (ESEC/FSE 2015): patches that pass the repair tests but fail held-out tests (overfitting).)
- paper: https://doi.org/10.1016/j.jss.2020.110825 (Ye, Martinez, Durieux and Monperrus (JSS 2021): repair tools on QuixBugs; plausible patches checked for overfitting with generated tests.)

## Limitations

- Uses the 31 Python programs with JSON tests; the nine graph programs are excluded, so results reported on all 40 programs are not comparable, nor are results of tools that call language models.
- Correctness is decided by tests: the visible tests, the remaining QuixBugs tests and 10 to 40 generated inputs per program checked against the reference fix. A semantically different but test-equivalent patch counts as a repair, and a wrong patch that agrees on all these inputs would also count; there is no manual patch review.
- Generated inputs respect the documented preconditions and avoid inputs whose correct output is not unique; inputs on which the reference fix itself loops (some wrap cases) are excluded.
- The programs and their fixes are public, so a candidate could embed remembered fixes; the contract excludes this but it cannot be checked automatically.
- Each repair() call gets 4 s of CPU time in a Python sandbox, far less than the hours given to repair tools in the literature.
- The suite's evaluator template imports the candidate module in the evaluator process before scoring; every scored call runs in the sandbox, which cannot read the held-out data, but deliberately hostile import-time code could. Untrusted evaluations should run in a container without the prepared files.
