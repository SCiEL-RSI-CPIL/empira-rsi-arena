# Frontier-CS Polyominoes packing

**Paper:** CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery (https://arxiv.org/pdf/2604.01658)\
**Task id:** `coral-frontier-cs-polyominoes-packing`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Provide a C++17 program (as the SOLUTION_CPP string) that chooses a W x H rectangle and places 100 to 10000 polyominoes, with optional rotation and reflection, on distinct cells inside it; scored by the official checker's mean of total cells / (W*H) x 100 over 70 Frontier-CS test cases, failed cases scoring 0 (higher is better).

Benchmark named in the paper: Frontier-CS. Paper metric: packing score.

## Candidate interface

SOLUTION_CPP: str, the source of a complete C++17 program compiled with GNU g++ -O2 -pipe -std=gnu++17 (bits/stdc++.h available) and run once per test case with 2 s CPU, 4 s wall-clock and 256 MiB memory. Input (stdin): n (100..10000) polyominoes; for each, k (1..10) and k lines 'x y' of 4-connected cells in a local frame (coordinates may be negative, holes allowed). Output (stdout): 'W H', then one line 'X Y R F' per polyomino in input order: each cell (x, y) is reflected across the y-axis if F = 1 ((x, y) -> (-x, y)), then rotated R times by 90 degrees clockwise ((x, y) -> (y, -x)), then translated by (X, Y); all transformed cells must be distinct and inside [0, W) x [0, H). The checker scores a case by total cells / (W * H). Score (maximize): mean over the 70 Frontier-CS test cases of the official checker's ratio x 100; a case with a rejected output, a runtime error or an exceeded limit scores 0, and a compilation error makes the candidate invalid

Parameters: `{"n_cases": 70, "time_limit_s": 2.0, "memory_mb": 256, "workers": 4}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `combinatorial_optimisation`, data `public_benchmark`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Source task: CORAL examples/frontier_cs_algo/0 at commit 0123dfb939b35228cf2c1fde224cd0561e727408 (Frontier-CS algorithmic problem 0, C++17, 70 cases, 2 s and 256 MB per case). Its grader calls the Frontier-CS judge service (go-judge in Docker) through the frontier_cs package, installed unpinned; this task runs the same ingredients locally from Frontier-CS commit e6ef46eb4848d99c3c88505d0084284b291bba76 (MIT): the 70 test inputs of algorithmic/problems/0/testdata and the unchanged testlib checker chk.cc (last changed 2025-12-15; testdata last changed 2026-07-26 UTC, #171, before CORAL's pinned commit), fetched by scripts/prepare_or_csp.py and locked in catalog/or_csp-data-lock.json.
- Harness differences from the Frontier-CS judge: compilation with a local GNU g++ (-O2 -pipe -std=gnu++17; the judge's -static -s are dropped because macOS cannot link statically); per-case limits applied by a small launcher with setrlimit (2 s CPU, killed at 4 s wall clock as the judge's clockLimit = 2 x time, output capped at 128 MiB) instead of go-judge's sandbox; memory is enforced as an address-space limit on Linux but only checked after the run from the peak resident set on macOS, and the stack limit is capped by the operating system's hard limit (64 MiB on macOS); up to four cases run in parallel. Scores of time-bounded heuristics depend on the machine. The candidate's Python file is evaluated in a sandboxed child process only. The program runs with the user's privileges in an empty working directory; the CPU time of every process it starts counts against the 2 s, and every such process is killed when its case ends (Linux); the checker is built and the inputs are read (and compared with the data lock) before the first case, and all outputs are checked after the last case from fresh copies. There is no file-system isolation: a program can alter the files of cases that run alongside it and any file the user can write.
- Score: the judge engine's rule (algorithmic/judge/src/judge_engine.js): a case scores the 'Ratio:' the checker prints when it accepts the output (total cells / (W x H)), otherwise 0, and the problem score is the mean ratio x 100. The checker's answer-file argument receives the placeholder 'zzz' that every testdata/*.ans file contains; chk.cc never reads it.
- Contract: the candidate is a Python module whose SOLUTION_CPP string holds the C++ program (CORAL's agent edits solution.cpp directly). CORAL's copy of the statement still says W = H; the Frontier-CS statement dropped that on 2026-05-28 and the checker never enforced it, so this task follows the checker.
- Verification: an independent Python checker written from the statement agreed with the official checker on all 210 outputs of three programs (the seed, Frontier-CS's examples/reference.cpp and examples/gpt5.cpp): equal verdicts and ratios within 5e-10. Through this harness the seed scores 43.30 and reference.cpp 89.12 (it uses nearly the full 2 s per case); examples/gpt5.cpp places cells out of bounds on every case and scores 0 under the official checker.
- Scores are not comparable with the paper's Polyominoes results: through this harness Frontier-CS's examples/reference.cpp scores 89.12, above the 87.0 the paper reports as the previous best.
- Frontier-CS reduced the problem-0 testdata to one case on 2026-07-08 (#160) and restored it on 2026-07-26 (#171); whether the restored files equal those used in the paper's runs is not verified. The pinned CORAL task.yaml configures claude-sonnet-4-6, while the paper's runs use Claude Opus 4.6.
- Runs the unchanged Frontier-CS checker on the official test inputs with a local judge harness; an independent checker agrees with it on every case of three programs.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `bin:g++`
- `file:data/or_csp/frontier_cs_0/chk.cc`
- `file:data/or_csp/frontier_cs_0/testlib.h`
- `file:data/or_csp/frontier_cs_0/1.in`
- `file:data/or_csp/frontier_cs_0/70.in`

## Sources

- paper: https://arxiv.org/pdf/2604.01658 (Section 4.1 stress tests)
- repository: https://github.com/Human-Agent-Society/CORAL/tree/0123dfb939b35228cf2c1fde224cd0561e727408 (examples/frontier_cs_algo/0)
- benchmark: https://github.com/FrontierCS/Frontier-CS/tree/e6ef46eb4848d99c3c88505d0084284b291bba76 (algorithmic problem 0 (polyomino packing): statement, 70 official test inputs and the official checker)

## Limitations

- Execution interface inspected in pinned source. Full paper experiments have not been reproduced here.
