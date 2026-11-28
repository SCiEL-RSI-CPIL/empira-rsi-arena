# XLA auto-sharding IOPDDL

**Paper:** Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve (https://arxiv.org/pdf/2601.21096)\
**Task id:** `magellan-xla-auto-sharding-iopddl`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

XLA auto-sharding IOPDDL task evaluated in Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve.

Benchmark named in the paper: XLA auto-sharding IOPDDL. Paper metric: contest objective under memory constraints.

## Candidate interface

solve(problem, time_limit_s) -> list with one strategy index per node, or [] for no solution. problem is the contest's JSON 'problem' object: name; nodes = {intervals: [start, end) per node, costs: per-strategy node costs, usages: per-strategy memory usages}; edges = {nodes: [u, v] pairs, costs: row-major |S_u| x |S_v| resharding cost tables}; usage_limit. The usages of the nodes whose interval contains t must sum to at most usage_limit for every t; forbidden choices cost 1e18. Score (maximize): mean contest points over the benchmarks, points = best submitted contest cost / candidate total cost, 0 for an infeasible, empty or late answer (later than time_limit_s + grace_s wall-clock seconds); a wrong-length or out-of-range answer is invalid

Parameters: `{"benchmarks": ["A", "G", "M", "S", "Y"], "time_limit_s": 10.0, "grace_s": 1.0}`

## Labels

Scientific contribution: `engineering` (tier 3); artifact `program`, domain `systems`, data `public_benchmark`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Problem and objective are those of the ASPLOS 2025 / EuroSys 2025 IOPDDL contest (github.com/asplos-contest/2025, IOPDDL.md), which the Magellan paper follows, and points are normalised by the best submitted contest cost (local differences are in the Score note); Magellan's evolved heuristic and prompts are not released; its search started from a Gemini-2.5-Pro-generated solution and ran for one week of evolution. None of this is reproduced (the local seed is a greedy heuristic).
- Benchmarks: the five instances released to participants during the contest (A, G, M, S, Y from github.com/google/iopddl at commit 8a61c4359c5a330a04c8ef1780402cc254d28f29, Apache-2.0, fetched by scripts/prepare_or_csp.py and locked in catalog/or_csp-data-lock.json). Magellan searched on these public instances and reports results on the 20 benchmarks that were withheld during the contest; all 25 are public in google/iopddl at the pinned commit, but the 20 evaluation benchmarks are not used here.
- Time limit: solve() gets 10 wall-clock seconds per benchmark in the evaluator process and answers later than 11 s earn 0 points; the contest ran C++ binaries for 1-5 minutes per benchmark on an 8-core workstation. Scores of time-bounded candidates depend on the machine. solve() and the module's top-level code run in the evaluator's process and can reach the scoring code and the reference costs there, so scores of untrusted candidates are not tamper-proof.
- Score: mean over the five benchmarks of best submitted contest cost / candidate cost, where the best submitted cost is the minimum over the teams in the contest's final results (asplos-contest/2025 misc/iopddl_final_results.csv at commit 06267509624cea22b98cd50427aaae6d01e3b6c8); the contest sums these points over its 20 evaluation benchmarks. An answer that exceeds the usage limit, an empty answer or a late answer earns 0 points here (a local rule; IOPDDL.md does not state how such answers are scored). Points are not capped at 1: the reference is the fixed best submitted cost, whereas a contest minimum includes the scored team, so contest points never exceed 1; a wrong-length, non-integer or out-of-range answer makes the candidate invalid.
- Verification: the evaluator ports iopddl::Evaluate (half-open intervals, row-major edge strategy index, exact integer totals). On 35 fixed answers over the five benchmarks (minimum-usage, minimum-node-cost, maximum-usage, random and seed answers, a truncated answer and an out-of-range index) it reproduced the official C++ evaluator's cost or error class exactly; the official unit-test example costs (495 and 445, and the usage-limit error) are reproduced by the unit tests.
- The benchmarks price forbidden strategies and strategy pairs at 1e18; an answer using them is feasible but earns almost no points. The seed (greedy plus best-response sweeps) reaches 0.60 points on M and almost none on the other benchmarks, so the task leaves a large gap to the best contest solutions.
- Port of iopddl::Evaluate (github.com/google/iopddl iopddl.cc) checked against the official C++ evaluator; contest points use the best submitted costs of the contest's final results.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/or_csp/iopddl/asplos-2025-iopddl-A.json.gz`
- `file:data/or_csp/iopddl/asplos-2025-iopddl-G.json.gz`
- `file:data/or_csp/iopddl/asplos-2025-iopddl-M.json.gz`
- `file:data/or_csp/iopddl/asplos-2025-iopddl-S.json.gz`
- `file:data/or_csp/iopddl/asplos-2025-iopddl-Y.json.gz`

## Sources

- paper: https://arxiv.org/pdf/2601.21096 (Sections 3.1-3.3)
- benchmark: https://github.com/google/iopddl/tree/8a61c4359c5a330a04c8ef1780402cc254d28f29 (IOPDDL contest problem format, public benchmarks A, G, M, S, Y and the reference cost evaluator)
- benchmark: https://github.com/asplos-contest/2025/tree/06267509624cea22b98cd50427aaae6d01e3b6c8 (ASPLOS 2025 contest final results used for the best-team reference costs)

## Limitations

- No official public runnable implementation verified from the inspected paper. External benchmark code may exist separately.
- No publicly accessible, paper-linked implementation with a verified immutable revision was established.
