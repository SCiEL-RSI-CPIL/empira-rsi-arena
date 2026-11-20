# Economics: Maximum Nash welfare allocation of indivisible goods

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-economics-maximum-nash-welfare`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Allocate indivisible goods among agents with additive Spliddit-style valuations to maximize Nash welfare, the geometric mean of the agents' utilities. The evaluator recomputes each allocation's Nash welfare and compares it with the proven optimum on eight fixed instances with 4 to 15 agents.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

allocate(values) -> list of m agent indices, the owner of each good. values is an (n, m) int array; row i holds agent i's additive values for the m indivisible goods and sums to 1000 (Spliddit-style point split). Eight fixed instances with (n, m) = (4, 16), (5, 20), (6, 24), (8, 32), (10, 40), (12, 48), (15, 60), (15, 60). Nash welfare is the geometric mean of the agents' utilities (0 if any agent gets nothing). Score: mean over instances of Nash welfare divided by the proven maximum Nash welfare (maximize; 1 is optimal); envy-freeness up to one good is reported per instance

## State of the art

The maximum Nash welfare of every instance is proven by the exact integer-utility program of Caragiannis et al., so the best possible score is 1; all eight optimal allocations are envy-free up to one good. The round-robin seed scores 0.872.

Reference value: proven optimum `1.0` (reviewed 2026-10-07).

- [The Unreasonable Fairness of Maximum Nash Welfare](https://doi.org/10.1145/3355902)

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `finance`, data `synthetic`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Divide indivisible goods fairly and efficiently among agents with additive valuations by maximizing Nash welfare.

Why it matters: Maximum Nash welfare guarantees envy-freeness up to one good and Pareto optimality and is deployed for real divisions on Spliddit, but it is NP-hard to compute, so exact and approximate algorithms are an active research topic.

- [The Unreasonable Fairness of Maximum Nash Welfare](https://doi.org/10.1145/3355902): Main research objective: Establish the fairness and efficiency guarantees of maximum Nash welfare allocations and compute them in practice. Task connection: The task scores allocations by Nash welfare against the optimum computed with the paper's integer program.
- [Approximating the Nash Social Welfare with Indivisible Items](https://doi.org/10.1137/15m1053682): Main research objective: Design approximation algorithms for maximizing Nash social welfare with indivisible items. Task connection: Candidate programs are algorithms for the same maximization problem.

Both papers have fair division by maximum Nash welfare as their main research question.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Maximum Nash welfare for additive valuations as computed on Spliddit by Caragiannis et al.: the reference optimum uses their integer-utility mixed-integer program (log utility replaced by secants between consecutive integers, exact at integer utilities) solved by HiGHS. Reference checks: the program matches brute force on small instances, and every computed optimum is envy-free up to one good, as their theorem predicts. Instances are synthetic Spliddit-style point splits from a fixed legacy NumPy random stream guarded by SHA-256.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://doi.org/10.1145/3355902 (Caragiannis, Kurokawa, Moulin, Procaccia, Shah and Wang (2019): maximum Nash welfare is envy-free up to one good and Pareto optimal; integer-utility mixed-integer program used on Spliddit (point splits of 1000).)
- paper: https://doi.org/10.1137/15m1053682 (Cole and Gkatzelis (2018): computing maximum Nash welfare with indivisible goods is hard to approximate; first constant-factor approximation.)

## Limitations

- Instances are synthetic Spliddit-style point splits (common good quality times idiosyncratic taste), not data from Spliddit users.
- Every instance's optimum is known (exact program solved in at most about 30 s), so a perfect score reproduces known optima; instance sizes are small enough for exact solution and do not test scalability.
- Only Nash welfare is scored; envy-freeness up to one good is reported but not required, and strategic misreporting of values is ignored.
