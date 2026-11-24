# Finance: Cardinality-constrained mean-variance frontier (OR-Library)

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-finance-cardinality-constrained-frontier`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Build mean-variance portfolios of exactly ten assets, each held between 1% and 100%, at 51 fixed target returns for five OR-Library stock markets with 31 to 225 assets. The evaluator checks every constraint and scores Chang et al.'s mean percentage deviation from the published unconstrained efficient frontier.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

build_frontier(problem) -> (51, N) float array: one portfolio per target return. problem = {'name': market, 'mean': (N,) expected weekly returns, 'cov': (N, N) covariance, 'K': 10, 'min_weight': 0.01, 'max_weight': 1.0, 'target_returns': (51,) evenly spaced from the return of the unconstrained minimum-variance portfolio to the highest return attainable with the constraints}. Every portfolio must hold exactly 10 assets (|w| <= 1e-12 counts as not held), each held weight in [0.01, 1], weights summing to 1 and expected return equal to its target (all within 1e-8). Called once for each OR-Library market: Hang Seng (N = 31), DAX 100 (85), FTSE 100 (89), S&P 100 (98), Nikkei 225 (225). Score: Chang et al.'s percentage error of each portfolio against the published unconstrained efficient frontier, min(100|s - s*(r)|/s*(r), 100|r - r*(s)|/r*(s)) by linear interpolation on the 2000 published frontier points (s = standard deviation, r = return), averaged over the 51 targets and then over the five markets (minimize). The candidate runs in a separate sandboxed process: numpy, scipy, scikit-learn, pandas and the other installed packages are importable, while the evaluator's library, its data files, the network and subprocesses are not accessible; one thread and 3 GB of memory. It receives only the arguments described here, one call at a time: the arguments of a call are sent after the value of the previous call has been received. Arguments and return values travel as numbers, strings, lists, dicts with string keys and numeric numpy arrays, so a tuple arrives as a list

## State of the art

No published value exists for the target-return protocol. Under the lambda sweep, Chang et al. report mean percentage errors of 1.0974/1.0957/1.1217 (GA/SA/TS) for Hang Seng, 2.5424/2.9297/3.3049 for DAX, 1.1076/1.4623/1.6080 for FTSE, 1.9328/3.0696/3.3092 for S&P and 0.7961/0.6732/0.8975 for Nikkei. Local references here: the frontier-support seed 1.535 and a single-swap local search 1.381 (mean over markets).

Reference value: no documented value for this finite instance (reviewed 2026-10-08).

- [Heuristics for cardinality constrained portfolio optimisation](https://doi.org/10.1016/s0305-0548(99)00074-x)
- [Markowitz-based cardinality constrained portfolio selection using Asexual Reproduction Optimization (ARO)](https://arxiv.org/abs/2101.03312)

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `finance`, data `public_benchmark`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Trace the mean-variance efficient frontier when portfolios must hold a fixed small number of assets with minimum and maximum holdings.

Why it matters: Practical portfolios limit the number of holdings and position sizes; these constraints make mean-variance optimization NP-hard and change the shape of the efficient frontier, so efficient algorithms are needed by portfolio managers.

- [Heuristics for cardinality constrained portfolio optimisation](https://doi.org/10.1016/s0305-0548(99)00074-x): Main research objective: Find the cardinality-constrained efficient frontier with heuristic algorithms and study how the constraints change it. Task connection: The task uses the paper's model, parameters, OR-Library markets and percentage-error measure, with portfolios required at fixed target returns.
- [Heuristic algorithms for the cardinality constrained efficient frontier](https://doi.org/10.1016/j.ejor.2011.03.030): Main research objective: Develop better heuristics for the cardinality-constrained efficient frontier. Task connection: Candidate programs are heuristics for the same frontier problem.

Both papers have the cardinality-constrained efficient frontier as their main research goal.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Cardinality-constrained mean-variance model of Chang, Meade, Beasley and Sharaiha (2000) on their five OR-Library markets with their parameters (K = 10, holdings 0.01 to 1) and their percentage-error measure against the published unconstrained efficient frontier. Deviation: portfolios are required at 51 fixed target returns (epsilon-constraint form) instead of the 51 risk-aversion weights of their lambda sweep, so that coverage of the frontier is enforced and the error cannot be lowered by clustering points; published mean percentage errors are therefore not directly comparable. Check: an independent long-only quadratic program on the parsed means and correlations reproduces OR-Library's published frontier points to 2.5e-5 relative variance.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:cvxpy`
- `file:data/finance/orlib_port1.csv`
- `file:data/finance/orlib_port2.csv`
- `file:data/finance/orlib_port3.csv`
- `file:data/finance/orlib_port4.csv`
- `file:data/finance/orlib_port5.csv`
- `file:data/finance/orlib_portef1.csv`
- `file:data/finance/orlib_portef2.csv`
- `file:data/finance/orlib_portef3.csv`
- `file:data/finance/orlib_portef4.csv`
- `file:data/finance/orlib_portef5.csv`

## Sources

- paper: https://doi.org/10.1016/s0305-0548(99)00074-x (Chang, Meade, Beasley and Sharaiha (2000): cardinality-constrained mean-variance model (exactly K assets, holding bounds), the five OR-Library markets, K = 10, epsilon = 0.01, delta = 1, and the percentage-error measure against the unconstrained frontier.)
- paper: https://doi.org/10.1016/j.ejor.2011.03.030 (Woodside-Oriakhi, Lucas and Beasley (2011): heuristic algorithms for the same cardinality-constrained frontier and its evaluation against the unconstrained frontier.)
- paper: https://arxiv.org/abs/2101.03312 (Restates Chang et al.'s percentage error (Eqs. 20-23: standard-deviation and return errors by linear interpolation on the published frontier, minimum of the two, averaged) and their GA, SA and TS results (Table 5).)
- benchmark: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html (OR-Library portfolio files port1-port5 (means, standard deviations, correlations) and portef1-portef5 (2000 frontier points each), pinned by SHA-256 in catalog/finance-data-lock.json.)

## Limitations

- Portfolios are required at 51 fixed target returns rather than at the 51 risk-aversion weights of the published lambda sweep, so published mean percentage errors (for example 1.0957 for Hang Seng by simulated annealing) are not directly comparable with this score.
- The unconstrained frontier is a relaxation: no cardinality-constrained portfolio reaches zero error, and the attainable minimum per market is unknown (the problem is NP-hard).
- Means and covariances are in-sample estimates from weekly prices of 1992-1997; the score measures optimization quality on these fixed inputs, not out-of-sample investment performance. Transaction costs, round lots and turnover are ignored.
- The candidate runs in a sandboxed child process that never receives later data, realised outcomes or reference solutions; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
