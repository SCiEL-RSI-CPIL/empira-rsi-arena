# Finance: Index tracking with few stocks and transaction costs (OR-Library)

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-finance-index-tracking`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Track five stock indices (Hang Seng, DAX 100, FTSE 100, S&P 100 and Nikkei 225) with buy-and-hold positions in at most ten constituents, formed from the first 145 weeks of prices and optionally rebalanced every 13 weeks at a 1% cost. The evaluator recomputes the portfolio values and scores the out-of-sample tracking error over the last 145 weeks.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

rebalance(state) -> array of N target weights, or None to keep the current shares. Called at weeks 145, 158, ..., 288 (every 13 weeks) for each OR-Library market: Hang Seng (N = 31), DAX 100 (85), FTSE 100 (89), S&P 100 (98), Nikkei 225 (225). state = {'week': t, 'index_prices': (t+1,) weekly index levels for weeks 0..t, 'stock_prices': (t+1, N), 'holdings': (N,) shares held, 'portfolio_value': value before trading, 'K': 10, 'min_weight': 0.01, 'max_weight': 1.0, 'cost_rate': 0.01, 'rebalance_every': 13}. Weights: at most 10 stocks held (|w| <= 1e-12 counts as not held), each held weight in [0.01, 1], summing to 1; a portfolio is required at week 145. The first purchase is free, later trades pay 1% of the traded value, and shares are held fixed between rebalances. Score: out-of-sample tracking error, the root-mean-square difference between weekly log returns of the portfolio (net of costs) and of the index over weeks 146..290, in percent, averaged over the five markets (minimize)

## State of the art

No published value exists for this protocol. Local references (mean weekly tracking error in percent): the correlation/NNLS buy-and-hold seed 1.087 and greedy forward selection on the last 104 weeks refreshed every 26 weeks 0.764.

Reference value: no documented value for this finite instance (reviewed 2026-10-08).

- [An evolutionary heuristic for the index tracking problem](https://doi.org/10.1016/s0377-2217(02)00425-3)

## Labels

Scientific contribution: `method` (tier 2); artifact `policy`, domain `finance`, data `public_benchmark`, verification `sampled`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Replicate the returns of a stock index with a small number of its constituents while limiting transaction costs.

Why it matters: Passive index funds manage a large share of equity assets; holding few stocks lowers costs but creates tracking error, and selecting them is a hard cardinality-constrained optimization problem.

- [An evolutionary heuristic for the index tracking problem](https://doi.org/10.1016/s0377-2217(02)00425-3): Main research objective: Construct tracking portfolios with few stocks and transaction costs that follow an index out of sample. Task connection: The task uses the paper's OR-Library markets, buy-and-hold tracking portfolios with at most ten stocks and its tracking-error measure out of sample.
- [Mixed-integer programming approaches for index tracking and enhanced indexation](https://doi.org/10.1016/j.ejor.2008.03.015): Main research objective: Solve index tracking and enhanced indexation with mixed-integer programming. Task connection: Candidate programs address the same tracking problem on the same data.

Both papers have index tracking as their main research goal.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Index tracking of Beasley, Meade and Chang (2003) on their five OR-Library markets: buy-and-hold positions in at most K = 10 stocks (holdings 1% to 100%), the first 145 weeks as the estimation period and the last 145 weeks for out-of-sample evaluation, root-mean-square tracking error of weekly log returns (their alpha = 2), proportional transaction costs of 1%. Deviations: optional rebalancing every 13 weeks with costs charged to the portfolio value (instead of their single rebalance with a cost budget and their excess-return trade-off), a free initial purchase and an at-most-K constraint. No published number is reproduced. Local references: the correlation/NNLS buy-and-hold seed 1.087; greedy forward selection on 104 weeks refreshed every 26 weeks 0.764.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:scipy`
- `file:data/finance/orlib_indtrack1.csv`
- `file:data/finance/orlib_indtrack2.csv`
- `file:data/finance/orlib_indtrack3.csv`
- `file:data/finance/orlib_indtrack4.csv`
- `file:data/finance/orlib_indtrack5.csv`

## Sources

- paper: https://doi.org/10.1016/s0377-2217(02)00425-3 (Beasley, Meade and Chang (2003): index tracking with buy-and-hold positions in at most K stocks, holding bounds, transaction costs, tracking error (sum of |portfolio - index log return|^alpha / T)^(1/alpha), and the five OR-Library markets.)
- paper: https://doi.org/10.1016/j.ejor.2008.03.015 (Canakgoz and Beasley (2009): mixed-integer programming approaches for index tracking on the same OR-Library data.)
- benchmark: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/indtrackinfo.html (OR-Library indtrack1-indtrack5 (weekly prices of the index and its constituents, weeks 0-290), pinned by SHA-256 in catalog/finance-data-lock.json.)

## Limitations

- The protocol (optional rebalancing every 13 weeks with costs charged to the portfolio, free initial purchase, at most K stocks, pure tracking without an excess-return objective) differs from Beasley et al.'s single-rebalance model with a cost budget, so their published tracking errors are not directly comparable.
- Weekly prices of 1992-1997 for constituents at the end of the sample; survivorship and constituent changes are not modelled, and dividends are ignored as in the OR-Library data.
- The out-of-sample period (145 weeks) is public and short; candidates tuned on it overfit, and tracking errors from one period have wide sampling uncertainty.
