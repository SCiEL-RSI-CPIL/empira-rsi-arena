# Finance: Out-of-sample portfolio choice against 1/N (49 industries)

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-finance-portfolio-choice-industry49`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Choose each month's portfolio of 49 US industry portfolios using only the previous 120 months of returns, a setting where estimation error is severe. The evaluator recomputes the Sharpe ratio of the next-month returns net of 50 bp trading costs over 1979-07 to 2025-09 and compares it with the 1/N rule.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

choose_weights(excess_returns, context) -> array of N finite positions x. excess_returns is a (120, N) float array of the previous 120 monthly excess returns (decimal, oldest first) of the task's N assets; context = {'cost_rate': 0.005, 'window': 120, 'max_gross': 20.0, 'risk_free': (120,) monthly T-bill returns, 'previous_weights': drifted relative weights just before rebalancing (None in the first month)}. The evaluator forms DeMiguel-Garlappi-Uppal relative weights w = x/|sum(x)| (sum(x) must be nonzero and sum|w| <= 20), holds w for the next month, charges 0.5% of sum|w - previous_weights| per rebalance and requires wealth to stay positive. Score: monthly Sharpe ratio of the net out-of-sample excess returns over every month of the task period (maximize); the 1/N rule is recomputed on the same months for comparison

## State of the art

No published value exists for this local dataset and period. The local 1/N seed scores 0.1571 net of costs over 1979-07 to 2025-09; the unconstrained sample minimum-variance rule scores 0.0332 because of estimation error.

Reference value: no documented value for this finite instance (reviewed 2026-10-06).

- [Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?](https://doi.org/10.1093/rfs/hhm075)

Parameters: `{"dataset": "industry49", "start": "196907", "end": "202509", "window": 120, "cost": 0.005, "max_gross": 20.0}`

## Labels

Scientific contribution: `method` (tier 2); artifact `policy`, domain `finance`, data `real_measurements`, verification `sampled`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Choose portfolio weights that perform well out of sample when expected returns and covariances must be estimated from short return histories.

Why it matters: Estimation error can make optimized portfolios worse than naive diversification; DGU find that none of 14 optimizing models reliably beats 1/N, so robust portfolio choice remains a central problem for investors and for asset-pricing research.

- [Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?](https://doi.org/10.1093/rfs/hhm075): Main research objective: Determine whether portfolio rules derived from mean-variance optimization, including rules designed to reduce estimation error, outperform naive 1/N diversification out of sample. Task connection: The local task applies DGU's rolling-window protocol to 49 industry portfolios, a larger cross-section that DGU did not use, and scores net-of-cost Sharpe ratios.

The cited paper's main objective is the out-of-sample performance of portfolio rules relative to 1/N, which is exactly what the task scores.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Same rolling-window backtest as the DGU datasets, applied to a dataset DGU did not use; the machinery is checked against DGU's published 1/N values on their datasets only.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/finance/ind49_vw_monthly.csv`
- `file:data/finance/ff_factors_monthly.csv`

## Sources

- paper: https://doi.org/10.1093/rfs/hhm075 (Sections 2-3: rolling-window protocol, relative weights, Sharpe ratio and 50 bp costs; the 49-industry dataset is not one of DGU's datasets.)
- archive: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (49 industry portfolios (value-weighted, complete from 1969-07) and the Fama/French T-bill series, monthly; CRSP 202509 files pinned in catalog/finance-data-lock.json.)

## Limitations

- DGU did not study this 49-industry dataset; the protocol is theirs but no published number is reproduced for it, and with N = 49 and a 120-month window the sample covariance matrix is poorly conditioned.
- The evaluation months are public and fixed. Strategies or hyperparameters tuned on them overfit, and a candidate that reads the prepared files or embeds remembered returns could look ahead; the evaluator only prevents look-ahead through its interface. Treat scores as development scores and freeze a candidate before testing it on months after 2025-09.
- A monthly Sharpe ratio estimated from 50 to 52 years of returns has a standard error of about 0.04, so differences from 1/N below that size are not statistically meaningful; the one-sided Jobson-Korkie-Memmel p-value against 1/N is reported in the details.
- Returns are value-weighted CRSP portfolio returns from the CRSP 202509 vintage of the French library; later vintages revise history slightly. Costs are a flat 50 bp of traded value with no market impact, borrowing spread, short-sale fees or taxes, and positions are financed at the one-month T-bill rate.
- Deviations from DeMiguel, Garlappi and Uppal: the scored period extends to 2025-09, returns are net of costs (their Sharpe table is gross), the first allocation is free, gross exposure is capped at 20 and a strategy whose wealth reaches zero is invalid.
