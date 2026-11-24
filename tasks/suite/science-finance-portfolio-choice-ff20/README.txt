# Finance: Out-of-sample portfolio choice against 1/N (20 size and value portfolios and the market)

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-finance-portfolio-choice-ff20`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Choose each month's portfolio of 20 size and book-to-market portfolios and the market using only the previous 120 months of returns. The evaluator recomputes the Sharpe ratio of the next-month returns net of 50 bp trading costs over 1973-07 to 2025-09 and compares it with the 1/N rule.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

choose_weights(excess_returns, context) -> array of N finite positions x. excess_returns is a (120, N) float array of the previous 120 monthly excess returns (decimal, oldest first) of the task's N assets; context = {'cost_rate': 0.005, 'window': 120, 'max_gross': 20.0, 'risk_free': (120,) monthly T-bill returns, 'previous_weights': drifted relative weights just before rebalancing (None in the first month)}. The evaluator forms DeMiguel-Garlappi-Uppal relative weights w = x/|sum(x)| (sum(x) must be nonzero and sum|w| <= 20), holds w for the next month, charges 0.5% of sum|w - previous_weights| per rebalance and requires wealth to stay positive. Score: monthly Sharpe ratio of the net out-of-sample excess returns over every month of the task period (maximize); the 1/N rule is recomputed on the same months for comparison. The candidate runs in a separate sandboxed process: numpy, scipy, scikit-learn, pandas and the other installed packages are importable, while the evaluator's library, its data files, the network and subprocesses are not accessible; one thread and 3 GB of memory. It receives only the arguments described here, one call at a time: the arguments of a call are sent after the value of the previous call has been received. Arguments and return values travel as numbers, strings, lists, dicts with string keys and numeric numpy arrays, so a tuple arrives as a list

## State of the art

No published value exists for this exact protocol. On DGU's 1963-07 to 2004-11 sample, 1/N has a monthly Sharpe ratio of 0.1623 for this dataset and the constrained minimum-variance rule 0.1546; the local 1/N seed scores 0.1501 net of costs over 1973-07 to 2025-09.

Reference value: no documented value for this finite instance (reviewed 2026-10-06).

- [Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?](https://doi.org/10.1093/rfs/hhm075)

Parameters: `{"dataset": "ff20_market", "start": "196307", "end": "202509", "window": 120, "cost": 0.005, "max_gross": 20.0}`

## Labels

Scientific contribution: `method` (tier 2); artifact `policy`, domain `finance`, data `real_measurements`, verification `sampled`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Choose portfolio weights that perform well out of sample when expected returns and covariances must be estimated from short return histories.

Why it matters: Estimation error can make optimized portfolios worse than naive diversification; DGU find that none of 14 optimizing models reliably beats 1/N, so robust portfolio choice remains a central problem for investors and for asset-pricing research.

- [Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?](https://doi.org/10.1093/rfs/hhm075): Main research objective: Determine whether portfolio rules derived from mean-variance optimization, including rules designed to reduce estimation error, outperform naive 1/N diversification out of sample. Task connection: The local task applies DGU's rolling-window protocol to their dataset 5 (20 size/book-to-market portfolios and the market) on a longer sample and scores net-of-cost Sharpe ratios.

The cited paper's main objective is the out-of-sample performance of portfolio rules relative to 1/N, which is exactly what the task scores.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Rolling-window backtest of DeMiguel, Garlappi and Uppal (2009): 120-month windows, relative weights, Sharpe ratio of out-of-sample excess returns and 50 bp proportional costs on turnover from drifted weights. Reference check: on DGU's sample (1963-07 to 2004-11) the evaluator's 1/N strategy reproduces DGU's published gross monthly Sharpe ratios 0.1353 (industry, exact to four decimals) and 0.1613 versus 0.1623 (FF-1-factor) and 1/N turnovers 0.0220 versus 0.0216 and 0.0164 versus 0.0162 with the current CRSP 202509 vintage of the French data (validation/finance-checks.json). Deviations: the scored period is 1973-07 to 2025-09, returns are net of costs, the first allocation is free, gross exposure is capped at 20 and a strategy that loses all wealth is invalid (DGU's unconstrained sample mean-variance policy violates the last two).

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/finance/ff25_vw_monthly.csv`
- `file:data/finance/ff_factors_monthly.csv`

## Sources

- paper: https://doi.org/10.1093/rfs/hhm075 (Sections 2-3 and Tables 2-5: dataset 5 (FF-1-factor: 20 size/book-to-market portfolios and the market, N = 21), 120-month rolling windows, relative weights, Sharpe ratio and 50 bp costs.)
- archive: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (25 portfolios formed on size and book-to-market (value-weighted; the five largest-size portfolios are dropped as in DGU) and Fama/French market and T-bill series, monthly; CRSP 202509 files pinned in catalog/finance-data-lock.json.)

## Limitations

- The evaluation months are public and fixed. Strategies or hyperparameters tuned on them overfit, and a candidate that reads the prepared files or embeds remembered returns could look ahead; the evaluator only prevents look-ahead through its interface. Treat scores as development scores and freeze a candidate before testing it on months after 2025-09.
- A monthly Sharpe ratio estimated from 50 to 52 years of returns has a standard error of about 0.04, so differences from 1/N below that size are not statistically meaningful; the one-sided Jobson-Korkie-Memmel p-value against 1/N is reported in the details.
- Returns are value-weighted CRSP portfolio returns from the CRSP 202509 vintage of the French library; later vintages revise history slightly. Costs are a flat 50 bp of traded value with no market impact, borrowing spread, short-sale fees or taxes, and positions are financed at the one-month T-bill rate.
- Deviations from DeMiguel, Garlappi and Uppal: the scored period extends to 2025-09, returns are net of costs (their Sharpe table is gross), the first allocation is free, gross exposure is capped at 20 and a strategy whose wealth reaches zero is invalid.
- The candidate runs in a sandboxed child process that never receives later data, realised outcomes or reference solutions; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
