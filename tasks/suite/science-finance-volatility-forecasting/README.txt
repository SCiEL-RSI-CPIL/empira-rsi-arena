# Finance: Daily volatility forecasting for US equity portfolios

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-finance-volatility-forecasting`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Forecast the next day's return variance of the US stock market and 49 industry portfolios, updating the forecasts day by day from 2000 to 2025 after a training period ending in 1999. The evaluator reveals each day's returns only after the forecasts are made and scores the QLIKE loss against squared returns.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

make_forecaster(training_returns, names) -> object with forecast() and update(returns). training_returns is a (7280, 50) float array of daily returns (decimal) from 1971-03-12 to 1999-12-31: column 0 is the CRSP value-weighted market (Mkt-RF + RF), columns 1-49 the value-weighted Fama-French industries; names lists the 50 series. The evaluator then walks through the 6,475 trading days 2000-01-03..2025-09-30: forecast() must return a (50,) array of next-day return variances in [1e-10, 1]; afterwards update(r) receives that day's (50,) returns. Score: mean QLIKE loss log(h) + r^2/h over all days and series (minimize); the RiskMetrics EWMA benchmark (lambda 0.94), the difference and a Diebold-Mariano statistic are reported

## State of the art

No published value exists for this local protocol. Local references (mean QLIKE): rolling 63-day variance -7.563 (the seed), RiskMetrics EWMA -7.6125, GARCH(1,1) -7.598 and GJR-GARCH(1,1) -7.610 with parameters estimated on 1990-1999 and kept fixed; a candidate has to beat RiskMetrics to show a useful improvement.

Reference value: no documented value for this finite instance (reviewed 2026-10-07).

- [A forecast comparison of volatility models: does anything beat a GARCH(1,1)?](https://doi.org/10.1002/jae.800)

## Labels

Scientific contribution: `discovery` (tier 1); artifact `model`, domain `finance`, data `real_measurements`, verification `sampled`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Forecast the time-varying volatility of asset returns out of sample.

Why it matters: Volatility forecasts drive risk management (value at risk), option pricing and portfolio allocation; whether elaborate models beat simple benchmarks out of sample is a central empirical question in financial econometrics.

- [A forecast comparison of volatility models: does anything beat a GARCH(1,1)?](https://doi.org/10.1002/jae.800): Main research objective: Compare the out-of-sample accuracy of a large set of volatility models. Task connection: The task compares candidate volatility forecasts out of sample on US equity portfolios with a robust loss and simple benchmarks.
- [Generalized autoregressive conditional heteroskedasticity](https://doi.org/10.1016/0304-4076(86)90063-1): Main research objective: Model and forecast conditional variances of economic time series. Task connection: GARCH-type forecasters are the natural starting point for candidates and serve as local references.

Both papers have volatility modelling and forecasting as their main research goal.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Out-of-sample comparison of daily variance forecasts in the spirit of Hansen and Lunde (2005), with squared daily returns as the volatility proxy and the QLIKE loss, which Patton (2011) shows ranks forecasts consistently with such noisy proxies. The evaluator enforces causality by revealing each day's returns only after that day's forecasts. Local references: rolling 63-day variance (seed) -7.563, RiskMetrics -7.6125, GARCH(1,1) and GJR-GARCH(1,1) estimated on 1990-1999 and kept fixed -7.598 and -7.610. No published number is reproduced: Hansen and Lunde use other assets, intraday realized-variance proxies and other periods.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/finance/ind49_vw_daily.csv`
- `file:data/finance/ff_factors_daily.csv`

## Sources

- paper: https://doi.org/10.1002/jae.800 (Hansen and Lunde (2005): out-of-sample comparison of many volatility models against GARCH(1,1) with several loss functions.)
- paper: https://doi.org/10.1016/j.jeconom.2010.03.034 (Patton (2011): loss functions, including QLIKE, that rank volatility forecasts consistently when the true variance is replaced by a noisy proxy such as the squared return.)
- paper: https://doi.org/10.1016/0304-4076(86)90063-1 (Bollerslev (1986): the GARCH model, the classic benchmark for conditional variance forecasts.)
- archive: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (Daily Fama/French factors (market = Mkt-RF + RF) and value-weighted 49 industry portfolio returns, CRSP 202509 files pinned in catalog/finance-data-lock.json.)

## Limitations

- Squared daily returns are a noisy proxy for the daily variance; QLIKE ranks forecasts consistently with this proxy but the loss level is not a variance-accuracy measure, and intraday realized variances are not available here.
- Forecasts are for close-to-close total returns (no mean adjustment); weekends, holidays and overnight effects are treated like any other trading day.
- The evaluation period 2000-2025 is public. A candidate that reads the prepared files or memorizes returns can look ahead; the interface only passes returns up to the previous day. Treat scores as development scores.
- One data vintage of CRSP value-weighted portfolio returns is used; the 50 series are strongly correlated, so they do not provide 50 independent comparisons.
