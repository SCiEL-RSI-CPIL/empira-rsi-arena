# Finance: Forecasting the US Treasury yield curve

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-finance-yield-curve-forecasting`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Forecast end-of-month US Treasury zero-coupon yields at maturities of 1 to 10 years, 1, 6 and 12 months ahead, from the curves observed up to each origin between 1990 and 2024. The evaluator recomputes the forecast errors and scores the root-mean-square error relative to the random walk.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

forecast_yields(history, horizons) -> (3, 10) float array: forecasts (percent) of the zero-coupon yields at maturities 1..10 years, 1, 6 and 12 months after the origin (rows follow horizons = (1, 6, 12)). history = {'yyyymm': (T,) ints, 'maturities': array 1.0..10.0 (years), 'yields': (T, 10) end-of-month Gurkaynak-Sack-Wright continuously compounded zero-coupon yields (percent) from 1972-01 through the origin month}. Called at the 415 month-end origins 1990-01..2024-07; |forecast| <= 30. Score: RMSE of the candidate divided by the RMSE of the random walk (no-change forecast), averaged over the three horizons and the maturities 1, 2, 3, 5, 7 and 10 years (minimize; the random-walk seed scores exactly 1)

## State of the art

No published value exists for this local protocol. Diebold and Li (2006) found their dynamic Nelson-Siegel forecasts beat the random walk at 6- and 12-month horizons on 1994-2000 Fama-Bliss data; on this 1990-2024 sample a local implementation with AR(1) factors scores 1.171 (worse than the random walk), while a random walk plus a small momentum term scores 0.993.

Reference value: no documented value for this finite instance (reviewed 2026-10-07).

- [Forecasting the term structure of government bond yields](https://doi.org/10.1016/j.jeconom.2005.03.005)

## Labels

Scientific contribution: `discovery` (tier 1); artifact `model`, domain `finance`, data `real_measurements`, verification `sampled`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Forecast the term structure of government bond yields out of sample.

Why it matters: Yield-curve forecasts matter for monetary policy analysis, bond portfolio management and debt issuance; beating the random walk out of sample is a long-standing open empirical question.

- [Forecasting the term structure of government bond yields](https://doi.org/10.1016/j.jeconom.2005.03.005): Main research objective: Forecast the government bond yield curve with a dynamic Nelson-Siegel factor model and evaluate it against standard benchmarks out of sample. Task connection: The task scores recursive 1-, 6- and 12-month-ahead yield forecasts relative to the random walk, the paper's main comparison.

The cited paper's main objective is out-of-sample forecasting of the yield curve.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Recursive out-of-sample yield-curve forecasting as in Diebold and Li (2006), scored by RMSE relative to the random walk, on Gurkaynak-Sack-Wright fitted zero-coupon yields instead of Fama-Bliss unsmoothed yields, with 1- to 10-year maturities and origins 1990-01..2024-07 instead of 3-month to 10-year maturities and 1994-2000. Local references: Diebold-Li dynamic Nelson-Siegel with AR(1) factors (15-year window) 1.171; a random walk plus 0.1 times the change over the horizon length 0.993. No published number is reproduced.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/finance/gsw_yields_monthly.csv`

## Sources

- paper: https://doi.org/10.1016/j.jeconom.2005.03.005 (Diebold and Li (2006): dynamic Nelson-Siegel forecasting of the yield curve, recursive out-of-sample forecasts 1, 6 and 12 months ahead, RMSE compared with the random walk.)
- paper: https://doi.org/10.1016/j.jmoneco.2007.06.029 (Gurkaynak, Sack and Wright (2007): the daily fitted Svensson zero-coupon curve used as data.)
- archive: https://www.federalreserve.gov/data/yield-curve-tables/feds200628.htm (Federal Reserve Board page of the GSW yield-curve tables (feds200628.csv, file pinned via an Internet Archive snapshot of 2025-08-09 in catalog/finance-data-lock.json).)

## Limitations

- GSW yields are smoothed fitted values from a Svensson curve, not observed bond yields; fitting error and smoothing make them easier to forecast than raw yields, and maturities below one year are excluded because GSW caution against them.
- The evaluation origins (1990-2024) and the data file are public; tuning on them or reading the prepared file can produce look-ahead. Forecasts at different origins overlap for the 6- and 12-month horizons, so the effective sample is smaller than 415.
- One data vintage (archived 2025-08-09) is used as if it were real-time; the live file is revised without notice.
- RMSE ratios against the random walk have wide sampling uncertainty; Diebold-Mariano statistics per horizon are reported but a ratio slightly below 1 is weak evidence.
