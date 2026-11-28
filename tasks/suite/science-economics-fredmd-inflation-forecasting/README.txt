# Economics: Forecasting US CPI inflation with FRED-MD

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-economics-fredmd-inflation-forecasting`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Forecast annualized US CPI inflation 1, 6 and 12 months ahead from the 126-series FRED-MD panel available at each month-end origin from 1990 to 2023. The evaluator recomputes the targets and scores the root-mean-square error relative to a direct autoregressive benchmark with BIC lag selection.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

forecast_target(history, horizons) -> array of 3 floats: forecasts of the task's target 1, 6 and 12 months after the origin (horizons = [1, 6, 12]). Targets: for INDPRO and CPIAUCSL the annualized average growth 1200/h * ln(x_{t+h}/x_t) (percent per year); for UNRATE the change u_{t+h} - u_t (percentage points). history = {'yyyymm': (T,) ints, 'series': list of the 126 FRED-MD mnemonics, 'tcodes': (126,) McCracken-Ng transformation codes, 'panel': (T, 126) raw levels from 1959-01 through the origin month (NaN = missing), 'target': mnemonic, 'target_kind': 'growth' or 'difference'}. Called at the 404 month-end origins 1990-01..2023-08 of the 2024-09 vintage (pseudo-real-time); |forecast| <= 1000. Score: mean over horizons of RMSE(candidate) / RMSE(direct autoregression with BIC lag order 0-6 estimated from 1960), excluding forecasts whose target window includes 2020-03..2020-12 (minimize; the autoregressive seed scores exactly 1). The candidate runs in a separate sandboxed process: numpy, scipy, scikit-learn and the other installed packages are importable, while the evaluator's library, its data files, the network and subprocesses are not accessible; one thread and 3 GB of memory. It receives only the arguments described here, one call at a time: the arguments of a call are sent after the value of the previous call has been received. Arguments and return values travel as numbers, strings, lists, dicts with string keys and numeric numpy arrays, so a tuple arrives as a list

## State of the art

No published value exists for this local protocol. Local references (mean RMSE ratio to the autoregressive benchmark): four-factor diffusion index 1.044, its equal-weight combination with the benchmark 0.985.

Reference value: no documented value for this finite instance (reviewed 2026-10-07).

- [Macroeconomic Forecasting Using Diffusion Indexes](https://doi.org/10.1198/073500102317351921)

Parameters: `{"target": "CPIAUCSL"}`

## Labels

Scientific contribution: `discovery` (tier 1); artifact `model`, domain `finance`, data `real_measurements`, verification `sampled`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Forecast key macroeconomic aggregates out of sample using information from many predictors.

Why it matters: Forecasts of output, inflation and unemployment guide monetary and fiscal policy; whether large panels of predictors improve on simple autoregressions out of sample is a central question in empirical macroeconomics.

- [Macroeconomic Forecasting Using Diffusion Indexes](https://doi.org/10.1198/073500102317351921): Main research objective: Forecast US macroeconomic time series using principal-component factors estimated from a large number of predictors. Task connection: The task forecasts CPI inflation from a large monthly panel; price inflation is among the paper's targets.
- [Forecasting Using Principal Components From a Large Number of Predictors](https://doi.org/10.1198/016214502388618960): Main research objective: Develop and justify forecasting with principal components of many predictors. Task connection: Factor-based forecasts of this kind are natural candidates and local references.

Both papers have macroeconomic forecasting with many predictors as their main research goal.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Pseudo-out-of-sample forecasting with many predictors as in Stock and Watson (2002) on the FRED-MD database of McCracken and Ng (2016): direct h-step forecasts, recursive estimation and RMSE relative to an autoregressive benchmark. Deviations: one data vintage (2024-09) instead of real-time vintages, origins 1990-01..2023-08, target windows touching March-December 2020 excluded, and CPI treated as a growth series: the target is inflation and the benchmark regresses it on lags of monthly inflation, although the FRED-MD transformation code of CPIAUCSL is 6 (second difference of logs). Local references (mean RMSE ratio): a four-factor diffusion index with two target lags 1.060 (INDPRO), 1.044 (CPIAUCSL), 1.008 (UNRATE); its equal-weight combination with the benchmark 1.003, 0.985 and 0.969. No published number is reproduced.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/finance/fredmd_2024_10.csv`

## Sources

- paper: https://doi.org/10.1198/073500102317351921 (Stock and Watson (2002): forecasting US macroeconomic series with diffusion indexes (principal components of many predictors), direct multi-step forecasts and comparison with autoregressions.)
- paper: https://doi.org/10.1080/07350015.2015.1086655 (McCracken and Ng (2016): the FRED-MD monthly database, its 128 (here 126) series and transformation codes.)
- archive: https://www.stlouisfed.org/research/economists/mccracken/fred-databases (FRED-MD databases page of the St. Louis Fed; the 2024-09 vintage (current.csv as archived on 2024-10-29, byte-identical to the official 2024-09.csv; data to 2024-08) is pinned in catalog/finance-data-lock.json.)

## Limitations

- A single data vintage (2024-09, data to 2024-08) is used as if it were available in real time: data revisions and publication lags (ragged edges) of the 126 series are ignored, so forecasts are pseudo-out-of-sample.
- Forecasts whose target window includes March-December 2020 are excluded from the RMSE; the pandemic months would otherwise dominate the comparison. The 2021-2023 inflation episode is included.
- The evaluation period is public; a candidate that reads the prepared file or is tuned on 1990-2024 can look ahead. RMSE ratios near 1 have wide sampling uncertainty (overlapping multi-step forecasts); per-horizon Diebold-Mariano p-values are reported.
- The candidate runs in a sandboxed child process that never receives later data, realised outcomes or reference solutions, and no statement of the candidate is executed in the evaluator's process; outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
