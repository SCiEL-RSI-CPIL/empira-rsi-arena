# Finance: Covariance estimation for minimum-variance portfolios

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-finance-covariance-minimum-variance`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Estimate the covariance matrix of 49 US industry portfolios from the previous 126 daily returns so that the implied global minimum-variance portfolio has the lowest realized out-of-sample volatility. The evaluator forms the portfolio from each estimate every 21 trading days from 1971 to 2025 and scores the annualized standard deviation of its daily returns.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

estimate_covariance(returns) -> (49, 49) symmetric positive-definite float array. returns is a (126, 49) float array of the previous 126 daily returns (decimal, oldest first) of the 49 Fama-French industry portfolios. Called every 21 trading days from 1971-09 to 2025-09 (649 times); the evaluator forms the global minimum-variance weights w = S^-1 1 / (1' S^-1 1) from the estimate, holds them for the next 21 days and scores the annualized standard deviation (percent) of the 13,629 daily out-of-sample portfolio returns (minimize). The estimate must be finite, symmetric (1e-10 relative) and positive definite with condition number below 1e12

## State of the art

No published value exists for this local dataset and window. Local references (annualized percent): sample covariance 11.97 (the seed), Ledoit-Wolf linear shrinkage to the identity 10.82 and to constant correlation 10.90, an implementation of Ledoit-Wolf analytical nonlinear shrinkage 10.80, diagonal covariance 15.16 and 1/N 16.79.

Reference value: no documented value for this finite instance (reviewed 2026-10-07).

- [Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection: Markowitz Meets Goldilocks](https://doi.org/10.1093/rfs/hhx052)
- [Analytical nonlinear shrinkage of large-dimensional covariance matrices](https://doi.org/10.1214/19-AOS1921)

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `finance`, data `real_measurements`, verification `sampled`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Estimate large covariance matrices of asset returns from short samples so that optimized portfolios have low realized risk.

Why it matters: The sample covariance matrix is badly conditioned when the number of assets is not small relative to the sample, which makes minimum-variance and mean-variance portfolios unstable; better estimators directly reduce realized portfolio risk.

- [Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection: Markowitz Meets Goldilocks](https://doi.org/10.1093/rfs/hhx052): Main research objective: Develop covariance-matrix estimators that minimize the out-of-sample variance of optimized portfolios. Task connection: The task uses the paper's criterion, the realized standard deviation of the global minimum-variance portfolio with monthly rebalancing, on industry portfolios.
- [Improved estimation of the covariance matrix of stock returns with an application to portfolio selection](https://doi.org/10.1016/S0927-5398(03)00007-0): Main research objective: Improve covariance estimation for portfolio selection by shrinking the sample covariance matrix. Task connection: Shrinkage estimators of this paper are local references for the task's score.

Both papers have covariance estimation for portfolio selection as their main research goal.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Out-of-sample global minimum-variance criterion of Ledoit and Wolf (2017) with monthly (21-day) rebalancing on daily industry-portfolio returns instead of their individual CRSP stocks, a 126-day window instead of 250 days, and N = 49. No published number is reproduced. Local references: sample covariance 11.97, Ledoit-Wolf shrinkage to the identity 10.82 (matches scikit-learn's LedoitWolf to 1e-19) and to constant correlation 10.90, an implementation of their analytical nonlinear shrinkage 10.80, and 1/N 16.79 (annualized percent).

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/finance/ind49_vw_daily.csv`

## Sources

- paper: https://doi.org/10.1093/rfs/hhx052 (Ledoit and Wolf (2017): covariance estimators judged by the out-of-sample standard deviation of the global minimum-variance portfolio with monthly (21-day) rebalancing on daily returns.)
- paper: https://doi.org/10.1016/S0927-5398(03)00007-0 (Ledoit and Wolf (2003): linear shrinkage of the sample covariance matrix for portfolio selection.)
- paper: https://doi.org/10.1214/19-AOS1921 (Ledoit and Wolf (2020): analytical nonlinear shrinkage estimator used as a local reference.)
- archive: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html (49 industry portfolios, value-weighted daily returns (CRSP 202509 file, complete from 1971-03-12), pinned in catalog/finance-data-lock.json.)

## Limitations

- Assets are 49 industry portfolios rather than individual stocks, so the cross-section is smaller and less noisy than in Ledoit and Wolf's experiments; N/T = 0.39 with the fixed 126-day window.
- The evaluation period is public and fixed; estimators tuned on it overfit, and a candidate that reads the prepared file could look ahead. The interface only passes the trailing window.
- Unconstrained minimum-variance weights are used (short positions allowed, no transaction costs or turnover limits), and returns are value-weighted CRSP portfolio returns of one data vintage.
- The score is a single realized standard deviation over 54 years; differences of a few hundredths of a percentage point are within sampling error.
