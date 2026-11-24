# Finance: Arbitrage-free SVI implied-volatility surface for DAX options

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-finance-arbitrage-free-svi-surface`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Fit a raw SVI smile to each of the eight maturities of a real DAX option implied-volatility surface (2002-07-05, 104 quotes) without static arbitrage. The evaluator checks butterfly and calendar-spread arbitrage on a dense moneyness grid and scores the root-mean-square implied-volatility error in volatility points.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fit_surface(quotes) -> (8, 5) array of raw SVI parameters [a, b, rho, m, sigma], one row per maturity in increasing order; total implied variance w(k) = a + b*(rho*(k - m) + sqrt((k - m)^2 + sigma^2)) in log-forward moneyness k = ln(K/F). quotes = {'spot': 4468.17, 'maturities': (8,) years (days/365), 'maturity_days', 'zero_rates', 'forwards' (spot*exp(r*T), no dividends), 'strikes': (13,), 'implied_vol': (8, 13), 'log_moneyness': (8, 13), 'total_variance': (8, 13)} for DAX index options on 2002-07-05. Valid only without static arbitrage: b >= 0, |rho| < 1, sigma >= 1e-4, a + b*sigma*sqrt(1 - rho^2) > 0, wing slopes b(1 + |rho|) <= 2, Durrleman's g(k) >= -1e-10 and w_{i+1}(k) - w_i(k) >= -1e-10 on a dense grid of |k| <= 6 refined near every vertex, and wing slopes b(1 +- rho) non-decreasing in maturity. Score: root-mean-square error of the fitted implied volatilities sqrt(w/T) against the 104 quotes, in volatility points (minimize). The candidate runs in a separate sandboxed process: numpy, scipy, scikit-learn, pandas and the other installed packages are importable, while the evaluator's library, its data files, the network and subprocesses are not accessible; one thread and 3 GB of memory. It receives only the arguments described here, one call at a time: the arguments of a call are sent after the value of the previous call has been received. Arguments and return values travel as numbers, strings, lists, dicts with string keys and numeric numpy arrays, so a tuple arrives as a list

## State of the art

No published SVI fit to this surface is known. An unconstrained per-slice SVI least-squares fit reaches about 0.275 volatility points but violates the arbitrage conditions; a simple slice-by-slice constrained fit reaches about 0.284; the flat seed scores about 5.95. For comparison, QuantLib's five-parameter Heston calibration to the same quotes has squared error 177.2 (about 1.31 volatility points RMSE with week-rounded maturities).

Reference value: no documented value for this finite instance (reviewed 2026-10-06).

- [Arbitrage-free SVI volatility surfaces](https://doi.org/10.1080/14697688.2013.819986)
- [QuantLib test-suite/hestonmodel.cpp (testDAXCalibration)](https://github.com/lballabio/QuantLib/blob/ea1ca12e7311813abc42353ab88e127b60ee9848/test-suite/hestonmodel.cpp)

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `finance`, data `real_measurements`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Construct an implied-volatility surface that fits option quotes closely while excluding static arbitrage (butterfly and calendar spreads).

Why it matters: Arbitrage-free volatility surfaces are required to price and hedge exotic and illiquid options consistently; fits that admit arbitrage imply negative probabilities and exploitable price inconsistencies.

- [Arbitrage-free SVI volatility surfaces](https://doi.org/10.1080/14697688.2013.819986): Main research objective: Characterize and construct SVI implied-volatility surfaces free of static arbitrage and calibrate them to option data. Task connection: The task scores raw SVI slices fitted to a real DAX surface under the paper's butterfly (Durrleman) and calendar-spread conditions.

The cited paper's main goal is exactly arbitrage-free calibration of SVI implied-volatility surfaces.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Static no-arbitrage conditions of Gatheral and Jacquier (2014) for raw SVI slices, checked numerically on a dense log-moneyness grid with tolerance 1e-10 plus exact wing conditions. Reference checks: the evaluator flags Gatheral and Jacquier's arbitrageable example (Vogt's parameters) with g(k) < 0 for k in about [0.64, 1.26]; the sign of g agrees with an independent finite-difference density of the implied Black prices; and an independent Heston pricer and Levenberg-Marquardt calibration on the same DAX quotes reproduces QuantLib's published calibration error 177.2 (177.23), validating forwards, rates and maturities. The arbitrage check is a dense numerical check on |k| <= 6, not a proof for every strike.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/finance/dax_2002_surface.csv`

## Sources

- paper: https://doi.org/10.1080/14697688.2013.819986 (Gatheral and Jacquier (2014): raw SVI parameterization, Durrleman's condition g(k) >= 0 for butterfly arbitrage (Lemma 2.2), calendar-spread condition on total variance, wing bound, and the arbitrageable Vogt example (Section 3).)
- repository: https://github.com/lballabio/QuantLib/blob/ea1ca12e7311813abc42353ab88e127b60ee9848/test-suite/hestonmodel.cpp (QuantLib test-suite/hestonmodel.cpp at a pinned commit: getDAXCalibrationMarketData() (13 strikes x 8 maturities, spot 4468.17, zero rates) and testDAXCalibration's reference error 177.2.)
- paper: https://doi.org/10.2139/ssrn.1412333 (A. Sepp, Pricing European-style options under jump diffusion processes with stochastic volatility: QuantLib's test attributes the DAX quotes of 2002-07-05 to this paper.)

## Limitations

- One historical surface with 104 quotes (mid implied volatilities without bid-ask spreads); a better fit on this surface is a development result, and quotes may themselves contain small arbitrages or rounding to four decimals.
- The no-arbitrage check is numerical: Durrleman's g and calendar spreads are evaluated on a dense grid of |k| <= 6 refined near every SVI vertex with tolerance 1e-10, plus exact wing conditions; it is not a proof for every strike, and the surface between quoted maturities is not checked.
- The candidate is restricted to raw SVI slices; other arbitrage-free parameterizations (for example SSVI, which is a special case, or non-parametric surfaces) must be expressed as raw SVI per maturity.
- Forwards use QuantLib's zero rates and no dividends (the DAX is a total-return index); maturities use the original day counts rather than QuantLib's week-rounded ones.
- The candidate runs in a sandboxed child process that never receives later data, realised outcomes or reference solutions; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
