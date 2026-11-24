# Finance: Optimal execution under Almgren-Chriss market impact

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-finance-optimal-execution`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Schedule the sale of a large block of shares over a fixed horizon to minimize expected trading cost plus a variance penalty in the Almgren-Chriss market-impact model, including power-law impact and an intraday liquidity profile. The evaluator recomputes each schedule's cost-risk objective exactly and compares it with the known optimum on 12 fixed instances.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

trade_schedule(problem) -> array of N nonnegative trade sizes n_1..n_N (shares sold in each period) summing to X (within 1e-6 relative). problem = {'shares': X, 'periods': N, 'horizon_days': T, 'tau': T/N, 'price': 50.0, 'sigma': (N,) volatility per period ($/share/day^0.5), 'eta': (N,) temporary-impact coefficients, 'beta': impact exponent (1 = linear), 'epsilon': fixed cost ($/share), 'gamma': linear permanent impact ($/share^2), 'risk_aversion': lambda (1/$), 'name': str}. Discrete Almgren-Chriss objective: E = gamma*X^2/2 - gamma*sum(n_k^2)/2 + epsilon*sum|n_k| + sum_k tau*eta_k*|n_k/tau|^(1+beta); V = tau*sum_k sigma_k^2*x_k^2 with holdings x_k = X - (n_1+...+n_k); U = E + lambda*V. Twelve fixed instances: Almgren-Chriss Table 1 parameters with varied N, lambda and volatility, power-law impact (beta 0.6 and 0.5) and U-shaped intraday liquidity and volatility. Score: mean over instances of 100*(U - U*)/U*, where U* is the exact optimum (closed form for constant linear impact, otherwise a convex solve); minimize, 0 is optimal. The candidate runs in a separate sandboxed process: numpy, scipy, scikit-learn, pandas and the other installed packages are importable, while the evaluator's library, its data files, the network and subprocesses are not accessible; one thread and 3 GB of memory. It receives only the arguments described here, one call at a time: the arguments of a call are sent after the value of the previous call has been received. Arguments and return values travel as numbers, strings, lists, dicts with string keys and numeric numpy arrays, so a tuple arrives as a list

## State of the art

Every instance has a known optimum: the Almgren-Chriss closed-form trajectory for constant linear impact and the unique minimizer of a strictly convex program otherwise, so the optimal score is 0. The TWAP seed scores about 221 and the closed form applied only where it is exact (TWAP elsewhere) about 151.

Reference value: proven optimum `0.0` (reviewed 2026-10-06).

- [Optimal execution of portfolio transactions](https://doi.org/10.21314/JOR.2001.041)
- [Optimal execution with nonlinear impact functions and trading-enhanced risk](https://doi.org/10.1080/135048602100056)

## Labels

Scientific contribution: `rediscovery` (tier 2); artifact `program`, domain `finance`, data `synthetic`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Execute a large order over time so as to balance market-impact costs against the price risk of holding the position.

Why it matters: Institutional investors trade orders that move prices; optimal execution determines realized trading costs for funds and is the foundation of algorithmic execution and transaction-cost analysis.

- [Optimal execution of portfolio transactions](https://doi.org/10.21314/JOR.2001.041): Main research objective: Derive trading trajectories that minimize a combination of expected execution cost and its variance under market impact. Task connection: The task uses the paper's discrete model, objective and Table 1 parameters, and scores schedules against the paper's closed-form optimum where it applies.
- [Optimal execution with nonlinear impact functions and trading-enhanced risk](https://doi.org/10.1080/135048602100056): Main research objective: Characterize optimal execution strategies when temporary market impact is a nonlinear function of the trading rate. Task connection: The power-law instances follow this paper's nonlinear-impact extension (without its trading-enhanced risk term).

Both papers have optimal trade execution under market impact as their main research question.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Discrete-time model of Almgren and Chriss (2000) with the power-law temporary impact of Almgren (2003) and a stylized U-shaped intraday profile. Reference checks: for AC Table 1 parameters the closed form gives kappa = 0.607/day and kappa*T = 3.04 (AC state kappa about 0.6/day, kappa*T about 3); the closed form and the CVXPY/Clarabel convex solve agree to 1e-15 relative on constant-impact instances, and an independent SLSQP solve matches every reference optimum to 1e-10 relative. Permanent impact is set to zero in the power-law instances so that the objective stays convex; buying back shares is not allowed.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:cvxpy`

## Sources

- paper: https://doi.org/10.21314/JOR.2001.041 (Almgren and Chriss (2000/2001), Sections 2-3: discrete trading model, linear permanent and temporary impact, mean-variance objective, closed-form efficient trajectories, Table 1 parameters (S0 = 50, X = 10^6, T = 5 days, sigma = 0.95, epsilon = 0.0625, gamma = 2.5e-7, eta = 2.5e-6, lambda = 10^-6).)
- paper: https://doi.org/10.1080/135048602100056 (Almgren (2003): optimal execution with nonlinear (power-law) temporary impact functions.)

## Limitations

- This is a model-based rediscovery task: the optimum of every instance is known (closed form or a convex solve), so a perfect score reproduces known solutions and says nothing about real trading costs.
- Impact parameters follow the stylized Almgren-Chriss example rather than estimates for a specific stock; the intraday volume and volatility profiles are stylized U shapes, not measured curves.
- Prices follow an arithmetic random walk without drift or information; the schedule is static (no adaptation to realized prices), permanent impact is zero in the power-law instances to keep the problem convex, and buying back shares is not allowed.
- The reference optimum is computed numerically for six of the twelve instances; scores within about 1e-7 percent of zero are optimal to solver tolerance and can be marginally negative.
- The candidate runs in a sandboxed child process that never receives later data, realised outcomes or reference solutions; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
