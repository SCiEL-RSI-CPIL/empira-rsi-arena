# Finance: Optimal market making with inventory risk

**Source collection:** [Finance and economics research: source-based research tasks](../../papers/science-finance_expansion.md)\
**Task id:** `science-finance-market-making`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Set bid and ask quotes around the mid price in the Avellaneda-Stoikov limit-order-book model to earn the spread while controlling inventory risk, in three simulated market regimes. The evaluator simulates fills on fixed random numbers and scores the risk-adjusted profit relative to the exact optimal quoting policy.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

quote(t, inventory, mid, params) -> (bid_depths, ask_depths): two float arrays of shape (4000,), one entry per simulated path, the distances below and above the mid price at which to quote in this step (np.inf = no quote; values >= -3/k). Called at every step n = 0..999 of each regime with t = n*dt, inventory (4000,) ints in [-Q, Q], mid (4000,) prices and params = {'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'}. Model: the bid is hit with probability 1 - exp(-A exp(-k bid_depth) dt) and the ask lifted likewise, independently; each fill earns its depth and moves inventory by one unit within |q| <= Q = 30 (quotes that would breach the bound are ignored); the mid moves by sigma*sqrt(dt)*N(0,1). Objective per path: spread captured minus (gamma/2)*sigma^2*sum(q^2 dt). Score: mean over three regimes (Avellaneda-Stoikov's example sigma 2, A 140, k 1.5, T 1 with gamma 0.05 and 0.5, and a thinner market) of the mean objective divided by that of the exact optimal policy of this discrete model on the same random numbers (maximize; about 1 at the optimum)

## State of the art

The exact optimal policy of the discrete model is computed by dynamic programming and scores 1 by construction; the closed-form approximation of Gueant, Lehalle and Fernandez-Tapia (model B form of Gueant 2017) scores 0.998 and the inventory-blind seed 0.742.

Reference value: proven optimum `1.0` (reviewed 2026-10-08).

- [Dealing with the inventory risk: a solution to the market making problem](https://doi.org/10.1007/s11579-012-0087-0)
- [Optimal market making](https://doi.org/10.1080/1350486x.2017.1342552)

## Labels

Scientific contribution: `rediscovery` (tier 2); artifact `policy`, domain `finance`, data `simulator`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `finance`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Choose bid and ask quotes that earn the bid-ask spread while controlling the inventory risk of a market maker.

Why it matters: Market makers provide liquidity in most electronic markets; their quoting determines spreads paid by investors, and the inventory-risk trade-off is a central problem of market microstructure.

- [High-frequency trading in a limit order book](https://doi.org/10.1080/14697680701381228): Main research objective: Derive optimal bid and ask quotes for a dealer facing inventory risk in a limit order book. Task connection: The task uses the paper's model and its simulation parameters.
- [Dealing with the inventory risk: a solution to the market making problem](https://doi.org/10.1007/s11579-012-0087-0): Main research objective: Solve the market-making problem with inventory constraints and derive closed-form approximations. Task connection: The task's inventory bounds and closed-form reference come from this line of work.

Both papers have optimal market making as their main research goal.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Avellaneda-Stoikov dealer model in discrete time with the risk-adjusted objective of Gueant (2017) (running inventory penalty instead of exponential utility) and inventory bounds as in Gueant, Lehalle and Fernandez-Tapia (2013). The normalizing policy is the exact optimum of the discrete model by backward induction. Checks: the dynamic-programming value agrees with the simulated mean of its own policy within two standard errors in every regime; the closed-form approximation 1/k +- (2q +- 1)/2 * sqrt(gamma sigma^2 e / (2 A k)) scores 0.998 and the inventory-blind seed 0.742. Scores are Monte Carlo estimates on 4000 fixed paths per regime.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://doi.org/10.1080/14697680701381228 (Avellaneda and Stoikov (2008): dealer quoting model with Brownian mid price and fill intensity A exp(-k depth); their simulation example (sigma 2, A 140, k 1.5, T 1).)
- paper: https://doi.org/10.1007/s11579-012-0087-0 (Gueant, Lehalle and Fernandez-Tapia (2013): inventory bounds, exact solution through a linear system and closed-form approximations of the optimal quotes.)
- paper: https://doi.org/10.1080/1350486x.2017.1342552 (Gueant (2017): the risk-adjusted expectation objective (running penalty (gamma/2) sigma^2 q^2) and its closed-form quote approximation.)

## Limitations

- The market is simulated: Brownian mid price, independent Poisson fills with exponential intensity, no adverse selection, latency, queue position or tick size. Good scores do not imply profitable trading in real limit-order books.
- The optimal policy of this discrete model is known (computed by dynamic programming), so a score of about 1 reproduces a known solution; the closed-form approximation of the literature already scores 0.998.
- Scores are Monte Carlo estimates on 4000 fixed paths per regime; the random stream is fixed and visible in the evaluator source, so a candidate that reproduces it could anticipate fills (treat this as cheating).
