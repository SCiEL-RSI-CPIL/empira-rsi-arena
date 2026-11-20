# Earth science: Gutenberg-Richter b-value from incomplete earthquake catalogues

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-earthquake-b-value-completeness`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Estimate the Gutenberg-Richter b-value and the magnitude of completeness of 600 synthetic earthquake catalogues whose small events are missed according to a realistic detection probability. The score is the mean absolute error of the estimated b-values against the generating values.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

estimate_b_and_mc(catalogues) -> float array (600, 2) of [b-value, magnitude of completeness] for each catalogue. catalogues is a list of 600 float arrays of earthquake magnitudes reported in 0.1 bins. Each synthetic catalogue has Gutenberg-Richter magnitudes (b drawn from 0.6-1.4) starting 3 sigma below mu and thinned by the detection probability Phi((m - mu) / sigma) (mu from 0.5-3.0, sigma from 0.05-0.5), with 150-5000 detected events. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 120 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: mean absolute error of b over the 600 catalogues; lower is better. The completeness estimate is compared with the 95 percent detection magnitude mu + 1.645 sigma and reported but not scored. Wrong shapes, NaN, b outside [0.1, 3] or Mc outside [-2, 10] are invalid

## State of the art

No published value exists for this generator. The seed (maximum curvature + 0.2, Aki-Utsu estimator) has a b-value mean absolute error of 0.097 with a bias of -0.077; the Aki-Utsu estimator applied above the true 95 percent detection magnitude, which a candidate does not know, has 0.069.

Reference value: no documented value for this finite instance (reviewed 2026-10-11).

- [Assessing the Quality of Earthquake Catalogues: Estimating the Magnitude of Completeness and Its Uncertainty](https://doi.org/10.1785/0120040007)

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `natural_science`, data `synthetic`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Estimate the Gutenberg-Richter b-value and the magnitude of completeness of an earthquake catalogue.

Why it matters: The b-value sets the expected rate of large earthquakes relative to small ones in every seismic hazard and earthquake forecast model, and its estimate is biased unless the completeness magnitude is handled correctly.

- [Assessing the Quality of Earthquake Catalogues: Estimating the Magnitude of Completeness and Its Uncertainty](https://doi.org/10.1785/0120040007): Main research objective: Compare methods for estimating the magnitude of completeness and quantify their effect on b-value estimates. Task connection: The task scores b-value and completeness estimators on synthetic incomplete catalogues, as the paper does.
- [Analysis of temporal and spatial heterogeneity of magnitude frequency distribution inferred from earthquake catalogues](https://doi.org/10.1111/j.1365-246X.1993.tb04663.x): Main research objective: Estimate the magnitude-frequency distribution, including the detection rate of small events, from catalogues. Task connection: The catalogues are generated with the paper's detection model; candidates may estimate it.

Both cited papers have estimating the magnitude-frequency distribution and completeness of catalogues as their main goal, the problem the local task scores; no reproduction of their numbers is claimed.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Scores are recomputed from pinned public data prepared by scripts/prepare_earth.py; held-out data never enter the candidate's sandboxed process. The held-out observations are public, so a fixed candidate should be checked on later data before any scientific claim.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/earth/completeness/synthetic_catalogues.json.gz`

## Sources

- paper: https://doi.org/10.1785/0120040007 (Woessner and Wiemer (2005): methods to estimate the magnitude of completeness and their effect on b-value estimates, compared on synthetic and real catalogues; the seed is their maximum-curvature method with the +0.2 correction.)
- paper: https://doi.org/10.1111/j.1365-246X.1993.tb04663.x (Ogata and Katsura (1993): Gutenberg-Richter magnitudes thinned by a cumulative-normal detection rate, the generating model of the catalogues.)

## Limitations

- The catalogues are synthetic: one stationary b-value and detection function per catalogue, no aftershock incompleteness, magnitude errors or mixed magnitude types; real catalogues are harder.
- The generating model (Gutenberg-Richter times cumulative-normal detection) is stated in the contract, so a candidate can fit it exactly; the score then measures statistical efficiency on this model rather than robustness to model error.
- The generator and its seed are in the evaluator's source; a candidate that re-implements them could reproduce the truth, so results are only meaningful for candidates that use the magnitudes alone.
- A better score is a better estimator on this benchmark, not a seismological discovery.
- The candidate runs in a sandboxed child process that never receives the generating parameters; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
