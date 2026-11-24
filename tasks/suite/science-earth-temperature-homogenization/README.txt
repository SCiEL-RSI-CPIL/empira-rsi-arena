# Earth science: homogenisation of monthly temperature networks

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-temperature-homogenization`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Remove non-climatic breaks, drifts and outliers from 40 synthetic networks of ten correlated 100-year monthly temperature records, as climate-data centres do before estimating trends. The score is the mean centred root-mean-square difference between the homogenised and the true annual means, which the generator knows exactly.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

homogenize(networks, calendar) -> list of 40 float arrays (10, 1200): homogenised monthly temperatures (deg C) of each network's stations. networks is a list of 40 arrays (10 stations, 1200 months) of observed monthly mean temperatures (deg C, NaN where missing) of 100-year synthetic networks; calendar holds year (1901-2000) and month (1-12) arrays (1200,). The observations contain non-climatic breaks (about five per century per station, sizes of standard deviation 0.8 deg C with a seasonal cycle; the most recent segment is unbiased), occasional network-wide clustered breaks, gradual drifts, outliers (1 percent, 3-5 deg C) and gaps (late starts, 2 percent random). Returned values at missing months are ignored. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 120 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: mean over the 400 stations of the centred root-mean-square difference between homogenised and true annual means (years with at least 10 observed months, computed over the observed months); lower is better. A wrong list length or array shape, or non-finite or implausible (|value| >= 100) values at observed months are invalid

## State of the art

No published value exists for this generator. Unhomogenised data score 0.614 deg C (annual CRMSD) and the seed, SNHT binary segmentation of station-minus-neighbour-median series with outlier replacement, 0.254 deg C; the network-mean trend error falls only from 0.50 to 0.44 deg C per century.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Benchmarking homogenization algorithms for monthly data](https://doi.org/10.5194/cp-8-89-2012)

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `natural_science`, data `synthetic`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Detect and correct non-climatic inhomogeneities in long station temperature records so that climate trends can be estimated reliably.

Why it matters: Station relocations and instrument changes introduce jumps comparable to a century of warming; global and national temperature records depend on homogenisation, and its errors propagate into trend estimates.

- [Benchmarking homogenization algorithms for monthly data](https://doi.org/10.5194/cp-8-89-2012): Main research objective: Compare the performance of homogenisation algorithms for monthly climate data on a realistic benchmark with known truth. Task connection: The task is a HOME-style benchmark: synthetic networks with inserted inhomogeneities, scored by the centred RMS difference of annual means.
- [Benchmarking the performance of pairwise homogenization of surface temperatures in the United States](https://doi.org/10.1029/2011JD016761): Main research objective: Assess how well pairwise homogenisation recovers true temperature trends in synthetic benchmark worlds. Task connection: Same problem (recovering homogeneous series and network trends from inhomogeneous ones); the task reports the network trend error too.

Both cited papers have the homogenisation of climate records as their main goal, the problem the local task scores; no reproduction of their numbers is claimed.

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
- `file:data/earth/homogenisation/synthetic_networks.json.gz`

## Sources

- paper: https://doi.org/10.5194/cp-8-89-2012 (Venema et al. (2012): the HOME benchmark of homogenisation algorithms for monthly data (surrogate networks, inserted breaks, outliers, drifts and gaps; centred RMSE of annual means), whose design the generator follows.)
- paper: https://doi.org/10.1029/2011JD016761 (Williams, Menne and Thorne (2012): benchmarking pairwise homogenisation of U.S. surface temperatures with synthetic worlds.)
- paper: https://doi.org/10.1002/joc.3370060607 (Alexandersson (1986): the standard normal homogeneity test used by the seed.)

## Limitations

- The networks are synthetic: an AR(1) regional signal with a random trend, AR(1) station noise and simple annual cycles, with inhomogeneities drawn from fixed distributions; real networks have more varied correlation structures, break statistics and metadata.
- The generator and its seed are in the evaluator's source; a candidate that re-implements them could reproduce the truth, so results are only meaningful for candidates that use the observations alone.
- No station metadata (documented relocations or instrument changes) are given, unlike in practical homogenisation.
- A better score is a better homogenisation method on this benchmark, not a statement about real climate records.
- The candidate runs in a sandboxed child process that never receives the true series, and no statement of the candidate is executed in the evaluator's process; outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
