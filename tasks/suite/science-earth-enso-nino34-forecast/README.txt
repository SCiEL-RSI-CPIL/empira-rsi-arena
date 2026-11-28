# Earth science: El Nino-Southern Oscillation forecasts of the Nino 3.4 index

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-enso-nino34-forecast`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Fit a statistical ENSO forecast model on 1950-1999 ERSSTv5 sea-surface temperatures and, for every month from 2000 to 2024, forecast the monthly Nino 3.4 index for the next 12 months using only data up to that month. The score is the mean root-mean-square error over the 12 lead times.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fit_enso_model(training) -> model; model.forecast(history) -> float array (12,) of the Nino 3.4 area-mean sea-surface temperature (deg C, ERSSTv5) for each of the 12 months after the last month in history. training and history are dicts of numpy arrays over consecutive months: year, month (1-12), nino12, nino3, nino34 and nino4 (cosine-weighted mean SST, deg C, over grid cells centred in 0-10S 90-80W, 5N-5S 150-90W, 5N-5S 170-120W and 5N-5S 160E-150W) and sst (months, 31, 86): ERSSTv5 SST on its 2-degree grid for 30S-30N (latitude ascending) and 120E-70W (longitude 120 to 290 deg E), NaN over land; training also holds lat (31,) and lon (86,). training covers 1950-01 to 1999-12. Forecasts are then requested in order for every origin month from 2000-01 to 2024-12; history holds all months from 1950-01 up to and including the origin month and nothing later. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 120 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: mean over leads 1-12 of the root-mean-square error (deg C) over the 300 origins; lower is better. Wrong shapes, NaN or values outside [10, 40] deg C are invalid

## State of the art

No published value exists for this exact protocol (SST-only statistical forecasts from 2000-2024 origins, verified on locally computed ERSSTv5 Nino 3.4). The evaluator reports internal references fitted on 1950-1999: monthly climatology (mean RMSE 0.847 deg C), anomaly persistence (0.900) and damped persistence (0.650, identical to the seed).

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Skill of Real-Time Seasonal ENSO Model Predictions during 2002-11: Is Our Capability Increasing?](https://doi.org/10.1175/BAMS-D-11-00111.1)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict the El Nino-Southern Oscillation state (the Nino 3.4 sea-surface temperature index) months ahead.

Why it matters: ENSO is the dominant source of seasonal climate predictability; Nino 3.4 forecasts drive seasonal outlooks for rainfall, temperature, agriculture and disaster preparedness worldwide.

- [Deep learning for multi-year ENSO forecasts](https://doi.org/10.1038/s41586-019-1559-7): Main research objective: Produce skilful long-lead forecasts of ENSO (the Nino 3.4 index) with a statistical deep-learning model trained on SST and heat-content fields. Task connection: The task asks for the same kind of statistical Nino 3.4 forecast at 1-12 month leads from past sea-surface temperatures, verified on held-out years; it uses only SST inputs and a 2000-2024 verification period instead of the paper's data and protocol.
- [Skill of Real-Time Seasonal ENSO Model Predictions during 2002-11: Is Our Capability Increasing?](https://doi.org/10.1175/BAMS-D-11-00111.1): Main research objective: Assess the real-time skill of dynamical and statistical Nino 3.4 prediction models over 2002-2011. Task connection: Defines the operational verification setting (Nino 3.4 forecasts verified by lead time) that the task follows with a local RMSE score.

Both cited papers have ENSO prediction as their main goal, the problem the local task scores; no reproduction of their numbers is claimed.

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
- `file:data/earth/enso/ersstv5_tropical_pacific_1950_2025.json.gz`

## Sources

- paper: https://doi.org/10.1038/s41586-019-1559-7 (Ham, Kim and Luo (2019): statistical (deep-learning) ENSO forecasts of the Nino 3.4 index at leads up to 18 months, verified by lead-dependent correlation skill against observed SST.)
- paper: https://doi.org/10.1175/BAMS-D-11-00111.1 (Barnston et al. (2012): skill of real-time dynamical and statistical Nino 3.4 predictions during 2002-2011, the operational context for statistical ENSO forecasting.)
- paper: https://doi.org/10.1175/JCLI-D-16-0836.1 (Huang et al. (2017): ERSSTv5, the 2-degree monthly SST reconstruction whose tropical Pacific values and Nino-box means the task uses.)
- archive: https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc (NOAA ERSSTv5 monthly means distributed by NOAA PSL (retrieved 2026-11-20; no constraints on use); months 1950-01 to 2025-12 are used.)

## Limitations

- Inputs are sea-surface temperatures only (Nino indices and the 2-degree tropical Pacific field); operational forecasts also use subsurface ocean heat content, winds and coupled dynamical models, so scores are not comparable with published operational skill.
- The Nino indices are computed locally as cosine-weighted means over ERSSTv5 cells centred in each box; they reproduce the familiar NOAA ONI peaks to about 0.1 deg C (checked for 1982/83, 1997/98, 2015/16 and 2023/24) but are not the official index files.
- The 300 monthly origins are strongly autocorrelated (about 25 independent years and only a handful of strong events), so score differences of a few hundredths of a degree are not statistically meaningful.
- ERSSTv5 is recomputed monthly by NOAA and every update revises recent years (NCEI rewrites its monthly files from January 2008 on; the update of November 2026 changed tropical Pacific cells in 52 months from December 2017, mostly one or two cells by 0.01 deg C, and left the Nino 3.4 series unchanged). The derived-file hash check then fails, so preparation needs an archived copy of the locked source file or a re-recorded lock, which changes inputs slightly.
- A better score is a better statistical forecast model on these years, not a scientific discovery; the held-out observations are public, so search feedback on this fixed split can overfit it.
- The candidate runs in a sandboxed child process that never receives verifying months, and no statement of the candidate is executed in the evaluator's process; outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
