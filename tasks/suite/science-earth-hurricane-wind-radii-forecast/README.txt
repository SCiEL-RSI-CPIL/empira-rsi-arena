# Earth science: Atlantic hurricane wind-radii (storm size) forecasting

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-hurricane-wind-radii-forecast`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Fit a model of tropical-cyclone size on the 2004-2014 Atlantic best track and forecast the radius of 34-kt winds in four quadrants 12-72 h ahead for every analysed time of the 2015-2024 storms, given the storm's history and its best-track track and intensity at the verifying times. The score is the mean absolute radius error.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fit_radii_model(training_storms) -> model; model.forecast(history, future) -> float array (5, 4) of 34-kt wind radii (n mi) in the NE, SE, SW and NW quadrants at 12, 24, 36, 48 and 72 h. training_storms is a list of the 2004-2014 Atlantic HURDAT2 storms as dicts with numpy arrays time_hours (hours since 2004-01-01 00 UTC), lat (deg N), lon (deg E, west negative), vmax_kt, mslp_hpa (NaN when missing) and r34, r50, r64 ((n, 4) radii in n mi for NE, SE, SW, NW; 0 where the threshold is not reached in a quadrant, NaN when not analysed), plus status (list of HURDAT2 codes) and year, one entry per 6-hourly synoptic record. Forecasts are requested in chronological order from every 6-hourly record of a 2015-2024 storm that is TS, HU or SS with maximum wind >= 34 kt and analysed 34-kt radii and that verifies at one or more leads; history holds that storm's records up to and including the forecast time (same fields); future holds lead_hours, verifies (1 or 0) and the best-track lat, lon, vmax_kt and mslp_hpa at each lead (NaN where the lead does not verify). A lead verifies when the storm is then TS, HU or SS with maximum wind >= 34 kt and analysed 34-kt radii. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 120 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: mean over the five leads of the mean absolute error (n mi) over verifying cases and the four quadrants; lower is better. Wrong shapes, NaN or values outside [0, 1000] n mi are invalid

## State of the art

No published value exists for this perfect-prognosis protocol. Internal references: persistence of the current radii (the seed) 32.19 n mi; least-squares regression per lead on current radii, intensity change, verifying intensity and latitude, fitted on 2004-2014, 27.88 n mi.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Statistical Tropical Cyclone Wind Radii Prediction Using Climatology and Persistence](https://doi.org/10.1175/WAF1026.1)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict the size of a tropical cyclone's wind field (the radii of damaging winds) one to three days ahead.

Why it matters: Wind radii determine the extent of watches and warnings, storm-surge and wave forecasts and the arrival time of tropical-storm-force winds, which sets evacuation timing.

- [Statistical Tropical Cyclone Wind Radii Prediction Using Climatology and Persistence](https://doi.org/10.1175/WAF1026.1): Main research objective: Forecast tropical-cyclone wind radii statistically from climatology, persistence and the track and intensity forecast. Task connection: The task scores the same kind of radii forecast at 12-72 h, using best-track instead of forecast track and intensity on the 2015-2024 Atlantic seasons.
- [A Global Statistical-Dynamical Tropical Cyclone Wind Radii Forecast Scheme](https://doi.org/10.1175/WAF-D-16-0168.1): Main research objective: Develop and evaluate a global statistical-dynamical scheme for forecasting wind radii. Task connection: Same forecast target and verification against best-track radii.

Both cited papers have wind-radii forecasting as their main goal, the problem the local task scores; no reproduction of their numbers is claimed.

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
- `file:data/earth/cyclones/hurdat2_atlantic_radii_2004_2024.json.gz`

## Sources

- paper: https://doi.org/10.1175/WAF1026.1 (Knaff, Sampson and DeMaria (2007): climatology-and-persistence prediction of tropical-cyclone wind radii driven by the track and intensity forecast.)
- paper: https://doi.org/10.1175/WAF-D-16-0168.1 (Knaff et al. (2017): a global statistical-dynamical wind-radii forecast scheme.)
- paper: https://doi.org/10.1175/MWR-D-12-00254.1 (Landsea and Franklin (2013): HURDAT2 format, including the four-quadrant wind radii since 2004, and their analysis uncertainty.)
- archive: https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt (HURDAT2 Atlantic best track, release of 4 April 2025 (public domain); the 2004-2024 synoptic records with wind radii are used.)

## Limitations

- Best-track wind radii are post-season analyses with large uncertainty (often tens of percent for the 34-kt radii), which bounds the attainable error.
- The verifying track, maximum wind and pressure are given (perfect prognosis); operational radii forecasts must use forecast track and intensity, so errors here are smaller than operational ones and not comparable with them.
- Only 34-kt radii are scored, at 6-hourly synoptic times; 50- and 64-kt radii and the radius of maximum wind are available as history but not forecast.
- A better score is a better statistical storm-size model on these seasons, not a scientific discovery; the held-out observations are public, so search feedback on this split can overfit it.
- The candidate runs in a sandboxed child process that receives each case's information only when it is requested; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
