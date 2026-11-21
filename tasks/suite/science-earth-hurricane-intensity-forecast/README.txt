# Earth science: Atlantic hurricane intensity forecasting from the best track

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-hurricane-intensity-forecast`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Fit a statistical tropical-cyclone intensity model on the 1950-2014 Atlantic best track and forecast the 12-120 h maximum sustained wind for every advisory-time record of the held-out 2015-2024 storms. The score is the mean absolute intensity error over seven lead times.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fit_intensity_model(training_storms) -> model; model.forecast(history) -> float array (7,) of forecast maximum sustained 1-min 10-m winds (kt) at 12, 24, 36, 48, 72, 96 and 120 h. Storms are dicts with read-only arrays time_hours (hours since 1950-01-01 00 UTC), month, day, hour, day_of_year, lat (deg N), lon (deg E, west negative), vmax_kt, mslp_hpa (NaN when missing), plus year (int) and status (tuple of HURDAT2 codes TD, TS, HU, EX, SD, SS, LO, WV, DB), one entry per 6-hourly synoptic record. training_storms are the 1950-2014 Atlantic HURDAT2 storms. Forecasts are requested in chronological order for every 6-hourly record of a held-out 2015-2024 storm at which it is TD/TS/HU/SD/SS and that verifies at one or more leads; history holds that storm's records up to and including the forecast time and nothing later. A lead verifies only if the storm is still TD/TS/HU/SD/SS at the verifying time (NHC convention). Score: mean over the seven leads of the mean absolute error (kt) against the best-track maximum wind; lower is better. Values outside [0, 300] kt, NaN or wrong shapes are invalid

## State of the art

No published value exists for this exact protocol. For context, NHC reports 2019-2023 mean Atlantic official intensity errors of 5.0, 7.3, 8.5, 9.7, 10.9, 12.9 and 15.5 kt at 12, 24, 36, 48, 72, 96 and 120 h and Decay-SHIFOR5 errors of 6.6, 10.2, 13.1, 15.6, 18.6, 21.8 and 22.6 kt on a different (operational, homogeneous) sample.

Reference value: no documented value for this finite instance (reviewed 2026-10-08).

- [National Hurricane Center Forecast Verification Report, 2024 Hurricane Season (Table 4)](https://www.nhc.noaa.gov/verification/pdfs/Verification_2024.pdf)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict the maximum sustained wind of an Atlantic tropical cyclone one to five days ahead.

Why it matters: Intensity determines wind damage and storm-surge risk, and intensity forecasts have improved more slowly than track forecasts; climatology-and-persistence models define the skill baseline.

- [Statistical, 5-Day Tropical Cyclone Intensity Forecasts Derived from Climatology and Persistence](https://doi.org/10.1175/1520-0434(2003)018<0080:SDTCIF>2.0.CO;2): Main research objective: Develop and evaluate statistical five-day tropical cyclone intensity forecasts from climatology and persistence predictors. Task connection: The task scores the same kind of history-only intensity forecast against the best track at 12-120 h on the 2015-2024 Atlantic seasons; it does not reproduce the paper's samples or inland decay component.

The cited paper's main goal is statistical tropical cyclone intensity prediction, the problem the local task scores; no reproduction of its numbers is claimed.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Scores are recomputed from the pinned public observations; candidates receive only training data and, at forecast time, information available before the verifying time. The held-out data are public, so a fixed candidate should be checked on later data before any scientific claim.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/earth/cyclones/hurdat2_atlantic_1950_2024.json.gz`

## Sources

- paper: https://doi.org/10.1175/1520-0434(2003)018<0080:SDTCIF>2.0.CO;2 (Knaff et al. (2003): SHIFOR5, five-day statistical intensity forecasts from climatology and persistence.)
- paper: https://doi.org/10.1175/MWR-D-12-00254.1 (Landsea and Franklin (2013): HURDAT2 format, content and best-track uncertainty.)
- archive: https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt (HURDAT2 Atlantic best track, release of 4 April 2025 (1851-2024); public domain; the 1950-2024 synoptic records are used.)
- benchmark: https://www.nhc.noaa.gov/verification/pdfs/Verification_2024.pdf (NHC 2024 forecast verification report: Tables 2 and 4 give 2019-2023 mean official, CLIPER5 and Decay-SHIFOR5 errors by lead.)

## Limitations

- Only best-track history is available (no sea-surface temperature, wind shear, land mask or forecast track), so rapid intensification and landfall decay can be anticipated only from history, date and position; official forecasts use far more information.
- Best-track maximum winds are rounded to 5 kt and carry analysis uncertainty of roughly 10 kt or more for weaker systems (Landsea and Franklin 2013), which bounds achievable errors.
- A better score is a better statistical intensity model on these seasons, not a scientific discovery.
- The held-out observations are public, so search feedback on this fixed split can overfit it; freeze a candidate and test it on later or independent data before claiming a scientific improvement.
- Candidate and evaluator share one Python process; the API withholds held-out data but is not a security sandbox.
