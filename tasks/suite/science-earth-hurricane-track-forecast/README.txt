# Earth science: Atlantic hurricane track forecasting from the best track

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-hurricane-track-forecast`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Fit a statistical tropical-cyclone track model on the 1950-2014 Atlantic best track and forecast 12-120 h centre positions for every advisory-time record of the held-out 2015-2024 storms. The score is the mean great-circle track error over seven lead times.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fit_track_model(training_storms) -> model; model.forecast(history) -> float array (7, 2) of [latitude deg N, longitude deg E] forecast centre positions at 12, 24, 36, 48, 72, 96 and 120 h. Storms are dicts with numpy arrays time_hours (hours since 1950-01-01 00 UTC), month, day, hour, day_of_year, lat (deg N), lon (deg E, west negative), vmax_kt, mslp_hpa (NaN when missing), plus year (int) and status (list of HURDAT2 codes TD, TS, HU, EX, SD, SS, LO, WV, DB), one entry per 6-hourly synoptic record. training_storms are the 1950-2014 Atlantic HURDAT2 storms. Forecasts are requested in chronological order for every 6-hourly record of a held-out 2015-2024 storm at which it is TD/TS/HU/SD/SS and that verifies at one or more leads; history holds that storm's records up to and including the forecast time and nothing later. A lead verifies only if the storm is still TD/TS/HU/SD/SS at the verifying time (NHC convention). The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 120 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: mean over the seven leads of the mean great-circle distance (km, sphere radius 6371 km) between forecast and best-track positions; lower is better. Latitudes outside [-90, 90], longitudes outside [-360, 360], NaN or wrong shapes are invalid

## State of the art

No published value exists for this exact protocol. For context, NHC reports 2019-2023 mean Atlantic official track errors of 23.9, 36.5, 49.3, 63.4, 93.4, 132.9 and 190.4 n mi at 12, 24, 36, 48, 72, 96 and 120 h (forecasts built with full numerical guidance) and CLIPER5 errors of 45.7, 97.1, 153.0, 205.4, 297.8, 372.7 and 439.1 n mi on a different (operational, homogeneous) sample.

Reference value: no documented value for this finite instance (reviewed 2026-10-08).

- [National Hurricane Center Forecast Verification Report, 2024 Hurricane Season (Table 2)](https://www.nhc.noaa.gov/verification/pdfs/Verification_2024.pdf)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict where an Atlantic tropical cyclone's centre will be one to five days ahead.

Why it matters: Track forecasts drive hurricane watches, warnings and evacuations; climatology-and-persistence models are the reference against which forecast skill is measured.

- [Five-Day Tropical Cyclone Track Forecasts in the North Atlantic Basin](https://doi.org/10.1175/1520-0434(1998)013<1005:FDTCTF>2.0.CO;2): Main research objective: Develop and evaluate a statistical climatology-and-persistence model (CLIPER5) that forecasts Atlantic tropical cyclone tracks out to five days. Task connection: The task asks for the same kind of history-only statistical track forecast, verified against the best track at 12-120 h; it uses the 2015-2024 seasons and a simplified verification rather than the paper's sample.

The cited paper's main goal is forecasting tropical cyclone tracks from climatology and persistence, which is what the local task scores; no reproduction of its numbers is claimed.

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

- paper: https://doi.org/10.1175/1520-0434(1998)013<1005:FDTCTF>2.0.CO;2 (Aberson (1998): CLIPER5, the climatology-and-persistence five-day Atlantic track forecast used as the no-skill baseline.)
- paper: https://doi.org/10.1175/MWR-D-12-00254.1 (Landsea and Franklin (2013): HURDAT2 format, content and best-track uncertainty.)
- archive: https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt (HURDAT2 Atlantic best track, release of 4 April 2025 (1851-2024); public domain; the 1950-2024 synoptic records are used.)
- benchmark: https://www.nhc.noaa.gov/verification/pdfs/Verification_2024.pdf (NHC 2024 forecast verification report: Tables 2 and 4 give 2019-2023 mean official, CLIPER5 and Decay-SHIFOR5 errors by lead.)

## Limitations

- Only best-track history is available to the candidate (position, intensity, pressure, stage and date); operational forecasts also use satellite and reconnaissance analyses and numerical weather prediction, so scores are not comparable with official NHC errors.
- Inputs are the post-season best track rather than the real-time operational estimates; verification is at 6-hourly synoptic times only, and off-synoptic landfall records are dropped.
- A better score is a better statistical track model on these seasons, not a scientific discovery.
- The held-out observations are public, so search feedback on this fixed split can overfit it; freeze a candidate and test it on later or independent data before claiming a scientific improvement.
- The candidate runs in a sandboxed child process that receives a storm's best track only up to the forecast time, and no statement of the candidate is executed in the evaluator's process; outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
