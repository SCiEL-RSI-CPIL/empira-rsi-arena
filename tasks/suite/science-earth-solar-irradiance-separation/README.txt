# Earth science: diffuse solar irradiance from global irradiance at unseen stations

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-solar-irradiance-separation`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Learn a separation model from 1-minute measurements of global, direct and diffuse solar irradiance at four NOAA SURFRAD stations and estimate the diffuse irradiance at three other stations from their global irradiance, solar geometry and meteorology on 48 days of 2022-2023. The score is the root-mean-square error of the estimated diffuse irradiance.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

estimate_diffuse(train, test) -> dict {station: float array (n_targets,)} of 1-minute diffuse horizontal irradiance (W m-2, 0 to 1500) at the target minutes of each test station. train (Bondville, Desert Rock, Fort Peck, Goodwin Creek: bon, dra, fpk, gwn) and test (Penn State, Sioux Falls, Table Mountain: psu, sxf, tbl) map station codes to dicts with lat, lon, elevation_m and arrays over every daytime minute (solar zenith < 90 deg) of days 8, 23, ..., 353 of 2022 and 2023: day (days since 2022-01-01), minute (UTC minute of day), zen (solar zenith angle, deg), ghi (global horizontal irradiance), e0_horizontal (1361 (1 + 0.033 cos(2 pi doy/365)) cos(zen)), temp (deg C), rh (%), pres (hPa), NaN where SURFRAD's quality flag is not 0, and target_index (float indices of the scored minutes). train also holds dni (direct normal) and dhi (diffuse) irradiance; test does not. Measurements at any minute of a test day may be used. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 180 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: root-mean-square error (W m-2) over the 87316 test-station target minutes (zenith < 85 deg, GHI >= 10 W m-2, all three components flagged good and closing within max(20 W m-2, 8 percent of GHI)); lower is better. Missing stations, wrong shapes, NaN or values outside [0, 1500] are invalid

## State of the art

No published value exists for this station split and day sample. Internal references fitted on the training stations: median diffuse fraction in clearness-index bins 75.8 W m-2; the seed (logistic diffuse fraction in the clearness index) 77.2 W m-2 (normalised RMSE 47 percent of the mean diffuse irradiance).

Reference value: no documented value for this finite instance (reviewed 2026-10-11).

- [Estimating 1-min beam and diffuse irradiance from the global irradiance: A review and an extensive worldwide comparison of latest separation models at 126 stations](https://doi.org/10.1016/j.rser.2022.112195)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Estimate the diffuse (and hence beam) component of solar irradiance from global irradiance measurements.

Why it matters: Global irradiance is measured or modelled far more widely than its components, but solar-energy yield on tilted or tracking surfaces, building energy and daylighting, and radiation climatology all need the diffuse and beam parts.

- [Estimation of the diffuse radiation fraction for hourly, daily and monthly-average global radiation](https://doi.org/10.1016/0038-092X(82)90302-4): Main research objective: Develop correlations that estimate the diffuse fraction of global radiation. Task connection: The task scores separation models of the same kind on 1-minute data at unseen stations.
- [Estimating 1-min beam and diffuse irradiance from the global irradiance: A review and an extensive worldwide comparison of latest separation models at 126 stations](https://doi.org/10.1016/j.rser.2022.112195): Main research objective: Review and compare separation models for 1-minute data worldwide. Task connection: Same problem and resolution; the task evaluates on three held-out SURFRAD stations.

Both cited papers have separating global irradiance into diffuse and beam components as their main goal, the problem the local task scores; no reproduction of their numbers is claimed.

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
- `file:data/earth/solar/surfrad_1min_7_stations_48_days.json.gz`

## Sources

- paper: https://doi.org/10.1016/0038-092X(82)90302-4 (Erbs, Klein and Duffie (1982): diffuse fraction as a function of the clearness index, the classic separation model.)
- paper: https://doi.org/10.1016/j.renene.2009.07.018 (Ridley, Boland and Lauret (2010): logistic separation model with multiple predictors (the seed is its single-predictor form).)
- paper: https://doi.org/10.1016/j.rser.2022.112195 (Yang (2022): review and worldwide comparison of 1-minute separation models, the evaluation setting followed here.)
- paper: https://doi.org/10.1175/1520-0477(2000)081<2341:SANSRB>2.3.CO;2 (Augustine et al. (2000): the SURFRAD network and its measurements.)
- archive: https://gml.noaa.gov/aftp/data/radiation/surfrad/ (SURFRAD 1-minute daily files of seven stations, 48 days of 2022-2023 (public domain; retrieved 2026-10-11).)

## Limitations

- Only 48 sampled days per station (every 15th day of 2022 and 2023) are used, and the three test stations are mid-latitude continental U.S. sites; separation models behave differently in other climates.
- The truth is the measured diffuse irradiance after quality control (flags, closure within 8 percent or 20 W m-2); measurement uncertainty of a few percent remains, and rejected minutes (often broken clouds or instrument issues) are not scored.
- No clear-sky irradiance model or aerosol data are given; candidates must build any clear-sky reference themselves from the inputs.
- A better score is a better separation model for these stations, not a scientific discovery; the data are public, so search feedback can overfit them.
- The candidate runs in a sandboxed child process that never receives the test stations' direct or diffuse irradiance; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
