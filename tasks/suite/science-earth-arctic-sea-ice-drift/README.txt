# Earth science: daily Arctic sea-ice drift from geostrophic winds

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-arctic-sea-ice-drift`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Learn how Arctic sea ice drifts in response to the wind from 2000-2014 drifting-buoy tracks and reanalysis sea-level pressure, then predict the daily velocity of every buoy north of 70 N for each day of 2015-2024 from that day's geostrophic wind and the past buoy record. The score is the root-mean-square vector error of the predicted velocities.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fit_drift_model(training) -> model; model.forecast(history, today) -> float array (n, 2) of the eastward and northward ice velocity (cm/s) of each buoy listed in today over that day (displacement from 00 UTC to 00 UTC of the next day). training and history are dicts of numpy arrays, one row per buoy-day, with day (days since 2000-01-01), buoy (id), lat and lon (00 UTC position, deg; buoys at 70-90 N from the IABP product C), slp_hpa, ug and vg (geostrophic wind, m/s east and north, from a weighted plane fit to NCEP/NCAR daily-mean sea-level pressure within 500 km, air density 1.3) and u, v (the observed velocity, cm/s); training covers 2000-2014. Forecasts are requested in order for every day of 2015-2024 with active buoys; today holds day, buoy, lat, lon, slp_hpa, ug and vg of that day's buoys (the geostrophic wind is the daily mean over the forecast day); history holds all rows of earlier days, including the previous day's velocities, and nothing later. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 240 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: root-mean-square vector error sqrt(mean((u - u_obs)^2 + (v - v_obs)^2)) in cm/s over all 356683 buoy-days; lower is better. Wrong shapes, NaN or values outside [-200, 200] cm/s are invalid

## State of the art

No published value exists for this protocol. The Thorndike-Colony complex linear model fitted on 2000-2014 (factor 1.03 percent of the geostrophic wind, turned 4.2 degrees to the right; the seed) scores 10.98 cm/s and explains 39 percent of the 2015-2024 velocity variance; zero drift scores 14.10 cm/s. Thorndike and Colony (1982) report that the geostrophic wind explains more than 70 percent of daily ice-velocity variance in the central Arctic.

Reference value: no documented value for this finite instance (reviewed 2026-10-11).

- [Sea ice motion in response to geostrophic winds](https://doi.org/10.1029/JC087iC08p05845)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict the drift of Arctic sea ice from atmospheric forcing.

Why it matters: Ice drift redistributes sea ice and freshwater, controls ice export through Fram Strait and is needed for navigation, search and rescue, and for the dynamics of sea-ice models.

- [Sea ice motion in response to geostrophic winds](https://doi.org/10.1029/JC087iC08p05845): Main research objective: Quantify how Arctic sea-ice motion responds to the geostrophic wind and the mean ocean circulation. Task connection: The task scores predictions of daily buoy drift from the geostrophic wind on held-out years; the seed is the paper's linear model.
- [Response of Sea Ice to the Arctic Oscillation](https://doi.org/10.1175/1520-0442(2002)015<2648:ROSITT>2.0.CO;2): Main research objective: Relate Arctic sea-ice motion and its changes to atmospheric circulation using IABP data. Task connection: Same observational basis (IABP buoys and sea-level pressure) and the same physical question of wind-driven drift.

The cited papers' main goal is understanding and predicting wind-driven sea-ice motion, the problem the local task scores; no reproduction of their numbers is claimed.

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
- `file:data/earth/icedrift/iabp_buoys_ncep_geostrophic_2000_2024.json.gz`

## Sources

- paper: https://doi.org/10.1029/JC087iC08p05845 (Thorndike and Colony (1982): sea-ice velocity as a linear response to the geostrophic wind (about 1 percent of the wind speed, a few degrees to the right) plus a mean current; the seed and reference use this model.)
- paper: https://doi.org/10.1175/1520-0442(2002)015<2648:ROSITT>2.0.CO;2 (Rigor, Wallace and Colony (2002): IABP buoy and pressure analyses of Arctic ice motion and its response to atmospheric circulation.)
- paper: https://doi.org/10.1175/1520-0477(1996)077<0437:TNYRP>2.0.CO;2 (Kalnay et al. (1996): the NCEP/NCAR reanalysis providing the daily-mean sea-level pressure.)
- archive: https://iabp.apl.uw.edu/Data_Products/C/ (IABP product C, interpolated Arctic buoy positions, 2000-2024 (no licence stated; not redistributed; retrieved 2026-10-11).)
- archive: https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/ (NCEP/NCAR Reanalysis 1 daily-mean sea-level pressure, 2000-2024 (public domain).)

## Limitations

- The geostrophic wind is computed from 2.5-degree reanalysis pressure by a local plane fit and given for the forecast day itself (a perfect-prognosis forcing); surface winds, ice concentration, thickness and ocean currents are not provided.
- Velocities are 24-hour displacements of interpolated buoy positions, so position noise and the interpolation of the product C add error; buoys include ones in the marginal ice zone and in summer, where the free-drift relation is weaker than in the central Arctic.
- The number of buoys grows strongly after 2014, so the test years weight recent conditions and regions with dense deployments.
- The IABP product states no licence; it is downloaded by the preparation script and not redistributed.
- A better score is a better statistical model of ice drift on these buoys, not a scientific discovery; the held-out tracks are public, so search feedback can overfit them.
- The candidate runs in a sandboxed child process that never receives positions after the start of a forecast day; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
