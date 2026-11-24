# Earth science: one-day earthquake forecasts for southern California

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-socal-earthquake-forecast`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Fit an earthquake occurrence model on the 1981-2015 southern California ComCat catalogue and, for every day of 2016-2024, forecast the expected number of magnitude 3.45 or larger earthquakes in each 0.1-degree cell using only events before that day. The score is the information gain per earthquake of the Poisson likelihood over a time-independent smoothed-seismicity reference.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fit_quake_model(training, grid) -> model; model.forecast(history, day) -> float array (50, 70) of the expected number of target earthquakes (magnitude >= 3.45, depth <= 30 km) during [day, day + 1) in each 0.1-degree cell: row i is latitude 32.0 + 0.1 i to 32.1 + 0.1 i deg N, column j is longitude -121.0 + 0.1 j to -120.9 + 0.1 j deg E (grid['lat_edges'] (51,), grid['lon_edges'] (71,)). training and history are dicts of numpy arrays t (days since 1981-01-01 00:00 UTC), lat, lon (deg), depth (km) and mag of all ComCat earthquakes with magnitude >= 2.5 in 31-38 N, 122-113 W, sorted by time; training holds the events before 2016-01-01. grid also gives target_min_magnitude 3.45, target_max_depth_km 30, first_forecast_day 12783 (2016-01-01) and last_forecast_day 16070 (2024-12-31). Forecasts are requested in order for every day from 2016-01-01 to 2024-12-31 (day is the float day number); history then holds every event before 00:00 UTC of that day and nothing later. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 300 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: information gain per target earthquake (nats) of the joint Poisson log-likelihood of the observed daily cell counts (1047 target events) over the evaluator's time-independent reference (1981-2015 epicentres smoothed with a 0.15-degree Gaussian kernel and scaled to the 1981-2015 target rate); higher is better. Rates below 1e-10, NaN, wrong shapes or daily totals above 1e4 are invalid

## State of the art

No published value exists for this exact protocol (region, period, M >= 3.45 targets and reference). For context, the seed (time-independent smoothing with a 0.1-degree kernel) gains 0.12 nats per earthquake over the reference and an untuned Omori-Utsu clustering model about 1.4 nats; Helmstetter et al. (2006) found short-term clustering models far more informative than time-independent ones for southern California daily forecasts.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Comparison of Short-Term and Time-Independent Earthquake Forecast Models for Southern California](https://doi.org/10.1785/0120050067)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Forecast where and when earthquakes will occur in the next day from the past earthquake catalogue (short-term earthquake forecasting).

Why it matters: Short-term, clustering-based earthquake forecasts underlie operational aftershock forecasting and time-dependent seismic hazard; likelihood testing of such forecasts in CSEP is how forecasting skill is established.

- [Comparison of Short-Term and Time-Independent Earthquake Forecast Models for Southern California](https://doi.org/10.1785/0120050067): Main research objective: Build and compare one-day and time-independent earthquake forecast models for southern California on a 0.1-degree grid using likelihood scores. Task connection: The task scores the same kind of daily gridded forecast by the joint Poisson log-likelihood; it uses a rectangular region, M >= 3.45 targets and the 2016-2024 ComCat catalogue instead of the paper's setting.
- [Statistical Models for Earthquake Occurrences and Residual Analysis for Point Processes](https://doi.org/10.1080/01621459.1988.10478560): Main research objective: Model earthquake occurrence as a self-exciting point process (ETAS) and assess the fit to catalogues. Task connection: ETAS-type clustering models are the natural candidates; the task scores their forecasts on held-out days.

The cited papers' main goal is statistical earthquake forecasting and occurrence modelling, the problem the local task scores; no reproduction of their numbers is claimed.

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
- `file:data/earth/quakes/comcat_socal_m2.5_1981_2024.json.gz`

## Sources

- paper: https://doi.org/10.1785/0120050067 (Helmstetter, Kagan and Jackson (2006): one-day forecasts of southern California earthquakes on a 0.1-degree grid, comparing short-term (clustering) and time-independent models by likelihood.)
- paper: https://doi.org/10.1080/01621459.1988.10478560 (Ogata (1988): the epidemic-type aftershock sequence (ETAS) point-process model of earthquake occurrence.)
- paper: https://doi.org/10.1785/0120090192 (Zechar, Gerstenberger and Rhoades (2010): Poisson joint-likelihood tests of gridded earthquake forecasts used by CSEP; the task uses the joint log-likelihood and reports the N-test ratio.)
- archive: https://earthquake.usgs.gov/fdsnws/event/1/ (USGS ANSS Comprehensive Catalog (ComCat, doi:10.5066/F7MS3QZH), FDSN event service: earthquakes with magnitude >= 2.5 in 31-38 N, 122-113 W, 1981-2024 (public domain; retrieved 2026-11-20).)

## Limitations

- The test region is a 0.1-degree rectangle (32-37 N, 121-114 W) rather than the CSEP southern California polygon; it includes offshore and northern Baja California cells where the catalogue is less complete.
- ComCat mixes magnitude types (mostly local and moment magnitudes) and is incomplete in the hours after large mainshocks; targets are the catalogue as distributed, not a homogenised or declustered catalogue.
- Forecasts are issued once per day at 00:00 UTC and verified on daily cell counts; magnitudes and depths of target events are not scored (no magnitude or depth test).
- The 2016-2024 period is dominated by the 2019 Ridgecrest sequence, so the score mostly reflects aftershock forecasting skill for one large sequence; the information gain is an estimate with substantial sampling uncertainty.
- ComCat is a live catalogue; if past events are revised, the derived-file hash check fails and the data must be re-recorded, which changes scores slightly.
- A better score is a better statistical forecast model on these years, not a scientific discovery; the held-out catalogue is public, so search feedback on this fixed split can overfit it.
- The candidate runs in a sandboxed child process that never receives events after a forecast day starts, and no statement of the candidate is executed in the evaluator's process; outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
