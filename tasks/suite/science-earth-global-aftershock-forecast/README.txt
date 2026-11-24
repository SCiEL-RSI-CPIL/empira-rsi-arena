# Earth science: probabilistic aftershock forecasts after large earthquakes worldwide

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-global-aftershock-forecast`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Fit an aftershock forecasting model on 1990-2014 sequences from the global ComCat catalogue and, for every large shallow earthquake of 2015-2024, forecast the probability distribution of the number of magnitude 5 or larger aftershocks in the week after day 1 and the month after day 7, using only events before each forecast. The score is the mean logarithmic score of the observed counts.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fit_aftershock_model(history, training_cases, info) -> model; model.forecast(history, case) -> float array (17,) of probabilities (each >= 1e-9, summing to 1 within 1e-6) that the number of target earthquakes falls in the bins [0], [1], ..., [9], [10-14], [15-19], [20-29], [30-49], [50-99], [100-199], [200 or more] (info['bin_edges'] = 0, 1, ..., 10, 15, 20, 30, 50, 100, 200). history is a dict of numpy arrays t (days since 1980-01-01 00:00 UTC), lat, lon, depth (km) and mag of all ComCat earthquakes with magnitude >= 5 worldwide, sorted by time. A case is a dict of floats main_t, main_lat, main_lon, main_depth, main_mag (a shallow, depth <= 70 km, magnitude >= 6 mainshock that is not within the zone of an equal or larger event of the previous 365 days), start (main_t + 1 or main_t + 7 days), window_days (7 or 30) and radius_km = max(50, 2 * 10^(0.5 main_mag - 1.8)); the target is the number of magnitude >= 5 events at any depth within radius_km of the mainshock epicentre during [start, start + window_days). training_cases holds the same fields plus the observed count for the 4207 cases with start before 2015-01-01, and the fit-time history holds the events before 2015-01-01. Forecasts are then requested in order of start for the 1443 cases starting from 2015-01-01 to 2024-11-22; history then holds every event before the case's start and nothing later. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 240 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: mean natural-log probability assigned to the observed bins; higher is better. Wrong shapes, NaN, probabilities below 1e-9 or not summing to 1 are invalid

## State of the art

No published value exists for this protocol. The evaluator's reference, a Reasenberg-Jones negative-binomial model with one global parameter set fitted on the training cases (a = -1.9, p = 0.95, c = 0.05 d, b = 1, shape 0.2), scores -1.048; the seed, the same model with Poisson counts, scores -1.401.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Three Ingredients for Improved Global Aftershock Forecasts: Tectonic Region, Time-Dependent Catalog Incompleteness, and Intersequence Variability](https://doi.org/10.1785/0120160073)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Forecast how many aftershocks will follow a large earthquake in the coming days and weeks, as a probability distribution.

Why it matters: Aftershocks cause casualties and damage after large earthquakes; operational aftershock forecasts guide emergency response and recovery decisions, and their statistical skill is an active research question.

- [Earthquake Hazard After a Mainshock in California](https://doi.org/10.1126/science.243.4895.1173): Main research objective: Estimate the probability of aftershocks and larger earthquakes after a mainshock from aftershock statistics. Task connection: The task scores probabilistic aftershock-count forecasts of exactly this kind on held-out global sequences; the seed and reference use the paper's model form.
- [Three Ingredients for Improved Global Aftershock Forecasts: Tectonic Region, Time-Dependent Catalog Incompleteness, and Intersequence Variability](https://doi.org/10.1785/0120160073): Main research objective: Improve global aftershock forecasts with regional parameters, incompleteness modelling and inter-sequence variability. Task connection: Candidates may add the same ingredients; the task verifies forecasts against the 2015-2024 catalogue.

Both cited papers have aftershock forecasting as their main goal, the problem the local task scores; no reproduction of their numbers is claimed.

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
- `py:scipy`
- `file:data/earth/aftershocks/comcat_global_m5_1980_2024.json.gz`

## Sources

- paper: https://doi.org/10.1126/science.243.4895.1173 (Reasenberg and Jones (1989): aftershock probabilities from the Gutenberg-Richter and modified Omori laws (the model family of the seed and the reference).)
- paper: https://doi.org/10.1785/0120160073 (Page et al. (2016): global aftershock forecasts with regional parameters, early-catalogue incompleteness and negative-binomial inter-sequence variability.)
- archive: https://earthquake.usgs.gov/fdsnws/event/1/ (USGS ANSS Comprehensive Catalog (ComCat, doi:10.5066/F7MS3QZH): earthquakes of magnitude >= 5 worldwide, 1980-2024 (public domain; retrieved 2026-10-10).)

## Limitations

- Mainshock selection (magnitude >= 6, depth <= 70 km, not inside the zone of an equal or larger event of the previous year) and the aftershock radius max(50, 2 * 10^(0.5 M - 1.8)) km are simple local rules, not the regionalised zones of operational forecasts; events of other sequences inside the radius are counted.
- Global ComCat magnitudes mix types (mb, Ms, Mw) and the catalogue is incomplete in the first hours and days after large mainshocks, so counts of magnitude >= 5 events are those of the catalogue as distributed.
- Counts are scored in 17 bins with the logarithmic score; probabilities below 1e-9 are not allowed, so one confident miss costs at most about 21 nats.
- Forecasts of the two windows of one mainshock and of overlapping sequences are not independent; the 1443 test forecasts come from 723 mainshocks.
- ComCat is a live catalogue; if past events are revised, the derived-file hash check fails and the data must be re-recorded, which changes scores slightly.
- A better score is a better statistical aftershock forecasting model on these years, not a scientific discovery; the held-out catalogue is public, so search feedback on this fixed split can overfit it.
- The candidate runs in a sandboxed child process that never receives events after a forecast start, and no statement of the candidate is executed in the evaluator's process; outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
- Three training cases start in the last days of 2014 and their target windows end after 2015-01-01 (all three counts are zero), so the fit-time counts reflect up to 18 days of the test period; one of them overlaps by one day the 30-day test window of the same mainshock.
