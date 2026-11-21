# Earth science: monthly Mauna Loa CO2 forecasts

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-mauna-loa-co2-forecast`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Fit a forecast model of atmospheric CO2 on the Mauna Loa monthly record up to 1999, together with an ENSO index, and for every month from 2000 to 2024 forecast the next 12 monthly means using only data up to that month. The score is the mean root-mean-square error over the 12 lead times.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fit_co2_model(training) -> model; model.forecast(history) -> float array (12,) of monthly mean atmospheric CO2 at Mauna Loa (ppm, NOAA GML/SIO record) for each of the 12 months after the last month in history. training and history are dicts of numpy arrays over consecutive months from 1958-03: year, month, co2_ppm, measured_days (number of measurement days; -1 for interpolated months and for the 1958-1974 Scripps values) and relative_nino34 (ERSSTv5 Nino 3.4 anomaly minus the 20S-20N mean SST anomaly, from 1951-1980 monthly climatologies). training ends in 1999-12. Forecasts are then requested in order for every origin month from 2000-01 to 2024-12; history holds all months up to and including the origin month and nothing later. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 120 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: mean over leads 1-12 of the root-mean-square error (ppm) over the 300 origins, using only measured (not interpolated) verifying months; lower is better. Wrong shapes, NaN or values outside [250, 600] ppm are invalid

## State of the art

No published value exists for this protocol. The evaluator's reference, seasonal persistence plus the last 12-month increase, scores 0.708 ppm; the seed, a 10-year quadratic trend with two annual harmonics refitted at every origin, 0.699 ppm.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [El Nino and a record CO2 rise](https://doi.org/10.1038/nclimate3063)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Forecast the near-term rise of atmospheric CO2, whose year-to-year variations reflect the response of land carbon sinks to climate variability such as ENSO.

Why it matters: Year-ahead CO2 forecasts test understanding of the carbon-cycle response to climate variability and give an early check on whether emissions and sinks behave as expected.

- [El Nino and a record CO2 rise](https://doi.org/10.1038/nclimate3063): Main research objective: Forecast the annual rise of Mauna Loa CO2 from emissions and the ENSO forecast. Task connection: The task scores monthly Mauna Loa CO2 forecasts at 1-12 months from the record and an ENSO index on held-out years.
- [A successful prediction of the record CO2 rise associated with the 2015/2016 El Nino](https://doi.org/10.1098/rstb.2017.0301): Main research objective: Verify the operational forecast of the 2016 CO2 rise. Task connection: Same forecast target and verification against the Mauna Loa record.

Both cited papers have forecasting the Mauna Loa CO2 rise as their main goal, the problem the local task scores; no reproduction of their numbers is claimed.

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
- `file:data/earth/co2/mlo_monthly_1958_2025_relative_nino34.json.gz`

## Sources

- paper: https://doi.org/10.1038/nclimate3063 (Betts et al. (2016): forecasting the annual rise of Mauna Loa CO2 from emissions and ENSO (the record 2015/16 rise).)
- paper: https://doi.org/10.1098/rstb.2017.0301 (Betts et al. (2018): verification of that operational CO2-rise forecast.)
- archive: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt (NOAA GML monthly mean CO2 at Mauna Loa (Scripps data before May 1974), March 1958 to December 2025; retrieved 2026-10-10; freely available under the GML data-use statement.)
- archive: https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc (NOAA ERSSTv5 monthly SST used for the relative Nino 3.4 input (no constraints on use).)

## Limitations

- Only the CO2 record and an ENSO index are given; operational forecasts also use fossil-fuel emission estimates, so scores are not comparable with them.
- Monthly means carry weather-driven noise of a few tenths of a ppm, and December 2022 - July 2023 values come from Maunakea after the Mauna Loa eruption; differences of a few hundredths of a ppm between candidates are not meaningful.
- The 300 origins overlap strongly, so the effective sample is about 25 annual growth anomalies.
- The GML file is updated monthly and recent values can be revised; if they change, the derived-file hash check fails and the data must be re-recorded.
- A better score is a better statistical forecast on these years, not a scientific discovery; the record is public, so search feedback can overfit it.
- The candidate runs in a sandboxed child process that never receives months after a forecast origin; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
