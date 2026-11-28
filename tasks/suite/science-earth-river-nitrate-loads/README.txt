# Earth science: river nitrate loads from sparse samples

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-river-nitrate-loads`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Estimate daily nitrate concentrations in eight U.S. rivers from about 16 samples per year and continuous daily discharge, as in routine water-quality monitoring. The score is the error of the resulting monthly nitrate loads against loads computed from continuous in-situ nitrate sensors, which the candidate never sees.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

estimate_nitrate(sites) -> dict {site_number: float array (n_days,)} of daily mean nitrate plus nitrite concentrations (mg/L as N, 0 to 100) for every day of each site's record. sites maps eight USGS site numbers to dicts with day (days since 1970-01-01, ascending, gaps where discharge is missing), decimal_year, discharge_m3s (daily mean discharge, m3/s, every day), sample_index (float indices into day of the sampled days: one random sensor day per calendar month plus four random days per water year with discharge in the site's top 20 percent), sample_nitrate_mg_l (the sensor daily means on those days), lat, lon and drainage_area_km2. Sensor concentrations on other days are never provided. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 120 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: mean over site-months (calendar months with at least 25 approved sensor days, 959 in all) of the absolute relative error (percent) of the monthly load sum(C * Q) over the sensor days, using the predicted instead of the sensor concentrations; lower is better. Missing sites, wrong shapes, NaN or values outside [0, 100] mg/L are invalid

## State of the art

No published value exists for this protocol. The evaluator's WRTDS reference gives a mean absolute monthly load error of 24.8 percent (annual 10.4 percent, 70 site-years); the seed, a per-site log-linear regression on discharge, season and trend with smearing correction, gives 37.9 percent (annual 21.6 percent).

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Weighted Regressions on Time, Discharge, and Season (WRTDS), with an Application to Chesapeake Bay River Inputs](https://doi.org/10.1111/j.1752-1688.2010.00482.x)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Estimate river nutrient concentrations and loads continuously in time from sparse water-quality samples and daily discharge.

Why it matters: Nitrate loads from rivers drive coastal hypoxia (for example in the northern Gulf of Mexico) and drinking-water problems; nutrient-reduction policy is tracked with load estimates built from sparse samples, so estimation error directly affects management decisions.

- [Weighted Regressions on Time, Discharge, and Season (WRTDS), with an Application to Chesapeake Bay River Inputs](https://doi.org/10.1111/j.1752-1688.2010.00482.x): Main research objective: Develop a method to estimate river concentrations, fluxes and trends from sparse samples and daily discharge. Task connection: The task scores exactly this estimation from samples plus discharge, verified against continuous sensor data; WRTDS is the evaluator's reference.
- [An evaluation of methods for estimating decadal stream loads](https://doi.org/10.1016/j.jhydrol.2016.08.059): Main research objective: Compare the accuracy of stream-load estimation methods. Task connection: Same question (which estimator gives accurate loads), with sensor-based truth and monthly loads.

The cited papers' main goal is estimating stream concentrations and loads from sparse samples, the problem the local task scores; no reproduction of their numbers is claimed.

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
- `file:data/earth/nitrate/nwis_sensor_nitrate_8_sites.json.gz`

## Sources

- paper: https://doi.org/10.1111/j.1752-1688.2010.00482.x (Hirsch, Moyer and Archfield (2010): weighted regressions on time, discharge and season (WRTDS) for river concentrations and fluxes from sparse samples; the evaluator's reference implements its weighting (half-windows 7 years, 2 log units, half a year; at least 100 samples).)
- paper: https://doi.org/10.1016/j.jhydrol.2016.08.059 (Lee et al. (2016): evaluation of stream-load estimation methods against loads from densely sampled records.)
- paper: https://doi.org/10.1111/1752-1688.12386 (Pellerin et al. (2016): continuous in-situ nitrate sensor networks of the USGS, the source of the daily verification data.)
- archive: https://waterservices.usgs.gov/nwis/dv/ (USGS NWIS daily values of discharge (00060) and nitrate plus nitrite (99133) for eight sites, 2008 to water year 2025 (public domain; retrieved 2026-10-10); only approved, non-estimated sensor values are used as truth.)

## Limitations

- The samples are drawn from the sensor record by a fixed random design rather than being real laboratory analyses, so they carry sensor rather than laboratory error and no sampling-time bias; real monitoring programmes differ.
- Loads are computed only over days with approved sensor data in months with at least 25 such days, so winter gaps (sensors removed for ice) and storms missed by the sensors are not scored.
- Sites are large Midwest and lower-Mississippi rivers with strong agricultural nitrate signals; the score says little about small streams or other regions.
- NWIS daily values are live and provisional values are approved long after the measurements; each record therefore ends where its approved discharge ended when the lock was recorded on 2026-10-10 (2024-04-14 at Baton Rouge, 2025-09-30 at the other sites). If approved values are revised, the derived-file hash check fails and the data must be re-recorded.
- The WRTDS reference is a local implementation following the published method (no edge adjustment); it was checked to recover an exact log-linear model but was not compared with the EGRET package, whose geospatial dependencies were not installed.
- A better score is a better load-estimation method on these records, not a scientific discovery; the sensor data are public, so search feedback can overfit them.
- The candidate runs in a sandboxed child process that never receives the unsampled sensor values, and no statement of the candidate is executed in the evaluator's process; outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
