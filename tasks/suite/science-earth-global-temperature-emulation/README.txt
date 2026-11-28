# Earth science: predicting recent global warming from forcing and early observations

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-global-temperature-emulation`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Fit a model of annual global mean surface temperature on the 1850-1979 HadCRUT5 record, given radiative forcing histories and an ENSO index for 1750-2024, and predict the 1980-2024 anomalies without seeing any later temperature. The score is the root-mean-square error of the predicted anomalies.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

predict_gmst(training, inputs) -> float array (45,) of annual global mean near-surface temperature anomalies (deg C from 1961-1990, HadCRUT5 analysis) for 1980-2024. training holds year (1850-1979), gmst and its 95% range gmst_low, gmst_high. inputs holds forcing_year (1750-2024) and forcing, a dict of annual effective radiative forcing series (W m-2, Forster et al. 2025 update of IPCC AR6) for CO2, CH4, N2O, halogen, minor, O3, H2O_stratospheric, contrails, aerosol-radiation_interactions, aerosol-cloud_interactions, BC_on_snow, land_use, solar, volcanic, anthro and total; enso_year, enso_month and relative_nino34 (1854-01 to 2024-12: ERSSTv5 Nino 3.4 anomaly minus the 20S-20N mean SST anomaly, from 1951-1980 monthly climatologies); and predict_years (1980-2024). No temperature after 1979 is provided. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 60 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: root-mean-square error (deg C) of the 45 predicted anomalies against HadCRUT5; lower is better. Wrong shapes, NaN or values outside [-5, 5] deg C are invalid

## State of the art

No published value exists for this split. Internal references fitted on 1850-1979: linear regression on total forcing (the seed) 0.464 deg C, a two-layer energy-balance model with illustrative, uncalibrated parameters (lambda 1.3, gamma 0.7 W m-2 K-1; heat capacities 8 and 100 W yr m-2 K-1) plus a fitted offset 0.163 deg C, and a Lean-Rind-style regression on smoothed anthropogenic, volcanic and solar forcing and ENSO 0.750 deg C.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [How natural and anthropogenic influences alter global and regional surface temperatures: 1889 to 2006](https://doi.org/10.1029/2008GL034864)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Explain and predict observed global mean temperature change from radiative forcing and internal variability (attribution and transient climate response).

Why it matters: How much warming follows a given forcing (the transient climate response) is one of the most consequential numbers in climate science; simple models fitted to observed warming and forcing are the standard tools for attribution and for constraining it.

- [How natural and anthropogenic influences alter global and regional surface temperatures: 1889 to 2006](https://doi.org/10.1029/2008GL034864): Main research objective: Quantify the contributions of anthropogenic, volcanic, solar and ENSO influences to observed temperature changes. Task connection: The task scores the same kind of forcing-plus-ENSO model of observed global temperature, but asks it to predict a held-out later period.
- [A real-time Global Warming Index](https://doi.org/10.1038/s41598-017-14828-5): Main research objective: Attribute observed global warming to human and natural forcing in real time with a simple climate response model fitted to observations. Task connection: Candidates can be such response models; the task verifies them on years withheld from fitting.
- [Energy budget constraints on climate response](https://doi.org/10.1038/ngeo1836): Main research objective: Estimate transient and equilibrium climate response from observed warming, forcing and ocean heat uptake. Task connection: A model that predicts the held-out warming well must encode a realistic transient response.

The cited papers' main goal is explaining observed temperature change from forcing (attribution and climate response), the problem the local task scores out of sample; no reproduction of their numbers is claimed.

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
- `file:data/earth/gmst/hadcrut5_erf_enso_1850_2024.json.gz`

## Sources

- paper: https://doi.org/10.1029/2008GL034864 (Lean and Rind (2008): multiple regression of observed global and regional temperature on anthropogenic, volcanic, solar and ENSO influences.)
- paper: https://doi.org/10.1038/s41598-017-14828-5 (Haustein et al. (2017): a simple impulse-response climate model fitted to observed temperature and forcing to attribute warming (Global Warming Index).)
- paper: https://doi.org/10.1038/ngeo1836 (Otto et al. (2013): energy-budget estimates of transient and equilibrium climate response from observed warming and forcing.)
- paper: https://doi.org/10.1029/2019JD032361 (Morice et al. (2021): HadCRUT5, the global temperature analysis whose annual means and 95% ranges are the target.)
- paper: https://doi.org/10.5194/essd-17-2641-2025 (Forster et al. (2025): Indicators of Global Climate Change 2024, the effective radiative forcing time series 1750-2024 used as inputs.)
- archive: https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv (HadCRUT5.0.2.0 global annual summary series (Open Government Licence; retrieved 2026-10-10); 1850-2024 used.)
- repository: https://github.com/ClimateIndicator/forcing-timeseries/tree/852f4f7b0733fa4d573c145cdfaf556da5589231 (Best-estimate ERF aggregates 1750-2024 (output/ERF_best_aggregates_1750-2024.csv, MIT licence) at a pinned commit.)

## Limitations

- The forcing inputs are not independent of the target: the assessed aerosol forcing behind the AR6 and 2024 updates uses observational constraints that include the recent warming, so the held-out period leaks partly into the inputs.
- Recent global temperatures are widely known; a candidate can encode remembered values or trends instead of inferring them from 1850-1979, so the score measures emulation skill only for candidates restricted to the training data. Freeze a candidate and test it on later years before any claim.
- Only 45 annual values are scored and internal variability not explained by the ENSO index (roughly 0.1 deg C per year) limits the attainable error; HadCRUT5 itself has a 95% range of about 0.03-0.05 deg C in this period.
- A better score is a better statistical or physical emulator of observed warming given these forcings, not by itself a new estimate of climate sensitivity.
- HadCRUT5 is a live product; if past annual means are revised, the derived-file hash check fails and the data must be re-recorded.
- The candidate runs in a sandboxed child process that never receives temperatures after 1979, and no statement of the candidate is executed in the evaluator's process; outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
