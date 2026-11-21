# Earth science: snow water equivalent from snow depth at unseen stations

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-snow-depth-to-swe`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Convert daily snow depth to snow water equivalent at 48 held-out SNOTEL stations across the western United States and Alaska, after learning from 114 other stations with both measurements over water years 2006-2025. The score is the root-mean-square error of snow water equivalent on the held-out stations' snow days.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

estimate_swe(train, test) -> dict {station_triplet: float array (n_targets,)} of snow water equivalent (mm, 0 to 10000) on the held-out target days of each test station. train and test map SNOTEL station triplets to dicts with state, lat, lon, elevation_m, precip_normal_mm (median water-year precipitation 2006-2015), day (days since 1970-01-01 of every day with a valid or edited snow-depth value, water years 2006-2025), depth_cm (snow depth on those days) and target_index (float indices into day of the scored days: depth >= 10.16 cm, SWE >= 10.16 mm, bulk density 0.05-0.65, both values flagged valid); train stations (114) also hold swe_mm (snow-pillow SWE on every day, NaN when not valid), test stations (48, 30 percent of each state) do not. Depth on all days, including later ones, may be used. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 180 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: root-mean-square error (mm) over the 103150 held-out target days; lower is better. Missing stations, wrong shapes, NaN or values outside [0, 10000] mm are invalid

## State of the art

No published value exists for this split. Internal references fitted on the training stations: Sturm-form density model 71.8 mm, constant median density 112.9 mm; the seed (bulk density linear in season day, log depth and elevation) scores 71.0 mm.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Converting snow depth to snow water equivalent using climatological variables](https://doi.org/10.5194/tc-13-1767-2019)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Estimate snow water equivalent, the water stored in the snowpack, from the much more widely measured snow depth.

Why it matters: Snowmelt supplies most of the water in the western United States and many mountain regions; depth can be measured cheaply (probes, ultrasonic sensors, lidar) while SWE is what water management needs, so accurate depth-to-SWE conversion directly improves water-supply estimates.

- [Estimating Snow Water Equivalent Using Snow Depth Data and Climate Classes](https://doi.org/10.1175/2010JHM1202.1): Main research objective: Develop a method to estimate snow water equivalent from snow depth, day of year and snow climate class. Task connection: The task scores the same conversion on held-out SNOTEL stations; the reference uses the paper's functional form with locally fitted coefficients.
- [Converting snow depth to snow water equivalent using climatological variables](https://doi.org/10.5194/tc-13-1767-2019): Main research objective: Build and evaluate a regression model converting snow depth to SWE with climatological predictors. Task connection: Same problem and the same kind of independent-station evaluation, with station metadata instead of gridded climatologies.

Both cited papers have depth-to-SWE conversion as their main goal, the problem the local task scores; no reproduction of their numbers is claimed.

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
- `file:data/earth/snow/snotel_depth_swe_162_stations.json.gz`

## Sources

- paper: https://doi.org/10.1175/2010JHM1202.1 (Sturm et al. (2010): estimating snow water equivalent from snow depth with a density model in depth and day of year by snow class; the evaluator's reference uses this functional form with coefficients fitted on the training stations.)
- paper: https://doi.org/10.5194/tc-13-1767-2019 (Hill et al. (2019): converting snow depth to snow water equivalent using climatological variables, evaluated on independent stations.)
- archive: https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/ (USDA NRCS Air and Water Database REST API: SNOTEL daily SNWD, WTEQ and PREC for 162 stations, water years 2006-2025 (public domain; retrieved 2026-10-10).)

## Limitations

- Snow-pillow SWE and ultrasonic depth are point measurements with their own errors (pillow bridging, depth sensor noise, depth reported in whole inches); only values flagged valid are scored and implausible densities (outside 0.05-0.65) are excluded.
- The hold-out is spatial (unseen stations) but the training stations cover the same years and weather, so performance in other years or at non-SNOTEL sites (lower elevations, shallow snow, forest openings) may differ.
- Only station metadata and a precipitation normal are given as climate information; gridded climatologies used by published models are not, so scores are not comparable with them.
- The Sturm-form reference is fitted on the training stations (its maximum-density parameter ends at the upper bound 1.0, so the fitted curve is nearly linear); published coefficients were not used because the paper could not be retrieved here.
- SNOTEL data are revised by NRCS; if values change, the derived-file hash check fails and the data must be re-recorded.
- A better score is a better depth-to-SWE conversion on this network, not a scientific discovery; the data are public, so search feedback can overfit them.
- The candidate runs in a sandboxed child process that never receives the held-out stations' SWE; module-level code still runs once in the evaluator process when the shared template imports the candidate, and outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
