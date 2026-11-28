# Biology: probabilistic forecasts of state influenza hospital admissions

**Source collection:** [Biology research: source-based research tasks](../../papers/science-biology_research.md)\
**Task id:** `science-biology-influenza-hospitalization-forecasting`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Issue weekly quantile forecasts of laboratory-confirmed influenza hospital admissions one to four weeks ahead for every US state through the held-out 2024-2025 season, seeing only the surveillance history up to each forecast week. The evaluator scores the quantiles with the weighted interval score used by the CDC FluSight hub.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

forecast(history, reference_date, quantiles, horizons) -> dict mapping every location code of history to a 4 x 23 list of predictive quantiles (rows = horizons 1..4 weeks ahead, columns = the 23 FluSight quantile levels, non-negative and non-decreasing along each row). history maps location codes ('US' plus the two-digit FIPS codes of the 50 states, DC and Puerto Rico) to the weekly admission counts up to and including the week ending on reference_date (None where the hub reports no value). forecast is called once per reference date, in chronological order, inside a sandboxed subprocess that never sees later weeks, with a 180 s CPU budget for the whole season. Score (minimize): mean weighted interval score (Bracher et al. 2021) over the 26 reference dates of the 2024-2025 season, the 52 state-level locations and the four horizons; the US aggregate is not scored

## State of the art

No published value for this exact protocol (final-snapshot data, 52 state-level locations, mean absolute WIS). For context, the flat random-walk seed scores 172.6 mean WIS on the 2024-2025 season here, with 81.6 at one week and 255.7 at four weeks ahead.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Evaluation of FluSight influenza forecasting in the 2021-22 and 2022-23 seasons](https://doi.org/10.1038/s41467-024-50601-9)

Parameters: `{"season": "2024-2025", "cpu_seconds": 180}`

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `biology`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Forecast how many people will be admitted to hospital with influenza in each US state over the coming weeks.

Why it matters: Short-term influenza hospitalization forecasts inform staffing, vaccine messaging and public-health situational awareness; CDC runs an annual collaborative challenge on exactly this target and evaluates it with the weighted interval score.

- [Evaluation of FluSight influenza forecasting in the 2021-22 and 2022-23 seasons with a new target laboratory-confirmed influenza hospitalizations](https://doi.org/10.1038/s41467-024-50601-9): Main research objective: Assess how well collaborative models forecast weekly laboratory-confirmed influenza hospital admissions in the United States. Task connection: The task scores the same target, horizons, locations and metric on a later held-out season using the hub's own target data.
- [A collaborative multiyear, multimodel assessment of seasonal influenza forecasting in the United States](https://doi.org/10.1073/pnas.1812594116): Main research objective: Measure the accuracy of seasonal influenza forecasts across many models and years. Task connection: Establishes the forecasting problem and the evaluation practice the task follows.

Both cited papers' main research goal is influenza forecasting accuracy, the quantity the evaluator measures.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Counts come from the FluSight hub's target file (CDC NHSN weekly laboratory-confirmed influenza hospital admissions). Both training history and verification use the latest published snapshot rather than the data vintages available in real time, so revisions are not modelled and the task is easier than the live challenge; forecast dates are the Saturdays of the season's active window (November to April). Scores are mean WIS in admissions, so they are dominated by populous states and are not comparable with the hub's relative-WIS leaderboard.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/bio_physics/flu/flusight_hospital_admissions.json.gz`

## Sources

- paper: https://doi.org/10.1038/s41467-024-50601-9 (Mathis et al. (2024): evaluation of the FluSight collaborative forecasts of laboratory-confirmed influenza hospital admissions, the exact target, horizons and WIS metric used here.)
- paper: https://doi.org/10.1073/pnas.1812594116 (Reich et al. (2019): multiyear, multimodel assessment of seasonal influenza forecasting in the United States, establishing the forecasting problem and its evaluation.)
- paper: https://doi.org/10.1371/journal.pcbi.1008618 (Bracher, Ray, Gneiting and Reich (2021): definition of the weighted interval score for quantile-format epidemic forecasts, implemented independently in the evaluator.)
- benchmark: https://github.com/cdcepi/FluSight-forecast-hub (FluSight forecast hub (MIT licence): target-data/target-hospital-admissions.csv provides the weekly state-level admission counts from CDC NHSN used for training and verification.)

## Limitations

- Training history and verification both use the latest published snapshot, not the data vintages available in real time, so reporting revisions and backfill are not modelled and the task is easier than the live challenge.
- Only the hospital-admission target, four horizons and the November-April window are scored; the hub's rate-change and seasonal targets are not.
- The score is the mean weighted interval score in admissions, so it is dominated by populous states and is not comparable with the hub's relative-WIS leaderboard.
- The held-out season is public, so search feedback on this fixed split can overfit it; a candidate should be frozen and tested on a later season before any claim of forecasting skill.
- A better score is a better statistical forecaster on these seasons, not an epidemiological discovery.
