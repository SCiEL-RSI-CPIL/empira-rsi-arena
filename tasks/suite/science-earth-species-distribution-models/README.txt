# Earth science: species distribution models from presence-only records

**Source collection:** [Earth and environmental science research: source-based research tasks](../../papers/science-earth_research.md)\
**Task id:** `science-earth-species-distribution-models`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Fit species distribution models for 226 anonymised species in six regions from presence-only records, background points and environmental covariates, and score sites of independent presence-absence surveys whose outcomes are withheld. The score is the mean area under the ROC curve over species.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

predict_distributions(regions) -> dict {region: {species_id: float array (n_sites,)}} of habitat-suitability scores (any finite real numbers; only their ranking matters) at the presence-absence survey sites of each species' biological group. regions maps AWT, CAN, NSW, NZ, SA and SWI to dicts with covariates (list of k names), categorical (names of categorical covariates: CAN ontveg, NSW vegsys, NZ age and toxicats, SWI calc), coordinates (text naming the x/y system), presence {species_id: {'x': (n, k) covariate values, 'xy': (n, 2) coordinates}} of presence-only records, background {'x': (m, k), 'xy': (m, 2)} of random background points, and sites {group: {'x': (s, k), 'xy': (s, 2), 'species': [ids of the species evaluated at these sites]}}; NaN marks a missing covariate value. Presence-absence outcomes at the sites are never provided. The candidate runs in a separate sandboxed process (numpy, scipy, scikit-learn and pandas importable; no file, network or subprocess access; one thread) with 300 s of CPU time for fitting and all predictions together; seed any randomness so that results are deterministic. Score: mean over the 226 species of the area under the ROC curve (tied scores count one half) of the site scores against the observed presences and absences; higher is better. Missing species, wrong shapes or non-finite scores are invalid

## State of the art

No published value is embedded (see limitations). The seed, a per-covariate Gaussian presence/background density ratio without categorical covariates, scores a mean AUC of 0.709 (region means 0.48 for CAN to 0.78 for SA).

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Predictive performance of presence-only species distribution models: a benchmark study with reproducible code](https://doi.org/10.1002/ecm.1486)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `earth_and_environment`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict where a species occurs from presence-only occurrence records and environmental covariates (species distribution modelling).

Why it matters: Species distribution models are the main tool for mapping biodiversity, planning conservation and projecting climate-change impacts on species; presence-only records are the commonest data, and how well methods predict independent surveys is a central methodological question in ecology.

- [Novel methods improve prediction of species' distributions from occurrence data](https://doi.org/10.1111/j.2006.0906-7590.04596.x): Main research objective: Compare the predictive performance of species distribution modelling methods fitted to presence-only data, using independent presence-absence data from six regions. Task connection: The task uses the released version of the same benchmark (226 species, six regions) and the same kind of evaluation (AUC on independent presence-absence sites).
- [Predictive performance of presence-only species distribution models: a benchmark study with reproducible code](https://doi.org/10.1002/ecm.1486): Main research objective: Benchmark presence-only species distribution modelling methods on the NCEAS data with reproducible code. Task connection: Same data and AUC evaluation; the task lets any method be evolved and allows coordinates as predictors.

Both cited papers have predictive species distribution modelling from presence-only data as their main goal, the problem the local task scores; no reproduction of their numbers is claimed.

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
- `file:data/earth/sdm/nceas_226_species.json.gz`

## Sources

- paper: https://doi.org/10.1111/j.2006.0906-7590.04596.x (Elith et al. (2006): comparison of species distribution modelling methods fitted to presence-only data and evaluated with independent presence-absence data in the same six regions.)
- paper: https://doi.org/10.17161/bi.v15i2.13384 (Elith et al. (2020): description of the released benchmark data (226 species, presence-only training, background, presence-absence evaluation).)
- paper: https://doi.org/10.1002/ecm.1486 (Valavi et al. (2022): benchmark of presence-only methods on these data with AUC as the main evaluation measure.)
- archive: https://osf.io/kwc4v/ (OSF project SDM_NCEAS (CC BY 4.0): train_po, train_bg, test_env and test_pa CSV files of the six regions; downloaded files match the SHA-256 hashes published by OSF.)

## Limitations

- AUC measures how well site scores rank surveyed presences above absences; it ignores calibration and prevalence, and survey absences include non-detections.
- Site and record coordinates are provided, so spatial predictors are allowed; published comparisons mostly used environmental covariates only, so scores are not directly comparable with them.
- No published value is embedded: the benchmark paper's tables could not be retrieved in this environment (the publisher's site refused the request), so the mean AUCs it reports were not checked.
- The presence-absence outcomes are public on OSF; the sandbox keeps them away from the candidate process, but search feedback on these fixed sites can still overfit them.
- A better score is a better species distribution modelling method on this benchmark, not an ecological discovery.
- The candidate runs in a sandboxed child process that never receives the survey outcomes, and no statement of the candidate is executed in the evaluator's process; outside macOS (kernel sandbox) deliberately hostile native code could bypass the file policy.
