# Physics: predicting newly measured nuclear masses from the AME2016 evaluation

**Source collection:** [Physics research: source-based research tasks](../../papers/science-physics_research.md)\
**Task id:** `science-physics-nuclear-mass-extrapolation`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Fit a nuclear mass model on the experimental masses of the 2016 Atomic Mass Evaluation and predict the mass excesses of the 71 nuclides first measured in time for AME2020. The candidate never sees the AME2020 values, and the score is the root-mean-square deviation from them in keV.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

predict_masses(training, queries) -> list of len(queries) atomic mass excesses in keV. training is a list of dicts with keys N, Z, A, element, mass_excess_keV and uncertainty_keV: the 2408 experimental (non-estimated) entries of the Atomic Mass Evaluation 2016 with N, Z >= 8. queries is a list of [N, Z] pairs: the 71 nuclides with N, Z >= 8 that are experimental in AME2020 but were missing or only estimated in AME2016. predict_masses runs in a sandboxed subprocess that never sees the AME2020 values, with a 120 s CPU budget. Score (minimize): root-mean-square deviation in keV between the predictions and the AME2020 mass excesses; predictions outside [-200000, 300000] keV, NaN or a wrong length are invalid

## State of the art

No published value for this exact split. For context, the AME2016 evaluators' own extrapolated estimates of 69 of the 71 targets deviate from the AME2020 measurements by 213 keV RMS (computed from the two published tables), and the liquid-drop seed gives 3121 keV.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [The AME2016 atomic mass evaluation (II)](https://doi.org/10.1088/1674-1137/41/3/030003)
- [The AME 2020 atomic mass evaluation (II)](https://doi.org/10.1088/1674-1137/abddaf)

Parameters: `{"cpu_seconds": 120}`

## Labels

Scientific contribution: `discovery` (tier 1); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `physics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict the masses of atomic nuclei that have not yet been measured.

Why it matters: Nuclear masses set reaction energies and drip lines and feed astrophysical r-process networks; global mass models are built and judged by how well they extrapolate to newly measured nuclides, and their residuals remain of order a few hundred keV.

- [Nuclear ground-state masses and deformations: FRDM(2012)](https://doi.org/10.1016/j.adt.2015.10.002): Main research objective: Build a global nuclear mass model that predicts ground-state masses, including for nuclei that have not been measured. Task connection: The task asks for exactly this kind of model and scores it on nuclides measured only after the training evaluation.
- [Surface diffuseness correction in global mass formula](https://doi.org/10.1016/j.physletb.2014.05.049): Main research objective: Improve a global nuclear mass formula and its extrapolation accuracy. Task connection: Represents the class of models the task measures, judged by RMS deviation in keV.

The cited papers' main research goal is predicting nuclear masses, and the evaluator scores exactly that on held-out measurements.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Training and target values come from the AMDC files of AME2016 and AME2020; the held-out set is defined only by measurement status (experimental in 2020, not experimental in 2016) and contains 71 nuclides from A = 30 to 266. The evaluator also reports the RMS of the AME2016 evaluators' own estimates ('#' values) for the 69 targets that had one; those estimates are published extrapolations, not measurements. The AME2020 values carry their own uncertainties (reported in the data), which are ignored by the unweighted RMS.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/bio_physics/nuclear/ame2016_to_ame2020.json.gz`

## Sources

- paper: https://doi.org/10.1016/j.adt.2015.10.002 (Moeller, Sierk, Ichikawa and Sagawa (2016), FRDM(2012): a global macroscopic-microscopic model whose purpose is predicting ground-state masses of known and unknown nuclei.)
- paper: https://doi.org/10.1103/PhysRevC.52.R23 (Duflo and Zuker (1995), Microscopic mass formulas: a compact global mass model fitted to measured masses and used to extrapolate.)
- paper: https://doi.org/10.1016/j.physletb.2014.05.049 (Wang, Liu, Wu and Meng (2014), Surface diffuseness correction in global mass formula (WS4): a mass model built for accurate extrapolation.)
- archive: https://doi.org/10.1088/1674-1137/41/3/030003 (AME2016 tables (Wang et al. 2017); experimental entries form the training data and the '#' estimates the context baseline. File: https://www-nds.iaea.org/amdc/ame2016/mass16.txt.)
- archive: https://doi.org/10.1088/1674-1137/abddaf (AME2020 tables (Wang et al. 2021, CC BY 3.0); the experimental entries absent from AME2016 are the held-out targets. File: https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt.)

## Limitations

- The held-out set is small (71 nuclides) and defined by which masses happened to be measured between the two evaluations, so it is not a uniform sample of the unknown region and the RMS is noisy.
- Both evaluations are public and so are their later successors, so a candidate tuned on this split can overfit it; any improvement should be checked on masses measured after AME2020 before a scientific claim.
- Training uses the AME2016 values as published; some of them were revised in AME2020, and the AME2020 targets carry their own uncertainties, which the unweighted RMS ignores.
- The AMDC files state no licence; they are downloaded by the prepare script and not redistributed.
