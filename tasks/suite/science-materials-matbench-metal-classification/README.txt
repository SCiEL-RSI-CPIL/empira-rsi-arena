# Matbench: metal or non-metal from composition

**Source collection:** [Molecular and materials design research: source-based research tasks](../../papers/science-chemistry_research.md)\
**Task id:** `science-materials-matbench-metal-classification`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Classify 4921 inorganic compounds as metals or non-metals from their composition alone, using experimental band-gap reports. The evaluator scores the balanced accuracy (Matbench's ROC AUC for label predictions) over the five official cross-validation folds.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fit_predict(train_inputs, train_targets, test_inputs, info) -> list of len(test_inputs) labels, each 0 or 1 (1 = metal). Inputs are compositions {'formula': str, 'composition': {element: amount}}; training targets are 0/1. Called once per official Matbench v0.1 fold (five calls), each in a fresh process that receives that fold's training inputs with labels and its held-out inputs without labels; info = {'task', 'input_type', 'target', 'unit', 'fold', 'n_train', 'n_test'}. Runs in a separate process that receives only these arguments (no test values), with at most 4 threads (OMP/MKL/OpenBLAS set to 4), no network, no subprocesses and no read access to the repository or prepared data; numpy, scipy, scikit-learn and RDKit are importable (pandas and torch only where the environment provides them; they are not requirements of the task); wall-clock limit 170 s per fold. Score: mean over the five folds of the balanced accuracy of the labels, the "mean balanced accuracy" column of the Matbench leaderboard (for 0/1 labels it equals the binary ROC AUC) (maximize); accuracy and F1 are reported

## State of the art

Best mean balanced accuracy on the Matbench v0.1 leaderboard for matbench_expt_is_metal: 0.9598 (Darwin, hard labels), read on 2026-11-24. This is the leaderboard column this task scores. The leaderboard lists probability ROC AUC separately, only for entries that saved probabilities (best 0.9850, XStack, whose balanced accuracy is 0.9433); it is not comparable with this task's score.

Reference value: best known value `0.9598` (reviewed 2026-11-24).

- [Matbench leaderboard: matbench_expt_is_metal](https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_expt_is_metal/)

Parameters: `{"task": "expt_is_metal"}`

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `public_benchmark`, verification `sampled`, reference `leaderboard`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `materials_science`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Decide from composition alone whether an inorganic compound is a metal or a non-metal.

Why it matters: Metallicity is the first screen for electronic, photovoltaic and thermoelectric applications, and composition-only classifiers screen compounds whose structures are unknown.

- [Predicting the Band Gaps of Inorganic Solids by Machine Learning](https://doi.org/10.1021/acs.jpclett.8b00124): Main research objective: Classify inorganic solids as metals or non-metals and predict band gaps from composition with machine learning. Task connection: The task uses the metal/non-metal labels collected there and scores composition-based classifiers.

The cited papers' main objective is the underlying problem the task scores; the local data, splits and limits are stated in the task's limitations.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Official Matbench v0.1 data and folds. Reference check: rescoring the stored label predictions of AMMExpress v2020 and RF-SCM/Magpie reproduces their leaderboard ROC AUC, balanced accuracy and F1 per fold (difference below 1e-15). Probability outputs are not accepted: the score is the balanced accuracy of hard labels, which the leaderboard reports for every entry, while its probability ROC AUC exists only for the entries that saved probabilities.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:rdkit`
- `py:sklearn`
- `file:data/chemistry/matbench/expt_is_metal.json`

## Sources

- paper: https://doi.org/10.1021/acs.jpclett.8b00124 (Zhuo, Mansouri Tehrani and Brgoch (2018): machine-learning classification of metals and prediction of band gaps from composition; source of the labels.)
- benchmark: https://doi.org/10.1038/s41524-020-00406-3 (Dunn et al. (2020): Matbench v0.1 tasks, nested cross-validation protocol (StratifiedKFold with seed 18012019 for classification) and metrics (MAE for regression, ROC AUC for classification).)
- repository: https://github.com/materialsproject/matbench/tree/936176db18ca4cd7b38cbd957c017a5bac770c6b (matbench_v0.1_validation.json (official folds) and stored per-fold predictions of published submissions (pinned commit).)
- benchmark: https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_expt_is_metal/ (Per-task Matbench v0.1 leaderboard (mean balanced accuracy over the five folds for every entry; probability ROC AUC for the entries that saved probabilities).)

## Limitations

- All labels are public in the Matbench file and every sample is a test sample of one fold; candidates or settings tuned on these folds overfit, so treat scores as development scores.
- Leaderboard entries were trained without this task's limits (170 s and at most 4 CPU threads per fold), so leaderboard values are reference points.
- Only 0/1 labels are accepted, so the score is the balanced accuracy; ROC AUC from probabilities, which Matbench also accepts, would not be comparable with the leaderboard entries that submitted labels. Labels are deduplicated by composition, so polymorphs share one label.
- Candidates run in a separate process that never receives held-out values, but the process isolation is an audit-hook policy for honest program search, not a security boundary (ctypes stays available for scientific libraries); run adversarial evaluations in a container.
