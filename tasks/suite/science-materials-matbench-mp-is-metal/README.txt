# Matbench v0.1 mp_is_metal

**Task id:** `science-materials-matbench-mp-is-metal`\
**Status:** requires_setup\
**Direction:** maximize\
**Family:** `matbench` (suite/_lib/arnew)\
**Areas:** materials science, physics, machine learning

## Task

Predict whether a crystal is a metal (1) or not (0) according to its band gap calculated by density functional theory from its crystal structure (Matbench v0.1 task mp_is_metal, 106113 samples). The evaluation uses the five official folds: for every fold the candidate is called once, in a separate process, with the other four folds and their target values as training data, and must predict the held-out fold within 600 seconds.

## Candidate interface

fit_predict(train_inputs, train_targets, test_inputs, info) -> list of len(test_inputs) numbers, each exactly 0 or 1 (class labels; any other value is invalid). Every input is a dict with the keys lattice (the three lattice vectors in angstrom, a 3 x 3 list), species (the element symbol of every site) and frac_coords (the fractional coordinates of every site). info has task, input_type, target, unit, fold, n_train and n_test. Score: mean over the five folds of the held-out ROC AUC of the 0/1 labels (which equals the balanced accuracy), maximized.

## The instance

Classifies 106,113 Materials Project crystals as metals (1) or non-metals (0) from their structure, scored by ROC AUC on 0/1 labels over the five official folds.

## Why it matters

**Problem:** Classify an inorganic crystal as a metal or a non-metal (DFT) from its structure.

Whether a material conducts is the first screen for electronic, photovoltaic and thermoelectric uses, and structure-based classifiers screen computed candidates cheaply.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://doi.org/10.1103/PhysRevLett.120.145301): Learn crystal properties directly from the atomic connections of a structure, with a graph neural network trained on Materials Project DFT data (eight properties). The paper includes a metal-versus-semiconductor classification of Materials Project crystals; the task is the same classification on the Matbench subset with free choice of model.

## Record

Leaderboard best 0.952 balanced accuracy (CGCNN v2019), snapshot of matbench.materialsproject.org read 2026-11-03, date unknown.

**Record validation:** matches a pinned source. 0.952 appears as the best balanced accuracy of matbench_mp_is_metal on the pinned leaderboard page. Note: The snapshot has no date on the page; the task scores the ROC AUC of 0/1 labels, which equals the balanced accuracy; the page's other classification column, probability ROC AUC, is not the record (re-checked on the live page 2026-11-18).

## Data and requirements

matbench_mp_is_metal.json.gz from ml.materialsproject.org, folds from the official validation file

Requires: `py:numpy`, `py:sklearn`, `file:data/matbench/mp_is_metal.json`. Run `python3 scripts/prepare_new_data.py --family matbench` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Leaders are graph neural networks and tuned ensembles; the leaderboard snapshot has no date.
- The classification score is the ROC AUC of the 0/1 labels, which equals the balanced accuracy: the record is the leaderboard's best balanced accuracy (the page has a second column, probability ROC AUC, that this task does not use).
- This task has 106k-133k structures: featurising them is slow, so a candidate must be efficient; the seed uses a cheap feature set.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
