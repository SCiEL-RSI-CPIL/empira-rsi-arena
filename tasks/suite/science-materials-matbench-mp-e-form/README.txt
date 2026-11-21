# Matbench v0.1 mp_e_form

**Task id:** `science-materials-matbench-mp-e-form`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `matbench` (suite/_lib/arnew)\
**Areas:** materials science, physics, machine learning

## Task

Predict the formation energy of a crystal in eV per atom, as calculated by density functional theory from its crystal structure (Matbench v0.1 task mp_e_form, 132752 samples). The evaluation uses the five official folds: for every fold the candidate is called once, in a separate process, with the other four folds and their target values as training data, and must predict the held-out fold within 600 seconds.

## Candidate interface

fit_predict(train_inputs, train_targets, test_inputs, info) -> list of len(test_inputs) numbers. Every input is a dict with the keys lattice (the three lattice vectors in angstrom, a 3 x 3 list), species (the element symbol of every site) and frac_coords (the fractional coordinates of every site). info has task, input_type, target, unit, fold, n_train and n_test. Score: mean over the five folds of the held-out mean absolute error, minimized.

## The instance

Predicts the DFT formation energy (eV/atom) of 132,752 Materials Project crystals from their structure, scored by mean absolute error over the five official folds.

## Why it matters

**Problem:** Predict the DFT formation energy per atom (eV/atom) of an inorganic crystal from its structure.

Formation energy measures thermodynamic stability and is the first filter for synthesisable compounds, while DFT calculation of every candidate is costly, so a learned predictor widens the space that can be screened.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://doi.org/10.1103/PhysRevLett.120.145301): Learn crystal properties directly from the atomic connections of a structure, with a graph neural network trained on Materials Project DFT data (eight properties). The paper's targets include the DFT formation energy of Materials Project crystals; the task predicts it on the Matbench subset with free choice of model.

## Record

Leaderboard best 0.017 MAE (coGN), snapshot of matbench.materialsproject.org read 2026-11-03, date unknown.

**Record validation:** matches a pinned source. 0.017 appears within the matbench_mp_e_form section of the pinned leaderboard page. Note: The snapshot has no date on the page.

## Data and requirements

matbench_mp_e_form.json.gz from ml.materialsproject.org, folds from the official validation file

Requires: `py:numpy`, `py:sklearn`, `file:data/matbench/mp_e_form.json`. Run `python3 scripts/prepare_new_data.py --family matbench` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Leaders are graph neural networks and tuned ensembles; the leaderboard snapshot has no date.
- Regression MAE averaged over the five official folds.
- This task has 106k-133k structures: featurising them is slow, so a candidate must be efficient; the seed uses a cheap feature set.
- One evaluation of the starting program took 190 s in the run of 2026-11-11, more than the arena's default limit of 120 s: raise AR_EVAL_TIMEOUT for this task.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
