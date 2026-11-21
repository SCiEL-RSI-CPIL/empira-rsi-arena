# Matbench v0.1 log_kvrh

**Task id:** `science-materials-matbench-log-kvrh`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `matbench` (suite/_lib/arnew)\
**Areas:** materials science, physics, machine learning

## Task

Predict the base-10 logarithm of the bulk modulus of a crystal (Voigt-Reuss-Hill average in GPa, calculated by density functional theory) from its crystal structure (Matbench v0.1 task log_kvrh, 10987 samples). The evaluation uses the five official folds: for every fold the candidate is called once, in a separate process, with the other four folds and their target values as training data, and must predict the held-out fold within 300 seconds.

## Candidate interface

fit_predict(train_inputs, train_targets, test_inputs, info) -> list of len(test_inputs) numbers. Every input is a dict with the keys lattice (the three lattice vectors in angstrom, a 3 x 3 list), species (the element symbol of every site) and frac_coords (the fractional coordinates of every site). info has task, input_type, target, unit, fold, n_train and n_test. Score: mean over the five folds of the held-out mean absolute error, minimized.

## The instance

Predicts log10 of the DFT Voigt-Reuss-Hill bulk modulus (GPa) of 10,987 inorganic crystals from their structure, scored by mean absolute error over the five official folds.

## Why it matters

**Problem:** Predict the base-10 logarithm of the DFT Voigt-Reuss-Hill bulk modulus (GPa) of an inorganic crystal from its structure.

The bulk modulus measures resistance to compression and underlies the design of hard and stiff materials, while DFT elastic calculations are costly, so a learned predictor screens large structure sets.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [A Statistical Learning Framework for Materials Science: Application to Elastic Moduli of k-nary Inorganic Polycrystalline Compounds](https://doi.org/10.1038/srep34256): Predict bulk and shear moduli of inorganic compounds with statistical learning. The task predicts the log VRH bulk modulus, the target of this paper, with free choice of model.
- [Charting the complete elastic properties of inorganic crystalline compounds](https://doi.org/10.1038/sdata.2015.9): Compute the full elastic tensors of inorganic crystals by DFT to chart their mechanical properties. The scored bulk moduli come from this elastic database.
- [Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://doi.org/10.1103/PhysRevLett.120.145301): Learn crystal properties directly from the atomic connections of a structure, with a graph neural network trained on Materials Project DFT data (eight properties). Among the paper's properties are the bulk and shear moduli of Materials Project crystals; the task predicts the log bulk modulus with free choice of model.

## Record

Leaderboard best 0.0491 MAE (coNGN), snapshot of matbench.materialsproject.org read 2026-11-03, date unknown.

**Record validation:** matches a pinned source. 0.0491 appears within the matbench_log_kvrh section of the pinned leaderboard page. Note: The snapshot has no date on the page.

## Data and requirements

matbench_log_kvrh.json.gz from ml.materialsproject.org, folds from the official validation file

Requires: `py:numpy`, `py:sklearn`, `file:data/matbench/log_kvrh.json`. Run `python3 scripts/prepare_new_data.py --family matbench` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Leaders are graph neural networks and tuned ensembles; the leaderboard snapshot has no date.
- Regression MAE averaged over the five official folds.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
