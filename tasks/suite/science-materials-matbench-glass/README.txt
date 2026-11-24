# Matbench v0.1 glass

**Task id:** `science-materials-matbench-glass`\
**Status:** requires_setup\
**Direction:** maximize\
**Family:** `matbench` (suite/_lib/arnew)\
**Areas:** materials science, physics, machine learning

## Task

Predict whether an alloy forms a metallic glass (1) or not (0) from its composition (Matbench v0.1 task glass, 5680 samples). The evaluation uses the five official folds: for every fold the candidate is called once, in a separate process, with the other four folds and their target values as training data, and must predict the held-out fold within 200 seconds.

## Candidate interface

fit_predict(train_inputs, train_targets, test_inputs, info) -> list of len(test_inputs) numbers, each exactly 0 or 1 (class labels; any other value is invalid). Every input is a dict with the keys formula (a string) and composition (a dict from element symbol to amount). info has task, input_type, target, unit, fold, n_train and n_test. Score: mean over the five folds of the held-out ROC AUC of the 0/1 labels (which equals the balanced accuracy), maximized.

## The instance

Classifies 5,680 alloy compositions as glass formers (1) or not (0) from the composition alone, scored by ROC AUC on 0/1 labels over the five official folds.

## Why it matters

**Problem:** Classify metallic alloy compositions as forming a metallic glass or not (glass-forming ability) from the composition alone.

Metallic glasses combine high strength, elasticity and corrosion resistance, but finding compositions that form glass by experiment is slow and expensive, so predicting glass-forming ability from composition guides alloy discovery.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Machine Learning Approach for Prediction and Understanding of Glass-Forming Ability](https://doi.org/10.1021/acs.jpclett.7b01046): Predict the glass-forming ability of alloys with machine learning and extract the features that explain it. The task predicts glass-forming ability from composition; the paper uses its own data and features, whereas the task uses the Matbench data set and folds.
- [Accelerated discovery of metallic glasses through iteration of machine learning and high-throughput experiments](https://doi.org/10.1126/sciadv.aaq1566): Discover new metallic glass systems by iterating machine-learning prediction of glass-forming ability with high-throughput experiments. The task is the machine-learning prediction half of this loop, on a fixed data set and without new experiments.

## Record

Leaderboard best 0.9603 balanced accuracy (MODNet v0.1.12), snapshot of matbench.materialsproject.org read 2026-11-03, date unknown.

**Record validation:** matches a pinned source. 0.9603 appears as the best balanced accuracy of matbench_glass on the pinned leaderboard page. Note: The snapshot has no date on the page; the task scores the ROC AUC of 0/1 labels, which equals the balanced accuracy; the page's other classification column, probability ROC AUC, is not the record (re-checked on the live page 2026-11-18).

## Data and requirements

matbench_glass.json.gz from ml.materialsproject.org, folds from the official validation file

Requires: `py:numpy`, `py:sklearn`, `file:data/arnew/matbench/glass.json`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family matbench` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Leaders are graph neural networks and tuned ensembles; the leaderboard snapshot has no date.
- The classification score is the ROC AUC of the 0/1 labels, which equals the balanced accuracy: the record is the leaderboard's best balanced accuracy (the page has a second column, probability ROC AUC, that this task does not use).
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
