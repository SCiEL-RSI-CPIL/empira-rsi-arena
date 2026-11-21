# TDC ADMET Caco2_Wang

**Task id:** `tdc-admet-caco2-wang`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `admet` (suite/_lib/arnew)\
**Areas:** chemistry, biology, machine learning

## Task

Predict the permeability of a compound through a layer of Caco-2 cells, an in-vitro model of absorption in the human intestine (logarithm of the apparent permeability in cm/s) from its molecular structure, given as a SMILES string (endpoint Caco2_Wang of the Therapeutics Data Commons ADMET benchmark group). The measured compounds are split by molecular scaffold into a training, a validation and a test part. The test part is fixed; the candidate is run five times (seeds 1 to 5), each time with a different division of the other compounds into training and validation. It receives the training and validation compounds with their values and the test compounds without, and returns one prediction per test compound. Each of the five runs is a separate process that receives only these arguments: it cannot read the evaluator's files, start other processes or open network connections, and it has 700 seconds. The score is the mean absolute error on the test compounds, averaged over the five runs; lower is better.

## Candidate interface

fit_predict(train_smiles, train_y, valid_smiles, valid_y, test_smiles, info) -> sequence of finite numbers, one per test molecule in the given order: the predicted value. train_smiles, valid_smiles and test_smiles are lists of SMILES strings, train_y and valid_y lists of numbers; info has name, metric, kind ('regression' or 'binary') and seed. Score: mean over the seeds 1 to 5 of the mean absolute error on the test part, minimized.

## The instance

Predicts the log apparent permeability (cm/s) across Caco-2 cell monolayers of 906 drug-like compounds from their SMILES strings, scored by mean absolute error.

## Why it matters

**Problem:** Predict the logarithm of the apparent Caco-2 cell permeability (cm/s) of a drug-like molecule from its structure.

Caco-2 monolayers are the standard in vitro model of intestinal absorption, so predicting permeability from structure screens candidate compounds for oral uptake before synthesis.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [ADME Properties Evaluation in Drug Discovery: Prediction of Caco-2 Cell Permeability Using a Combination of NSGA-II and Boosting](https://doi.org/10.1021/acs.jcim.5b00642): Build a predictive model of Caco-2 cell permeability from molecular structure, using a multi-objective genetic algorithm for descriptor selection and boosting. The task predicts the Caco-2 permeability of the 906-compound set that this paper assembled and modelled; the local split, metric and candidate freedom differ from the paper's.

## Record

Top entry CaliciBoost: 0.256.

**Record validation:** matches a pinned source. leaderboard rank 1 (MAE): CaliciBoost Weibin Ren 0.256. Note: Leaderboard page is a live snapshot, undated.

## Data and requirements

Downloaded by PyTDC at first use into data/admet

Requires: `py:numpy`, `py:sklearn`, `py:rdkit`, `py:PyTDC`. Run `python3 scripts/prepare_new_data.py --family admet` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Leaders are large tuned ensembles or foundation-model fine-tunes.
- Needs PyTDC and rdkit: not in .venv-tasks, so the evaluator restarts itself under tools/venv-admet (created by the README's setup steps).
- Every scoring call of the candidate runs in the main suite's sandbox (a child process with an audit hook, 700 s and 6 GB for one call): it keeps a search from coming across the test labels and is not a security boundary against a candidate written to attack the evaluator.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
