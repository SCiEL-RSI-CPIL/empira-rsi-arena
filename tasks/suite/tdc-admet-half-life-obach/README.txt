# TDC ADMET Half_Life_Obach

**Task id:** `tdc-admet-half-life-obach`\
**Status:** requires_setup\
**Direction:** maximize\
**Family:** `admet` (suite/_lib/arnew)\
**Areas:** chemistry, biology, machine learning

## Task

Predict the elimination half-life of a drug in humans (hours) from its molecular structure, given as a SMILES string (endpoint Half_Life_Obach of the Therapeutics Data Commons ADMET benchmark group). The measured compounds are split by molecular scaffold into a training, a validation and a test part. The test part is fixed; the candidate is run five times (seeds 1 to 5), each time with a different division of the other compounds into training and validation. It receives the training and validation compounds with their values and the test compounds without, and returns one prediction per test compound. Each of the five runs is a separate process that receives only these arguments: it cannot read the evaluator's files, start other processes or open network connections, and it has 700 seconds. The score is the Spearman rank correlation between the predictions and the measured values of the test compounds, averaged over the five runs; higher is better, and only the order of the predictions counts.

## Candidate interface

fit_predict(train_smiles, train_y, valid_smiles, valid_y, test_smiles, info) -> sequence of finite numbers, one per test molecule in the given order: the predicted value. train_smiles, valid_smiles and test_smiles are lists of SMILES strings, train_y and valid_y lists of numbers; info has name, metric, kind ('regression' or 'binary') and seed. Score: mean over the seeds 1 to 5 of the Spearman rank correlation between predictions and measured values on the test part, maximized.

## The instance

Predicts the human intravenous half-life of 667 drugs from their SMILES strings, scored by Spearman rank correlation.

## Why it matters

**Problem:** Predict the human elimination half-life (hours) of a drug after intravenous dosing from its structure.

Half-life sets how long a drug acts and how often it must be dosed, so predicting it from structure helps select candidates with a workable dosing regimen.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Trend Analysis of a Database of Intravenous Pharmacokinetic Parameters in Humans for 670 Drug Compounds](https://doi.org/10.1124/dmd.108.020479): Assemble human intravenous pharmacokinetic parameters (clearance, volume of distribution, half-life) for 670 drugs and analyse their trends with structure and properties. The task predicts the half-life values of this database from structure; the paper analyses the data and does not fit a predictor of the kind scored here.

## Record

Top entry CFA: 0.576.

**Record validation:** matches a pinned source. leaderboard rank 1 (Spearman): CFA Nan Jiang 0.576. Note: Leaderboard page is a live snapshot, undated.

## Data and requirements

Downloaded by PyTDC at first use into data/admet

Requires: `py:numpy`, `py:sklearn`, `py:rdkit`, `py:PyTDC`. Run `python3 scripts/prepare_new_data.py --family admet` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Leaders are large tuned ensembles or foundation-model fine-tunes.
- Needs PyTDC and rdkit: not in .venv-tasks, so the evaluator restarts itself under tools/venv-admet (created by the README's setup steps).
- Every scoring call of the candidate runs in the main suite's sandbox (a child process with an audit hook, 700 s and 6 GB for one call): it keeps a search from coming across the test labels and is not a security boundary against a candidate written to attack the evaluator.
- One evaluation of the starting program took 130 s in the run of 2026-11-11, more than the arena's default limit of 120 s: raise AR_EVAL_TIMEOUT for this task.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
