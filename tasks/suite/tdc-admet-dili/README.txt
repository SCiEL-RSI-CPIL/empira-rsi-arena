# TDC ADMET DILI

**Task id:** `tdc-admet-dili`\
**Status:** requires_setup\
**Direction:** maximize\
**Family:** `admet` (suite/_lib/arnew)\
**Areas:** chemistry, biology, machine learning

## Task

Predict whether a drug can cause liver injury (1) or not (0) from its molecular structure, given as a SMILES string (endpoint DILI of the Therapeutics Data Commons ADMET benchmark group). The measured compounds are split by molecular scaffold into a training, a validation and a test part. The test part is fixed; the candidate is run five times (seeds 1 to 5), each time with a different division of the other compounds into training and validation. It receives the training and validation compounds with their values and the test compounds without, and returns one prediction per test compound. Each of the five runs is a separate process that receives only these arguments: it cannot read the evaluator's files, start other processes or open network connections, and it has 700 seconds. The score is the area under the ROC curve on the test compounds, averaged over the five runs; higher is better.

## Candidate interface

fit_predict(train_smiles, train_y, valid_smiles, valid_y, test_smiles, info) -> sequence of finite numbers, one per test molecule in the given order: a score that is larger for class 1 (for example its probability). train_smiles, valid_smiles and test_smiles are lists of SMILES strings, train_y and valid_y lists of numbers; info has name, metric, kind ('regression' or 'binary') and seed. Score: mean over the seeds 1 to 5 of the area under the ROC curve on the test part, maximized.

## The instance

Predicts the binary drug-induced-liver-injury label (1 = causes liver injury) of 475 drugs from their SMILES strings, scored by AUROC.

## Why it matters

**Problem:** Classify drugs as causing or not causing drug-induced liver injury from their structure.

Drug-induced liver injury is a leading cause of safety-related drug withdrawals and trial failures, so predicting it from structure helps remove risky candidates early.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Deep Learning for Drug-Induced Liver Injury](https://doi.org/10.1021/acs.jcim.5b00238): Predict drug-induced liver injury of compounds from structure with deep learning, on a data set of drugs labelled by liver-injury concern. The task asks for the DILI label on the 475-compound set of this paper; local split and metric differ.

## Record

Top entry MiniMol: 0.956.

**Record validation:** matches a pinned source. leaderboard rank 1 (AUROC): MiniMol Blazej Banaszewski 0.956. Note: Leaderboard page is a live snapshot, undated.

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
