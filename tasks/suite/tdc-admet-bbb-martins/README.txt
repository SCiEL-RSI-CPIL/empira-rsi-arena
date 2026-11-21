# TDC ADMET BBB_Martins

**Task id:** `tdc-admet-bbb-martins`\
**Status:** requires_setup\
**Direction:** maximize\
**Family:** `admet` (suite/_lib/arnew)\
**Areas:** chemistry, biology, machine learning

## Task

Predict whether a compound crosses the blood-brain barrier (1) or not (0) from its molecular structure, given as a SMILES string (endpoint BBB_Martins of the Therapeutics Data Commons ADMET benchmark group). The measured compounds are split by molecular scaffold into a training, a validation and a test part. The test part is fixed; the candidate is run five times (seeds 1 to 5), each time with a different division of the other compounds into training and validation. It receives the training and validation compounds with their values and the test compounds without, and returns one prediction per test compound. Each of the five runs is a separate process that receives only these arguments: it cannot read the evaluator's files, start other processes or open network connections, and it has 700 seconds. The score is the area under the ROC curve on the test compounds, averaged over the five runs; higher is better.

## Candidate interface

fit_predict(train_smiles, train_y, valid_smiles, valid_y, test_smiles, info) -> sequence of finite numbers, one per test molecule in the given order: a score that is larger for class 1 (for example its probability). train_smiles, valid_smiles and test_smiles are lists of SMILES strings, train_y and valid_y lists of numbers; info has name, metric, kind ('regression' or 'binary') and seed. Score: mean over the seeds 1 to 5 of the area under the ROC curve on the test part, maximized.

## The instance

Predicts the binary blood-brain-barrier penetration class (1 = penetrates) of 1,975 compounds from their SMILES strings, scored by AUROC.

## Why it matters

**Problem:** Classify molecules as able or unable to cross the blood-brain barrier from their structure.

Drugs for the central nervous system must cross the barrier, while peripherally acting drugs should not, so penetration is predicted early in design.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling](https://doi.org/10.1021/ci300124c): Model blood-brain barrier penetration of compounds from molecular structure with Bayesian classifiers, using a large compiled data set. The task asks for the BBB penetration label on the compound set compiled in this paper (1,975 in TDC); local split and metric differ.

## Record

Top entry MiniMol: 0.924.

**Record validation:** matches a pinned source. leaderboard rank 1 (AUROC): MiniMol Blazej Banaszewski 0.924. Note: Leaderboard page is a live snapshot, undated.

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
