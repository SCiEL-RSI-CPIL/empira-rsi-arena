# TDC ADMET CYP3A4_Veith

**Task id:** `tdc-admet-cyp3a4-veith`\
**Status:** requires_setup\
**Direction:** maximize\
**Family:** `admet` (suite/_lib/arnew)\
**Areas:** chemistry, biology, machine learning

## Task

Predict whether a compound inhibits the drug-metabolising enzyme cytochrome P450 3A4 (1) or not (0) from its molecular structure, given as a SMILES string (endpoint CYP3A4_Veith of the Therapeutics Data Commons ADMET benchmark group). The measured compounds are split by molecular scaffold into a training, a validation and a test part. The test part is fixed; the candidate is run five times (seeds 1 to 5), each time with a different division of the other compounds into training and validation. It receives the training and validation compounds with their values and the test compounds without, and returns one prediction per test compound. Each of the five runs is a separate process that receives only these arguments: it cannot read the evaluator's files, start other processes or open network connections, and it has 700 seconds. The score is the area under the precision-recall curve on the test compounds, with class 1 as the positive class, averaged over the five runs; higher is better.

## Candidate interface

fit_predict(train_smiles, train_y, valid_smiles, valid_y, test_smiles, info) -> sequence of finite numbers, one per test molecule in the given order: a score that is larger for class 1 (for example its probability). train_smiles, valid_smiles and test_smiles are lists of SMILES strings, train_y and valid_y lists of numbers; info has name, metric, kind ('regression' or 'binary') and seed. Score: mean over the seeds 1 to 5 of the area under the precision-recall curve (class 1 positive) on the test part, maximized.

## The instance

Predicts the binary CYP3A4 inhibition label (1 = inhibitor, from a quantitative high-throughput screen) of 12,328 compounds from their SMILES strings, scored by AUPRC.

## Why it matters

**Problem:** Classify molecules as inhibitors or non-inhibitors of the cytochrome P450 isozyme CYP3A4 from their structure.

CYP3A4 is one of the main drug-metabolising enzymes; its inhibition by a co-administered drug raises the exposure of other drugs and causes drug-drug interactions, so inhibition is screened for early.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Comprehensive characterization of cytochrome P450 isozyme selectivity across chemical libraries](https://doi.org/10.1038/nbt.1581): Measure the inhibition of five major cytochrome P450 isozymes by about 17,000 compounds in a quantitative high-throughput screen and characterise isozyme selectivity. The task predicts, from structure, the inhibition label of the isozyme named in the task, which this screen measured for about 12,000 compounds; the paper's own aim is the measurement, not a predictor, and the local split and metric are different.

## Record

Top entry MapLight + GNN: 0.916.

**Record validation:** matches a pinned source. leaderboard rank 1 (AUPRC): MapLight + GNN Jim Notwell 0.916. Note: Leaderboard page is a live snapshot, undated.

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
