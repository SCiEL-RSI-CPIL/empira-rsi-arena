# TDC ADMET LD50_Zhu

**Task id:** `tdc-admet-ld50-zhu`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `admet` (suite/_lib/arnew)\
**Areas:** chemistry, biology, machine learning

## Task

Predict the acute oral toxicity of a compound, the dose that kills half of the test animals (LD50 as log(1/(mol/kg)), so a larger value means a more toxic compound) from its molecular structure, given as a SMILES string (endpoint LD50_Zhu of the Therapeutics Data Commons ADMET benchmark group). The measured compounds are split by molecular scaffold into a training, a validation and a test part. The test part is fixed; the candidate is run five times (seeds 1 to 5), each time with a different division of the other compounds into training and validation. It receives the training and validation compounds with their values and the test compounds without, and returns one prediction per test compound. Each of the five runs is a separate process that receives only these arguments: it cannot read the evaluator's files, start other processes or open network connections, and it has 700 seconds. The score is the mean absolute error on the test compounds, averaged over the five runs; lower is better.

## Candidate interface

fit_predict(train_smiles, train_y, valid_smiles, valid_y, test_smiles, info) -> sequence of finite numbers, one per test molecule in the given order: the predicted value. train_smiles, valid_smiles and test_smiles are lists of SMILES strings, train_y and valid_y lists of numbers; info has name, metric, kind ('regression' or 'binary') and seed. Score: mean over the seeds 1 to 5 of the mean absolute error on the test part, minimized.

## The instance

Predicts the rat oral LD50 (log scale, larger = more toxic) of 7,385 compounds from their SMILES strings, scored by mean absolute error.

## Why it matters

**Problem:** Predict the rat acute oral toxicity (LD50) of a chemical from its structure.

Acute oral toxicity testing uses many animals, so structure-based prediction of LD50 supports hazard assessment and the reduction of animal testing.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Quantitative Structure-Activity Relationship Modeling of Rat Acute Toxicity by Oral Exposure](https://doi.org/10.1021/tx900189p): Build QSAR models of rat acute oral toxicity (LD50) from a large set of compounds with several modelling approaches. The task predicts the LD50 value of the 7,385-compound set of this paper (TDC states the data were provided by its authors); local split and metric differ.

## Record

Top entry BaseBoosting: 0.552.

**Record validation:** matches a pinned source. leaderboard rank 1 (MAE): BaseBoosting KyQVZ6b2 David Huang 0.552. Note: Leaderboard page is a live snapshot, undated.

## Data and requirements

Downloaded by PyTDC at first use into data/admet

Requires: `py:numpy`, `py:sklearn`, `py:rdkit`, `py:PyTDC`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family admet` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Leaders are large tuned ensembles or foundation-model fine-tunes.
- Needs PyTDC and rdkit: not in .venv-tasks, so the evaluator restarts itself under tools/venv-admet (created by the README's setup steps).
- Every scoring call of the candidate runs in the main suite's sandbox (a child process with an audit hook, 700 s and 6 GB for one call): it keeps a search from coming across the test labels and is not a security boundary against a candidate written to attack the evaluator.
- One evaluation of the starting program took 1170 s in the run of 2026-11-11, more than the arena's default limit of 120 s: raise AR_EVAL_TIMEOUT for this task.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
