# ADME: aqueous solubility of drug-like compounds (Biogen public set)

**Source collection:** [Molecular and materials design research: source-based research tasks](../../papers/science-chemistry_research.md)\
**Task id:** `science-chemistry-adme-solubility-biogen`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Predict the aqueous solubility at pH 6.8 (log10 ug/mL) of 435 test compounds from 1738 training compounds measured in the same Biogen assay. The evaluator scores the mean absolute error on the authors' official test split.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

predict(train_smiles, train_values, test_smiles, info) -> list of len(test_smiles) floats: predicted log10 property values of the test molecules. The training split of the Biogen public ADME set (Fang et al. 2023; official split of molecularinformatics/Computational-ADME) is given as SMILES with measured log10 values, the test split as SMILES only; info = {'endpoint', 'property', 'n_train', 'n_test'}. Runs in a separate process that receives only these arguments (no test values), with at most 4 threads (OMP/MKL/OpenBLAS set to 4), no network, no subprocesses and no read access to the repository or prepared data; numpy, scipy, scikit-learn and RDKit are importable (pandas and torch only where the environment provides them; they are not requirements of the task); wall-clock limit 540 s. Score: mean absolute error on the official test split (log10 units; minimize); RMSE, R^2, Pearson and Spearman correlations and the MAE of the training-mean predictor are reported

## State of the art

No verified published value exists for this exact split in the sources we could read; the training-mean predictor's MAE is reported with every score for scale.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).


Parameters: `{"endpoint": "solubility"}`

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `sampled`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `chemistry`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict the aqueous solubility at pH 6.8 of drug-like molecules from their chemical structure.

Why it matters: Poor solubility limits absorption and causes failures in assays and formulation, so predicting it from structure guides compound design.

- [Prospective Validation of Machine Learning Algorithms for Absorption, Distribution, Metabolism, and Excretion Prediction: An Industrial Perspective](https://doi.org/10.1021/acs.jcim.3c00160): Main research objective: Build and prospectively validate machine-learning models for six in vitro ADME endpoints, including solubility, liver-microsomal stability and MDR1-MDCK efflux, in drug discovery. Task connection: The task uses the authors' public aqueous solubility at pH 6.8 data and official split and scores the test-set error of a predictor.

The cited papers' main objective is the underlying problem the task scores; the local data, splits and limits are stated in the task's limitations.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Official train/test files of the authors' repository at a pinned commit; the candidate runs in chem_sandbox without access to the test values. No published value on these splits could be read, so none is reproduced.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:rdkit`
- `py:sklearn`
- `file:data/chemistry/adme/solubility_train.csv`
- `file:data/chemistry/adme/solubility_test.csv`

## Sources

- paper: https://doi.org/10.1021/acs.jcim.3c00160 (Fang et al. (2023): prospective validation of ML models for six in vitro ADME endpoints; disclosure of the 3521-compound public set.)
- repository: https://github.com/molecularinformatics/Computational-ADME/tree/b00df003de117ce9e5b381afd886095c5f2af2d5/MPNN (Official train/test CSV files of each endpoint (pinned commit), MIT licence.)

## Limitations

- The test values are public in the authors' repository; a candidate that embeds them, or hyperparameters tuned on them, overfits. Treat scores as development scores.
- The split files differ slightly from the values in ADME_public_set_3521.csv (up to 0.1-0.3 log units for a few compounds), and one MDR1-MDCK test compound equals a training compound after canonicalisation; the official split files are used unchanged.
- The article's own results on this split could not be read (publisher site not reachable), so no published value is reproduced.
- Candidates run in a separate process that never receives held-out values, but the process isolation is an audit-hook policy for honest program search, not a security boundary (ctypes stays available for scientific libraries); run adversarial evaluations in a container.
