# Suzuki-Miyaura yield prediction on random splits

**Source collection:** [Molecular and materials design research: source-based research tasks](../../papers/science-chemistry_research.md)\
**Task id:** `science-chemistry-suzuki-miyaura-yield-random-split`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Predict the yields of 5760 Suzuki-Miyaura couplings screened by Perera et al. across electrophiles, nucleophiles, ligands, bases and solvents, on ten random 70/30 splits. The evaluator scores the mean coefficient of determination (R^2) over the splits.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

predict_yields(train_reactions, train_yields, test_reactions, info) -> list of len(test_reactions) predicted yields (percent). Suzuki-Miyaura couplings of Perera et al. (2018); reaction = {'reaction_smiles': 'reactants.reagents>>product'}. Ten random 70/30 splits of rxn_yields (4032 training, 1728 test reactions) are called separately. Each split is predicted in a fresh process that receives the training reactions with yields and the test reactions without yields; info = {'dataset', 'split', 'n_train', 'n_test', 'yield_unit': 'percent'}. Runs in a separate process that receives only these arguments (no test values), with at most 4 threads (OMP/MKL/OpenBLAS set to 4), no network, no subprocesses and no read access to the repository or prepared data; numpy, scipy, scikit-learn and RDKit are importable (pandas and torch only where the environment provides them; they are not requirements of the task); wall-clock limit 80 s per split. Score: mean R^2 over the ten splits (maximize)

## State of the art

No best-known value was verified. Published comparison values: Yield-BERT (Schwaller et al. 2021) mean R^2 0.81 +- 0.01 over the ten splits and the multimodal BERT of Baraka and El Kerdawy (2022) 0.833 +- 0.01; later work may report higher values, and this task's seed scores 0.823.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Prediction of chemical reaction yields using deep learning](https://doi.org/10.1088/2632-2153/abc81d)
- [Multimodal Transformer-based Model for Buchwald-Hartwig and Suzuki-Miyaura Reaction Yield Prediction](https://arxiv.org/abs/2204.14062)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `sampled`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `chemistry`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict the yield of Suzuki-Miyaura cross-couplings across substrates and conditions.

Why it matters: Suzuki-Miyaura coupling is a workhorse C-C bond formation in drug synthesis; predicting yields across ligands, bases and solvents reduces condition screening.

- [Prediction of chemical reaction yields using deep learning](https://doi.org/10.1088/2632-2153/abc81d): Main research objective: Predict reaction yields from reaction SMILES with transformer models. Task connection: The task uses the same ten random splits of the Perera et al. data and compares with the published R^2.
- [A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow](https://doi.org/10.1126/science.aap9112): Main research objective: Screen thousands of Suzuki-Miyaura reaction conditions automatically to find high-yielding conditions. Task connection: Its 5760 measured yields are the data of the task.

The cited papers' main objective is the underlying problem the task scores; the local data, splits and limits are stated in the task's limitations.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- The ten rxn_yields random splits, first 4032 reactions training as in Schwaller et al. (2021); yields are converted from fractions to percent. Published values are compared, not reproduced.
- The rxn_yields evaluation clips predictions to [0, 100] before computing R^2; this evaluator scores the returned values unchanged, so a candidate should clip its own predictions.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:rdkit`
- `py:sklearn`
- `file:data/chemistry/yields/suzuki_miyaura.json`

## Sources

- paper: https://doi.org/10.1126/science.aap9112 (Perera et al. (2018): automated nanomole-scale screening of 5760 Suzuki-Miyaura reactions; source of the yields.)
- paper: https://doi.org/10.1088/2632-2153/abc81d (Schwaller et al. (2021): Yield-BERT; the ten random splits (first 4032 reactions training) and published R^2.)
- repository: https://github.com/rxn4chemistry/rxn_yields/tree/d9e6b87ce1b881978490d68bfc00021e3b48127a/data (Buchwald-Hartwig spreadsheet (Test1-Test4 sheets) and Suzuki-Miyaura random splits used by Yield-BERT (pinned commit, MIT).)
- paper: https://arxiv.org/abs/2204.14062 (Baraka and El Kerdawy (2022): comparison tables of published R^2 values on both datasets (Yield-BERT, DFT random forest, MFF and others), from which the comparison values were read.)

## Limitations

- Random splits of a full factorial screen interpolate between measured combinations of the same components; they do not test transfer to new substrates or conditions.
- The test yields are public in the source repository; candidates or settings tuned on these splits overfit, so treat scores as development scores.
- Published values were read from the comparison table of Baraka and El Kerdawy (2022) because the primary articles' sites were not reachable; they are compared, not reproduced, and no published predictions are available.
- Candidates run in a separate process that never receives held-out values, but the process isolation is an audit-hook policy for honest program search, not a security boundary (ctypes stays available for scientific libraries); run adversarial evaluations in a container.
