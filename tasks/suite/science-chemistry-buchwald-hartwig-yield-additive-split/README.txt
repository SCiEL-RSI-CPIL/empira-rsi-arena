# Buchwald-Hartwig yield prediction for unseen additives

**Source collection:** [Molecular and materials design research: source-based research tasks](../../papers/science-chemistry_research.md)\
**Task id:** `science-chemistry-buchwald-hartwig-yield-additive-split`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Predict the yields of palladium-catalysed C-N couplings that use isoxazole additives never seen in training, on the four out-of-sample tests of Ahneman et al. The evaluator scores the mean coefficient of determination (R^2) over the four tests.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

predict_yields(train_reactions, train_yields, test_reactions, info) -> list of len(test_reactions) predicted yields (percent). Buchwald-Hartwig couplings of Ahneman et al. (2018); reaction = {'aryl_halide', 'ligand', 'base', 'additive' (canonical SMILES), 'amine', 'catalyst', 'product', 'reaction_smiles' (rxn_yields format)}. The four out-of-sample tests Test1-Test4 (3057/3055/3058/3055 training and 898/900/897/900 test reactions) are called separately; test additives never occur in training. Each split is predicted in a fresh process that receives the training reactions with yields and the test reactions without yields; info = {'dataset', 'split', 'n_train', 'n_test', 'yield_unit': 'percent'}. Runs in a separate process that receives only these arguments (no test values), with at most 4 threads (OMP/MKL/OpenBLAS set to 4), no network, no subprocesses and no read access to the repository or prepared data; numpy, scipy, scikit-learn, pandas, RDKit and torch are importable; wall-clock limit 200 s per test. Score: mean coefficient of determination R^2 over the four tests (maximize)

## State of the art

No best-known value was verified. Published comparison values: Yield-BERT (Schwaller et al. 2021) R^2 0.84, 0.84, 0.75 and 0.49 on Tests 1-4 (mean 0.73) and the DFT-descriptor random forest of Ahneman et al. 0.80, 0.77, 0.64 and 0.54 (mean 0.69), as tabulated by Baraka and El Kerdawy (2022). Later work may report higher values, and this task's fingerprint random-forest seed already scores 0.753.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Prediction of chemical reaction yields using deep learning](https://doi.org/10.1088/2632-2153/abc81d)
- [Predicting reaction performance in C-N cross-coupling using machine learning](https://doi.org/10.1126/science.aar5169)
- [Multimodal Transformer-based Model for Buchwald-Hartwig and Suzuki-Miyaura Reaction Yield Prediction](https://arxiv.org/abs/2204.14062)

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `real_measurements`, verification `sampled`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `chemistry`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict the yield of a catalytic reaction from its components, including for reaction conditions never tested.

Why it matters: Accurate yield prediction reduces wet-lab experimentation and helps chemists choose catalysts and conditions; Buchwald-Hartwig C-N couplings are among the most used reactions in pharmaceutical synthesis.

- [Predicting reaction performance in C-N cross-coupling using machine learning](https://doi.org/10.1126/science.aar5169): Main research objective: Predict yields of Pd-catalysed C-N cross-couplings from high-throughput data with machine learning, including for unseen additives. Task connection: The task uses its data and its four out-of-sample additive tests.
- [Prediction of chemical reaction yields using deep learning](https://doi.org/10.1088/2632-2153/abc81d): Main research objective: Predict reaction yields from reaction SMILES with transformer models. Task connection: The task uses the same splits and compares with the published R^2.

The cited papers' main objective is the underlying problem the task scores; the local data, splits and limits are stated in the task's limitations.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Split boundaries of rxn_yields (rows before 3058/3056/3059/3056 train); the preparation checks that training and test additives are disjoint. Reference check: the reaction SMILES rebuilt by the evaluator are identical to the output of rxn_yields' generate_buchwald_hartwig_rxns (pinned commit) for all 3955 reactions of each test, and R^2 equals scikit-learn's r2_score. Published R^2 values are compared, not reproduced (no published predictions are available).

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:rdkit`
- `py:sklearn`
- `file:data/chemistry/yields/buchwald_hartwig.json`

## Sources

- paper: https://doi.org/10.1126/science.aar5169 (Ahneman et al. (2018): machine-learning prediction of C-N cross-coupling yields from 4608 nanomole-scale experiments; out-of-sample additive tests 1-4.)
- paper: https://doi.org/10.1088/2632-2153/abc81d (Schwaller et al. (2021): Yield-BERT; split boundaries of the out-of-sample tests and published R^2 values.)
- repository: https://github.com/rxn4chemistry/rxn_yields/tree/d9e6b87ce1b881978490d68bfc00021e3b48127a/data (Buchwald-Hartwig spreadsheet (Test1-Test4 sheets) and Suzuki-Miyaura random splits used by Yield-BERT (pinned commit, MIT).)
- paper: https://arxiv.org/abs/2204.14062 (Baraka and El Kerdawy (2022): comparison tables of published R^2 values on both datasets (Yield-BERT, DFT random forest, MFF and others), from which the comparison values were read.)

## Limitations

- One reaction class with 15 aryl halides, 4 ligands, 3 bases and 23 isoxazole additives that act as deliberate poisons; generalisation to unseen additives says little about other substrates or reaction types.
- The test yields are public in the source repository; candidates or settings tuned on these splits overfit, so treat scores as development scores.
- Published values were read from the comparison table of Baraka and El Kerdawy (2022) because the primary articles' sites were not reachable; they are compared, not reproduced, and no published predictions are available.
- Candidates run in a separate process that never receives held-out values, but the process isolation is an audit-hook policy for honest program search, not a security boundary (ctypes stays available for scientific libraries); run adversarial evaluations in a container.
- The fingerprint random-forest seed (0.753) already exceeds the published Yield-BERT mean (0.73), so the published values are not an upper bound.
