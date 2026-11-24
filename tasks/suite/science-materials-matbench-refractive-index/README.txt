# Matbench: refractive index of inorganic crystals

**Source collection:** [Molecular and materials design research: source-based research tasks](../../papers/science-chemistry_research.md)\
**Task id:** `science-materials-matbench-refractive-index`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Predict the DFT refractive index of 4764 inorganic crystals from their structure. The evaluator scores the mean absolute error over Matbench's five official cross-validation folds.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

fit_predict(train_inputs, train_targets, test_inputs, info) -> list of len(test_inputs) floats. Inputs are ordered crystal structures {'lattice': 3x3 lattice vectors as rows (Angstrom), 'species': element symbols, 'frac_coords': fractional coordinates per site}. Called once per official Matbench v0.1 fold (five calls, KFold(5, shuffle, random_state=18012019)), each in a fresh process that receives that fold's training inputs with targets and its held-out inputs without targets; info = {'task', 'input_type', 'target', 'unit', 'fold', 'n_train', 'n_test'}. Runs in a separate process that receives only these arguments (no test values), with at most 4 threads (OMP/MKL/OpenBLAS set to 4), no network, no subprocesses and no read access to the repository or prepared data; numpy, scipy, scikit-learn and RDKit are importable (pandas and torch only where the environment provides them; they are not requirements of the task); wall-clock limit 170 s per fold. Score: mean over the five folds of the held-out mean absolute error in target units (minimize), the Matbench leaderboard metric

## State of the art

Best mean MAE on the Matbench v0.1 leaderboard for matbench_dielectric: 0.2711 dimensionless (MODNet (v0.1.12)), read on 2026-10-10.

Reference value: best known value `0.2711` (reviewed 2026-10-10).

- [Matbench leaderboard: matbench_dielectric](https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_dielectric/)

Parameters: `{"task": "dielectric"}`

## Labels

Scientific contribution: `method` (tier 2); artifact `model`, domain `natural_science`, data `public_benchmark`, verification `sampled`, reference `leaderboard`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `materials_science`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Predict the refractive index of inorganic crystals from their structure.

Why it matters: Refractive index and dielectric response decide the use of materials in optics, photonics and capacitors; screening needs fast predictions before expensive calculations.

- [High-throughput screening of inorganic compounds for the discovery of novel dielectric and optical materials](https://doi.org/10.1038/sdata.2016.134): Main research objective: Discover dielectric and optical materials by computing dielectric constants and refractive indices of many compounds. Task connection: The task predicts those refractive indices from structure.

The cited papers' main objective is the underlying problem the task scores; the local data, splits and limits are stated in the task's limitations.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Official Matbench v0.1 data (SHA-256 equal to matminer's recorded hashes) and folds; structures are passed as plain lattice, species and fractional coordinates instead of pymatgen objects. Reference check: rescoring the stored predictions of AMMExpress v2020 and RF-SCM/Magpie reproduces their leaderboard MAEs exactly.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:rdkit`
- `py:sklearn`
- `file:data/chemistry/matbench/dielectric.json`

## Sources

- paper: https://doi.org/10.1038/sdata.2016.134 (Petousis et al. (2017): high-throughput DFPT screening of inorganic compounds for dielectric and optical materials; source of the targets.)
- benchmark: https://doi.org/10.1038/s41524-020-00406-3 (Dunn et al. (2020): Matbench v0.1 tasks, nested cross-validation protocol and MAE metric.)
- repository: https://github.com/materialsproject/matbench/tree/936176db18ca4cd7b38cbd957c017a5bac770c6b (matbench_v0.1_validation.json (official folds) and stored per-fold predictions of published submissions (pinned commit).)
- benchmark: https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_dielectric/ (Per-task Matbench v0.1 leaderboard (mean MAE over the five folds).)

## Limitations

- All targets are public in the Matbench files and every sample is a test sample of one fold; candidates or settings tuned on these folds overfit, so treat scores as development scores.
- Leaderboard entries were trained without this task's limits (170 s and at most 4 CPU threads per fold, CPU only), so leaderboard values are reference points rather than targets attainable under the same budget.
- Targets are DFT (DFPT) refractive indices with a few extreme values (up to 62) that dominate fold maxima; Materials Project data retrieved in 2019.
- Candidates run in a separate process that never receives held-out values, but the process isolation is an audit-hook policy for honest program search, not a security boundary (ctypes stays available for scientific libraries); run adversarial evaluations in a container.
