# stanford-covid-vaccine

**Paper:** LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm (https://arxiv.org/pdf/2512.24077)\
**Task id:** `loongflow-stanford-covid-vaccine`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Write a Python fit_predict(train, test_inputs, time_budget_s) function that learns from 1,349 OpenVaccine RNA sequences with per-position degradation measurements and predicts reactivity, deg_Mg_pH10 and deg_Mg_50C at the first 68 positions of 240 test RNAs, within a 300 s wall-clock budget on 4 CPU threads; scored by MCRMSE (lower is better).

Benchmark named in the paper: MLE-bench. Paper metric: mean columnwise root mean squared error.

## Candidate interface

fit_predict(train, test_inputs, time_budget_s) -> float array of shape (n_test, 68, 3): reactivity, deg_Mg_pH10 and deg_Mg_50C at the first seq_scored = 68 positions of each test RNA. train: pandas.DataFrame with the competition's train.json columns: index, id, sequence (107 nt), structure (dot-bracket), predicted_loop_type, signal_to_noise, SN_filter, seq_length, seq_scored, the five per-position target lists reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C (68 values each) and their *_error lists; test_inputs: index, id, sequence, structure, predicted_loop_type, seq_length, seq_scored. time_budget_s = 300. fit_predict runs in a sandboxed child process: 4 CPU threads (thread settings of the numerical libraries, not an operating-system limit), no GPU, no network, no child processes, no pretrained weights or external data, reads limited to the Python installation and the task's public inputs, writes limited to a temporary working directory, memory limit 3.5 GB. time_budget_s wall-clock seconds start when the candidate's source starts executing in the child (imports and module-level code count); the process is killed 15 s after the budget (score: invalid). The evaluator holds the test labels and recomputes the metric; nothing the candidate reports is trusted. Score (minimize): MCRMSE, the mean over the three scored targets of the RMSE pooled over all scored positions (MLE-bench grade.py)

Parameters: `{"budget_s": 300, "memory_mb": 3500}`

## Labels

Scientific contribution: `engineering` (tier 3); artifact `model`, domain `natural_science`, data `real_measurements`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Data: Kaggle_RYOS_trainset.csv of eternagame/KaggleOpenVaccine (commit b95af835, MIT), the 1,589 rows of the competition's train.json with SN_filter = 1, in their original order with the original row index (0 to 2399) and the competition's columns; its extra columns derived from labels or models (k_deg, half-life, model predictions) are dropped.
- Split replayed from MLE-bench: the test set is the 240 rows that MLE-bench's prepare.py samples among the SN_filter > 0 rows (pandas sample, n = 240, random_state 0). Running MLE-bench's prepare.py unchanged on a 2,400-row train.json made of the public rows at their original index plus placeholders for the 811 missing rows selects the same 240 RNAs and reproduces, byte for byte, the MD5 checksums MLE-bench records for its answers file (private test.csv: ids and all five targets at 107 positions) and its sample submission. Its prepared test.json (inputs) is not reproduced under any column order or column subset, so some input strings (sequence, structure, loop type) or their serialization may differ from Kaggle's train.json; this was not resolved (no Kaggle access). Deviation: the training set is the remaining 1,349 SN_filter = 1 rows; MLE-bench's training set also contains the 811 low signal-to-noise rows (SN_filter = 0) of train.json, which the public file omits (2,160 rows in total). Every candidate is scored on these 240 test RNAs and that score is the search signal (the evaluator holds back no validation split); LoongFlow's released MLE-bench evaluator (agents/ml_agent/examples/mlebench/eval_program.py at 945c78bc) instead scores candidates on their out-of-fold predictions for the training data and grades the test submission only in a debug mode. The paper states no budget, hardware or split for its MLE-bench runs.
- Metric exactly as MLE-bench's grade.py: MCRMSE, the mean over reactivity, deg_Mg_pH10 and deg_Mg_50C of the RMSE pooled over the first seq_scored = 68 positions of all test RNAs (config.yaml names a log-loss grader, but grade.py computes MCRMSE). The candidate returns only the scored array (240, 68, 3) instead of the 107-position five-column submission; the unscored columns and positions carry no weight in either.
- The test RNAs come from the competition's training sequences (as in MLE-bench), not from the 130-nt private test constructs, so Kaggle leaderboard values and medal thresholds are not comparable.
- Compute: CPU, 4 threads, sandboxed child process without network, external folding packages or pretrained models (structure and predicted_loop_type columns are given), 3.5 GB, killed 15 s after a 300 s budget. Label isolation as in the other kaggle tasks (the evaluator never imports the candidate; its source runs only in the child, which on Linux with unprivileged user namespaces runs in its own user, mount, PID, network and IPC namespaces where data/kaggle and kaggle-raw of the task cache are empty except this task's public directory and every file system but the working directory and /dev/shm is read-only, on macOS under sandbox-exec, and elsewhere only under an audit hook that native code can bypass; the result field os_sandbox names what was in force). The budget is wall-clock time, so scores depend on the machine's speed and load.
- Seed (mean target per nucleotide and loop type) scores 0.4237 MCRMSE in under a second.
- Kaggle_RYOS_trainset.csv of eternagame/KaggleOpenVaccine (MIT), MLE-bench's SN_filter test sample replayed.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:sklearn`
- `py:pandas`
- `py:psutil`
- `file:data/kaggle/openvaccine/public/train.jsonl`
- `file:data/kaggle/openvaccine/public/test.jsonl`
- `file:data/kaggle/openvaccine/private/test_labels.npz`

## Sources

- paper: https://arxiv.org/pdf/2512.24077 (Table 2)
- repository: https://github.com/baidu-baige/LoongFlow/tree/945c78bc1554f8281aac40320b3599bd68d528d7 (agents/ml_agent/examples/mlebench/eval_program.py)
- repository: https://github.com/eternagame/KaggleOpenVaccine/tree/b95af835cc38bd7dc55ed6e7397d124787f86996/data/Kaggle_RYOS_data (Kaggle_RYOS_trainset.csv (MIT): the SN_filter = 1 rows of the competition's train.json)
- paper: https://doi.org/10.1038/s42256-022-00571-8 (Wayment-Steele et al. 2022, Deep learning models for predicting RNA degradation via dual crowdsourcing)
- repository: https://github.com/openai/mle-bench/tree/507f92e1138bb6e40dac5c6ee7a6758e6424bf97/mlebench/competitions/stanford-covid-vaccine (prepare.py (240 SN_filter > 0 test rows, random_state 0) and grade.py (MCRMSE over reactivity, deg_Mg_pH10, deg_Mg_50C on scored positions))
- benchmark: https://www.kaggle.com/c/stanford-covid-vaccine (competition definition)
