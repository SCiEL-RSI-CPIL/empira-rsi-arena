# denoising-dirty-documents

**Paper:** LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm (https://arxiv.org/pdf/2512.24077)\
**Task id:** `loongflow-denoising-dirty-documents`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Write a Python fit_predict(train, test_inputs, time_budget_s) function that learns from 144 noisy/clean NoisyOffice document image pairs and returns a cleaned image (intensities on a 0-1 scale) for each of 72 noisy test images of an unseen text passage, within a 300 s wall-clock budget on 4 CPU threads; scored by pixel RMSE against the clean pages (lower is better).

Benchmark named in the paper: MLE-bench. Paper metric: root mean squared error.

## Candidate interface

fit_predict(train, test_inputs, time_budget_s) -> list of float arrays. train: dict with 'noisy' and 'clean', lists of 2-D uint8 grey-level images (0 = black, 255 = white; pairs share their shape); test_inputs: dict with 'noisy' (list of 2-D uint8 images); return one cleaned image per test image, same order and shape, intensities on the [0, 1] scale (finite; values are not clipped). time_budget_s = 300. fit_predict runs in a sandboxed child process: 4 CPU threads (thread settings of the numerical libraries, not an operating-system limit), no GPU, no network, no child processes, no pretrained weights or external data, reads limited to the Python installation and the task's public inputs, writes limited to a temporary working directory, memory limit 3.5 GB. time_budget_s wall-clock seconds start when the candidate's source starts executing in the child (imports and module-level code count); the process is killed 15 s after the budget (score: invalid). The evaluator holds the test labels and recomputes the metric; nothing the candidate reports is trusted. Score (minimize): root mean squared error over all pixels of all test images against the clean ground truth divided by 255 (MLE-bench grade.py)

Parameters: `{"budget_s": 300, "memory_mb": 3500}`

## Labels

Scientific contribution: `engineering` (tier 3); artifact `model`, domain `machine_learning`, data `public_benchmark`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Analogue built from the UCI NoisyOffice corpus (CC BY 4.0), not from Kaggle's files. The competition credits the NoisyOffice authors and UCI for its data, and the simulated part matches it in count and size (216 noisy grey-level images of 540 x 258 or 540 x 420 pixels; Kaggle had 144 training and 72 test images), but pixel identity with Kaggle's files was not checked (no Kaggle access).
- Each NoisyOffice partition (TR, VA, TE) is one text passage typeset in 18 fonts, and every typeset page appears with 4 noise types (folded, wrinkled, coffee stains, footprints). Split by text: the 144 noisy images of the TR and VA passages (with their clean pages) are the training set and the 72 noisy images of the TE passage the test set. MLE-bench instead re-splits Kaggle's 144 training images at random 80/20 (random_state 0), which puts other noise versions of the same typeset page into training; the text-disjoint split here is harder, so scores are not comparable with MLE-bench or Kaggle values and medal thresholds do not apply. Every candidate is scored on this test set and that score is the search signal (the evaluator holds back no validation split); LoongFlow's released MLE-bench evaluator (agents/ml_agent/examples/mlebench/eval_program.py at 945c78bc) instead scores candidates on their out-of-fold predictions for the training data and grades the test submission only in a debug mode. The paper states no budget, hardware or split for its MLE-bench runs.
- Images are handed over as uint8 arrays (the noisy files are RGB PNGs with three identical channels and are reduced to one channel exactly); presentation order is seeded and carries no font, noise or passage information. Metric: RMSE over every pixel of every test image against the clean grey level / 255, pooled over images as in MLE-bench's grade.py; predictions are not clipped.
- Compute: fit_predict runs on CPU with 4 threads in a sandboxed child process (no network, no GPU, no pretrained weights, 3.5 GB memory) and is killed 15 s after a 300 s wall-clock budget; models are trained from scratch. The budget is wall-clock time, so scores depend on the machine's speed and load.
- Label isolation as in the other kaggle tasks: clean test pages are in data/kaggle/noisyoffice/private, which the child cannot read ; the evaluator never imports the candidate; its source runs only in the child, which on Linux with unprivileged user namespaces runs in its own user, mount, PID, network and IPC namespaces where data/kaggle and kaggle-raw of the task cache are empty except this task's public directory and every file system but the working directory and /dev/shm is read-only, on macOS under sandbox-exec, and elsewhere only under an audit hook that native code can bypass; the result field os_sandbox names what was in force.
- Seed (lookup table from noisy grey level to mean clean intensity) scores 0.0903 RMSE in under a second.
- UCI NoisyOffice simulated set (the corpus the Kaggle competition was built from), split by underlying text.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:pandas`
- `py:psutil`
- `file:data/kaggle/noisyoffice/public/train_noisy.npz`
- `file:data/kaggle/noisyoffice/public/train_clean.npz`
- `file:data/kaggle/noisyoffice/public/test_noisy.npz`
- `file:data/kaggle/noisyoffice/private/test_clean.npz`

## Sources

- paper: https://arxiv.org/pdf/2512.24077 (Table 2)
- repository: https://github.com/baidu-baige/LoongFlow/tree/945c78bc1554f8281aac40320b3599bd68d528d7 (agents/ml_agent/examples/mlebench/eval_program.py)
- archive: https://archive.ics.uci.edu/dataset/318/noisyoffice (NoisyOffice (Espana-Boquera, Pastor-Pellicer, Castro-Bleda, Zamora-Martinez; CC BY 4.0): SimulatedNoisyOffice noisy and clean grey-level images)
- repository: https://github.com/openai/mle-bench/tree/507f92e1138bb6e40dac5c6ee7a6758e6424bf97/mlebench/competitions/denoising-dirty-documents (prepare.py (ground truth = clean grey level / 255) and grade.py (RMSE over all pixels))
- benchmark: https://www.kaggle.com/c/denoising-dirty-documents (competition definition; its acknowledgement names the NoisyOffice authors and the UCI repository)
