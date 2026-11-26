# us-patent-phrase-to-phrase-matching

**Paper:** LoongFlow: Directed Evolutionary Search via a Cognitive Plan-Execute-Summarize Paradigm (https://arxiv.org/pdf/2512.24077)\
**Task id:** `loongflow-us-patent-phrase-to-phrase-matching`\
**Protocol:** `recreated`\
**Direction:** maximize\

## Task

Write a Python fit_predict(train, test_inputs, time_budget_s) function that learns from 39,316 scored patent phrase pairs with their CPC context and predicts the similarity of 9,232 test pairs whose anchors never occur in training, within a 300 s wall-clock budget on 4 CPU threads without pretrained models; scored by Pearson correlation (higher is better).

Benchmark named in the paper: MLE-bench. Paper metric: Pearson correlation.

## Candidate interface

fit_predict(train, test_inputs, time_budget_s) -> float array of shape (n_test,): predicted semantic similarity of each (anchor, target) phrase pair in the given CPC context. train: pandas.DataFrame with columns id, anchor, target, context (CPC class such as 'H01') and score (0, 0.25, 0.5, 0.75 or 1); test_inputs: id, anchor, target, context; test anchors never occur in train. time_budget_s = 300. fit_predict runs in a sandboxed child process: 4 CPU threads (thread settings of the numerical libraries, not an operating-system limit), no GPU, no network, no child processes, no pretrained weights or external data, reads limited to the Python installation and the task's public inputs, writes limited to a temporary working directory, memory limit 3.5 GB. time_budget_s wall-clock seconds start when the candidate's source starts executing in the child (imports and module-level code count); the process is killed 15 s after the budget (score: invalid). The evaluator holds the test labels and recomputes the metric; nothing the candidate reports is trusted. Score (maximize): Pearson correlation with the annotated scores (MLE-bench grade.py; constant predictions are invalid)

Parameters: `{"budget_s": 300, "memory_mb": 3500}`

## Labels

Scientific contribution: `engineering` (tier 3); artifact `model`, domain `machine_learning`, data `public_benchmark`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Data: Google's Patent Phrase Similarity dataset (CC BY 4.0, the data of the competition per its data page) from the Hugging Face copy tasksource/patent-phrase-similarity (revision ff186879). Checked: 36,473 / 2,843 / 9,232 pairs in the train / validation / test splits (48,548 in total, as in the dataset paper), 733 / 57 / 183 anchors with no anchor shared between splits, 106 CPC contexts. The competition's training file has 36,473 rows like the training split; row-level identity was not checked (no Kaggle access).
- Split: the dataset's own anchor-disjoint split; the candidate trains on the training and validation splits (39,316 pairs) and is scored on the test split (9,232 pairs, unseen anchors). MLE-bench instead re-splits the competition's training file at random 90/10 (random_state 0), which shares anchors between training and test and is easier; the copy has no competition ids, so that split cannot be replayed. Scores are therefore not comparable with MLE-bench or Kaggle values, and medal thresholds do not apply. The fine-grained rating column of the dataset is not given (the competition provides only the score). Every candidate is scored on the 9,232 test pairs and that score is the search signal (the evaluator holds back no validation split); LoongFlow's released MLE-bench evaluator (agents/ml_agent/examples/mlebench/eval_program.py at 945c78bc) instead scores candidates on their out-of-fold predictions for the training data and grades the test submission only in a debug mode. The paper states no budget, hardware or split for its MLE-bench runs.
- Metric: Pearson correlation of predicted and annotated scores (scipy.stats.pearsonr, as in MLE-bench's grade.py); constant predictions are invalid.
- Compute: CPU, 4 threads, sandboxed child process without network, pretrained language models, word vectors or CPC title tables (the strongest competition solutions fine-tuned pretrained transformers), 3.5 GB, killed 15 s after a 300 s budget; models are built from the training pairs only. Label isolation as in the other kaggle tasks (the evaluator never imports the candidate; its source runs only in the child, which on Linux with unprivileged user namespaces runs in its own user, mount, PID, network and IPC namespaces where data/kaggle and kaggle-raw of the task cache are empty except this task's public directory and every file system but the working directory and /dev/shm is read-only, on macOS under sandbox-exec, and elsewhere only under an audit hook that native code can bypass; the result field os_sandbox names what was in force). The budget is wall-clock time, so scores depend on the machine's speed and load.
- Seed (cosine similarity of character 2-4-gram TF-IDF vectors, no training) scores Pearson 0.4496 in about 1 s.
- Google's patent phrase similarity dataset (CC BY 4.0) with its anchor-disjoint train/validation/test split.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:sklearn`
- `py:pandas`
- `py:scipy`
- `py:psutil`
- `file:data/kaggle/patent/public/train.csv`
- `file:data/kaggle/patent/public/test.csv`
- `file:data/kaggle/patent/private/test_labels.csv`

## Sources

- paper: https://arxiv.org/pdf/2512.24077 (Table 2)
- repository: https://github.com/baidu-baige/LoongFlow/tree/945c78bc1554f8281aac40320b3599bd68d528d7 (agents/ml_agent/examples/mlebench/eval_program.py)
- archive: https://huggingface.co/datasets/tasksource/patent-phrase-similarity/tree/ff1868796e44ac9ec02e870b447fde6c5bf2d6e4 (copy of Google's Patent Phrase Similarity dataset (train, validation and test splits))
- archive: https://www.kaggle.com/datasets/google/google-patent-phrase-similarity-dataset (original release, CC BY 4.0)
- paper: https://arxiv.org/abs/2208.01171 (Aslanyan and Wetherbee 2022: 48,548 pairs, 973 anchors, anchor-disjoint 75/5/20 split)
- repository: https://github.com/openai/mle-bench/tree/507f92e1138bb6e40dac5c6ee7a6758e6424bf97/mlebench/competitions/us-patent-phrase-to-phrase-matching (grade.py (Pearson correlation))
- benchmark: https://www.kaggle.com/c/us-patent-phrase-to-phrase-matching (competition definition)
