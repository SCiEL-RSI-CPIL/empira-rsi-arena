# Software engineering: log parsing (event template identification) on the 16 Loghub-2k data sets

**Source collection:** [Software engineering research: source-based research tasks](../../papers/science-software_engineering.md)\
**Task id:** `science-software-log-parsing-loghub2k`\
**Protocol:** `recreated_verified`\
**Direction:** maximize\

## Task

Group the messages of 16 system logs (2,000 lines each, from HDFS, Hadoop, Spark, Linux, OpenSSH and others) by the log statement that produced them, given only the message texts. The score is the mean grouping accuracy against the corrected ground-truth events.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

parse(dataset) -> list. dataset = {'name': one of Android, Apache, BGL, HDFS, HPC, Hadoop, HealthApp, Linux, Mac, OpenSSH, OpenStack, Proxifier, Spark, Thunderbird, Windows, Zookeeper, 'messages': [2000 str]} gives the message contents of 2,000 log lines of one system in their original order (header fields such as time, level and component already removed). Return one group label (int or str) per message; messages with equal labels are claimed to come from the same log statement (event template). Grouping accuracy of a data set: the share of messages whose group contains exactly the messages of their ground-truth event, no more and no fewer. Score: mean grouping accuracy over the 16 data sets, maximised. parse() is called once per data set, all data sets in one sandboxed subprocess (no files, network or subprocesses; 120 s CPU in total, 2 GB); per-data-set settings are allowed, as in the published benchmarks; it should be deterministic (seed any randomness, do not read clocks).

## State of the art

No directly comparable published value exists for the corrected labels. As a reference computed with this evaluator, logparser's Drain (He et al. 2017) with its published benchmark settings scores 0.8605 mean grouping accuracy on these labels; on the original Loghub-2k labels it scores 0.8654, reproducing the published table exactly. Language-model-based parsers report higher accuracies but are outside this CPU contract.

Reference value: no documented value for this finite instance (reviewed 2026-10-11).

- [Drain: An Online Log Parsing Approach with Fixed Depth Tree](https://doi.org/10.1109/ICWS.2017.13)
- [Tools and Benchmarks for Automated Log Parsing](https://doi.org/10.1109/ICSE-SEIP.2019.00021)

Parameters: `{"cpu_seconds": 120}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `software_engineering`, data `public_benchmark`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `software_engineering`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Log parsing: recover, from raw log messages, which log statement (event template) produced each message, so that logs can be turned into event sequences for monitoring, anomaly detection and diagnosis.

Why it matters: Large systems emit millions of unstructured log lines whose statements are unknown or change over time; template identification is the first step of nearly all automated log analysis, and its accuracy directly affects downstream anomaly detection.

- [Drain: An Online Log Parsing Approach with Fixed Depth Tree](https://doi.org/10.1109/ICWS.2017.13): Main research objective: Parse streaming log messages into event templates accurately and efficiently. Task connection: The task asks for the same grouping of messages into templates and uses Drain as the reference.
- [Clustering event logs using iterative partitioning](https://doi.org/10.1145/1557019.1557154): Main research objective: Find the message types (templates) of event logs by clustering log messages. Task connection: The task scores such a clustering of log messages against ground-truth events.
- [Guidelines for assessing the accuracy of log message template identification techniques](https://doi.org/10.1145/3510003.3510101): Main research objective: Assess log template identification accurately, including corrected ground truth for Loghub-2k. Task connection: The task uses these corrected labels and the grouping-accuracy metric.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Data: Loghub-2k with the ground truth corrected by Khan et al. (ICSE 2022), from the logparser repository (data/loghub_2k_corrected at commit d9d41807; Loghub licence: free for research or academic use with attribution, not redistributed). The grouping accuracy and pairwise F1 implemented here agree exactly with logparser's evaluator on Drain's output for all 16 data sets, and running logparser's Drain with its benchmark settings reproduces its published accuracies (mean 0.8654 on the original labels; 0.8605 on the corrected labels used here); hence recreated_verified. The candidate receives message contents instead of raw lines, so header parsing (log formats) is not part of the task.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/software/logparse/datasets.json`

## Sources

- paper: https://doi.org/10.1109/ICWS.2017.13 (He, Zhu, Zheng and Lyu (ICWS 2017): Drain, online log parsing with a fixed-depth tree; the reference parser.)
- paper: https://doi.org/10.1145/1557019.1557154 (Makanju, Zincir-Heywood and Milios (KDD 2009): IPLoM, clustering event logs by iterative partitioning.)
- paper: https://doi.org/10.1109/ICSE-SEIP.2019.00021 (Zhu et al. (ICSE-SEIP 2019): the Loghub-2k benchmark (16 systems x 2,000 lines) and the grouping-accuracy metric.)
- paper: https://doi.org/10.1145/3510003.3510101 (Khan, Shin, Bianculli and Briand (ICSE 2022): corrected Loghub-2k ground truth, used here.)
- repository: https://github.com/logpai/logparser/tree/d9d4180784cde9afef990eeeb458591011933f9b/data (Pinned logparser repository: data/loghub_2k_corrected (Loghub licence, research or academic use with attribution) and the Drain implementation and evaluator used for the reference check.)

## Limitations

- Grouping accuracy is the classic metric; later work also reports parsing accuracy of the template strings and template-level F1 (Jiang et al., ISSTA 2024), which are not scored here.
- Each data set has only 2,000 messages, and newer, larger annotations (Loghub-2.0) show lower accuracies for the same parsers; results here say little about scale.
- The candidate gets message contents with the headers already removed by the benchmark's log formats, so header parsing is not part of the task; per-data-set settings, as in the published benchmarks, are allowed.
- Reference values computed with this evaluator: logparser's Drain with its benchmark settings scores 0.8605 mean grouping accuracy on the corrected labels (0.8654 on the original labels, as published); the digit-masking seed scores 0.7937.
- The logs and their labels are public; embedding labels is outside the contract but cannot be checked automatically. The Loghub licence permits research or academic use with attribution only.
