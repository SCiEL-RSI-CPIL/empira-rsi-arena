"""Log parsing (message template identification) on the 16 Loghub-2k data sets.

Each data set has 2,000 log messages from one system (HDFS, Hadoop, Spark,
...), sampled by Zhu et al. (ICSE-SEIP 2019) from Loghub, with ground-truth
event ids as corrected by Khan et al. (ICSE 2022) (logparser repository,
data/loghub_2k_corrected). The candidate receives only the message contents
(the header fields removed by the benchmark's log formats) and returns one
group label per message. Grouping accuracy (GA, the logparser evaluator's
"accuracy"): a message counts when its predicted group contains exactly the
messages of its true event, no more and no fewer. The score is the mean GA
over the 16 data sets.
"""
from __future__ import annotations

import collections
import csv
import io
import json
import statistics
from pathlib import Path

from . import software_data, software_sandbox
from .runtime import CandidateError

DATASETS = ("Android", "Apache", "BGL", "HDFS", "HPC", "Hadoop", "HealthApp", "Linux", "Mac", "OpenSSH", "OpenStack",
            "Proxifier", "Spark", "Thunderbird", "Windows", "Zookeeper")
SOURCES = tuple(f"loghub2k-{name}" for name in DATASETS)
DRIVER = '''

def _artask_parse_all(datasets):
    return [parse(dataset) for dataset in datasets]
'''


def derive(raw):
    datasets = []
    for name in DATASETS:
        rows = list(csv.DictReader(io.StringIO(raw[f"loghub2k-{name}"].decode("utf-8"))))
        if len(rows) != 2000 or [int(r["LineId"]) for r in rows] != list(range(1, 2001)):
            raise ValueError(f"{name}: expected 2,000 consecutive lines")
        datasets.append({"name": name, "messages": [r["Content"] for r in rows], "events": [r["EventId"] for r in rows]})
    return {"logparse/datasets.json": json.dumps(datasets, ensure_ascii=False, separators=(",", ":")).encode("utf-8")}


def grouping_accuracy(truth, predicted):
    """Share of messages whose predicted group equals their true event's message set (logparser's accuracy)."""
    true_size = collections.Counter(truth)
    members = collections.defaultdict(list)
    for index, label in enumerate(predicted):
        members[label].append(index)
    correct = 0
    for indices in members.values():
        events = {truth[i] for i in indices}
        if len(events) == 1 and true_size[next(iter(events))] == len(indices):
            correct += len(indices)
    return correct / len(truth)


def pairwise_f1(truth, predicted):
    """F-measure over pairs of messages placed in the same group (logparser's F1_measure)."""
    def pairs(labels):
        return sum(n * (n - 1) // 2 for n in collections.Counter(labels).values())
    true_pairs, predicted_pairs = pairs(truth), pairs(predicted)
    both = pairs(list(zip(truth, predicted)))
    if not true_pairs or not predicted_pairs or not both:
        return 0.0
    precision, recall = both / predicted_pairs, both / true_pairs
    return 2 * precision * recall / (precision + recall)


def logparse_score(module, *, cpu_seconds=120):
    path = software_sandbox.require_entry(module, "parse")
    datasets = software_data.load_json("logparse/datasets.json")
    source = Path(path).read_text(encoding="utf-8") + DRIVER
    public = [{"name": d["name"], "messages": list(d["messages"])} for d in datasets]
    try:
        answers = software_sandbox.execute(source, "_artask_parse_all", [public], cpu_seconds=cpu_seconds, memory_mb=2048,
                                           filename="candidate.py")
    except software_sandbox.SandboxFailure as failure:
        raise CandidateError(f"candidate parse() failed in the sandbox ({failure.kind}): {failure}"[:2000]) from None
    if not isinstance(answers, list) or len(answers) != len(datasets):
        raise CandidateError("parse() must return one label list per data set")
    accuracy, f1, groups = {}, {}, {}
    for dataset, labels in zip(datasets, answers):
        if not isinstance(labels, list) or len(labels) != len(dataset["messages"]) or not all(
                type(x) in (int, str) for x in labels):
            raise CandidateError(f"parse() must return {len(dataset['messages'])} int or str labels for {dataset['name']}")
        accuracy[dataset["name"]] = grouping_accuracy(dataset["events"], labels)
        f1[dataset["name"]] = pairwise_f1(dataset["events"], labels)
        groups[dataset["name"]] = [len(set(labels)), len(set(dataset["events"]))]
    return statistics.fmean(accuracy.values()), {"datasets": len(datasets), "grouping_accuracy": accuracy,
                                                 "pairwise_f1": f1, "mean_pairwise_f1": statistics.fmean(f1.values()),
                                                 "groups_predicted_and_true": groups}
