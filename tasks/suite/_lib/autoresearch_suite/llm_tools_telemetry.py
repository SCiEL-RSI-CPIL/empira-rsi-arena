"""ADRS telemetry repair (Abilene interface counters) on the pre-release ADRS evaluator's protocol.

Port of the ``evaluate_legacy`` path of ``openevolve/examples/ADRS/telemetry_repair/
evaluator.py`` in UCB-ADRS/ADRS at commit 259292f3 (Apache-2.0): 30 rows of the
Abilene snapshot ``evaluation_data.csv`` are drawn with ``random.seed(42)``; for
every row the interfaces (rx from ``*_ingress_from_*``, tx from ``*_egress_to_*``
columns) receive +-2 % noise and one randomly chosen bug type (dropped, scaled,
correlated dropped, correlated scaled counters); the candidate's
``run_repair(telemetry, topology)`` returns ``(original, repaired, confidence)``
triples and the row score is ``0.75 * counter repair + 0.05 * status repair +
0.2 * confidence calibration``.  The perturbation draws use one
``random.Random(42)`` stream in the original order, so the scenarios are
identical to the original's for every candidate that does not itself draw from
Python's global random generator inside the original evaluator's process.

Deviations (stated in the task notes): the candidate runs in a sandboxed child
that cannot read the data files; the ground-truth fields that the original
leaves in correlated-bug interfaces are removed before the candidate sees
them; and the output must contain every interface with well-formed triples
(the original silently skipped missing interfaces, which let a candidate score
1.0 by returning a single interface).
"""
from __future__ import annotations

import ast
import csv
import math
import random

from . import llm_tools_common as common
from .runtime import CandidateError

CSV_NAME = "telemetry/evaluation_data.csv"
WEIGHTS = (0.75, 0.05, 0.2)


def load_rows():
    with common.data_file(CSV_NAME).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_interfaces(row):
    """``convert_csv_row_to_interfaces`` of the original: interface id -> telemetry with ground truth."""
    interfaces = {}
    for column, cell in row.items():
        if not (column.startswith("low_") and ("_egress_to_" in column or "_ingress_from_" in column)):
            continue
        try:
            data = ast.literal_eval(cell)
            if "ground_truth" not in data:
                continue
            observed = data["perturbed"] if data["perturbed"] is not None else data["ground_truth"]
            if "_egress_to_" in column:
                parts = column.replace("low_", "").split("_egress_to_")
                if len(parts) != 2:
                    continue
                local, remote = parts
                key, rate, truth = f"{local}_to_{remote}", "tx_rate", "_ground_truth_tx"
            else:
                parts = column.replace("low_", "").split("_ingress_from_")
                if len(parts) != 2:
                    continue
                local, remote = parts
                key, rate, truth = f"{local}_to_{remote}", "rx_rate", "_ground_truth_rx"
            entry = interfaces.setdefault(key, {"interface_status": "up", "rx_rate": 0.0, "tx_rate": 0.0,
                                                "connected_to": f"{remote}_to_{local}", "local_router": local, "remote_router": remote})
            entry[rate] = float(observed) if observed is not None else 0.0
            entry[truth] = float(data["ground_truth"])
        except (ValueError, SyntaxError, KeyError):
            continue
    return interfaces


def router_topology(interfaces):
    topology = {}
    for interface_id, telemetry in interfaces.items():
        topology.setdefault(telemetry.get("local_router"), []).append(interface_id)
    return topology


def perturb(interfaces, topology, rng):
    """``apply_test_perturbations`` of the original, drawing from ``rng`` in the same order."""
    perturbed, truth = {}, {}
    bug = rng.choice(["dropped", "scaled", "correlated_dropped", "correlated_scaled"])
    for interface_id, data in interfaces.items():
        truth[interface_id] = {"rx_rate": data.get("_ground_truth_rx", data["rx_rate"]),
                               "tx_rate": data.get("_ground_truth_tx", data["tx_rate"]),
                               "interface_status": data["interface_status"]}
        noisy = data.copy()
        noisy["rx_rate"] *= rng.uniform(0.98, 1.02)
        noisy["tx_rate"] *= rng.uniform(0.98, 1.02)
        if bug == "dropped":
            if rng.random() < 0.20:
                noisy["rx_rate"] = 0
                noisy["tx_rate"] = 0
        elif bug == "scaled":
            if rng.random() < 0.20:
                noisy["rx_rate"] *= rng.choice([rng.uniform(0.5, 0.8), rng.uniform(1.2, 1.5)])
                noisy["tx_rate"] *= rng.choice([rng.uniform(0.5, 0.8), rng.uniform(1.2, 1.5)])
        noisy.pop("_ground_truth_rx", None)
        noisy.pop("_ground_truth_tx", None)
        perturbed[interface_id] = noisy
    for _, members in topology.items():
        if rng.random() < 0.80:
            continue
        if bug == "correlated_dropped":
            for interface_id in members:
                bugged = interfaces[interface_id].copy()
                bugged["rx_rate"] = 0
                bugged["tx_rate"] = 0
                perturbed[interface_id] = bugged
        elif bug == "correlated_scaled":
            for interface_id in members:
                bugged = interfaces[interface_id].copy()
                bugged["rx_rate"] *= rng.choice([rng.uniform(0.5, 0.8), rng.uniform(1.2, 1.5)])
                bugged["tx_rate"] *= rng.choice([rng.uniform(0.5, 0.8), rng.uniform(1.2, 1.5)])
                perturbed[interface_id] = bugged
    return perturbed, truth, bug


def scenarios(rows, seed=42, count=30):
    """The original's 30 test rows and perturbations; ground-truth fields are stripped from the candidate's view."""
    rng = random.Random(seed)
    chosen = rng.sample(rows, min(count, len(rows)))
    out = []
    for row in chosen:
        interfaces = row_interfaces(row)
        if not interfaces:
            continue
        topology = router_topology(interfaces)
        perturbed, truth, bug = perturb(interfaces, topology, rng)
        visible = {key: {name: value for name, value in data.items() if not name.startswith("_ground_truth")}
                   for key, data in perturbed.items()}
        out.append({"timestamp": row.get("timestamp"), "bug": bug, "telemetry": visible, "topology": topology, "truth": truth})
    return out


def _finite(number):
    """Whether an int or float is a finite float: an integer beyond the float range is not."""
    try:
        return math.isfinite(float(number))
    except OverflowError:
        return False


def _triple(value, where, numeric):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise CandidateError(f"{where} must be an (original, repaired, confidence) triple")
    repaired, confidence = value[1], value[2]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise CandidateError(f"{where}: confidence must be a number in [0, 1]")
    if numeric:
        if isinstance(repaired, bool) or not isinstance(repaired, (int, float)) or not _finite(repaired):
            raise CandidateError(f"{where}: repaired value must be a finite number")
        return float(repaired), float(confidence)
    if not isinstance(repaired, str):
        raise CandidateError(f"{where}: repaired status must be a string")
    return repaired, float(confidence)


def validate(output, truth, index):
    """Strict contract: every interface present with rx/tx/status triples."""
    if not isinstance(output, dict):
        raise CandidateError(f"scenario {index}: run_repair must return a dict keyed by interface id")
    checked = {}
    for interface_id in truth:
        entry = output.get(interface_id)
        if not isinstance(entry, dict):
            raise CandidateError(f"scenario {index}: interface {interface_id} is missing from the repair output")
        checked[interface_id] = {"rx_rate": _triple(entry.get("rx_rate"), f"{interface_id}.rx_rate", True),
                                 "tx_rate": _triple(entry.get("tx_rate"), f"{interface_id}.tx_rate", True),
                                 "interface_status": _triple(entry.get("interface_status"), f"{interface_id}.interface_status", False)}
    return checked


def counter_repair(checked, truth):
    """``calculate_counter_repair_quality``: 1 - mean capped relative error over nonzero ground truth."""
    total, count = 0.0, 0
    for interface_id, gold in truth.items():
        for rate in ("rx_rate", "tx_rate"):
            if gold[rate] > 0:
                total += min(abs(checked[interface_id][rate][0] - gold[rate]) / gold[rate], 1.0)
                count += 1
    return max(0.0, 1.0 - total / count) if count else 0.0


def status_repair(checked, truth):
    correct = sum(checked[i]["interface_status"][0] == gold["interface_status"] for i, gold in truth.items())
    return correct / len(truth) if truth else 1.0


def calibration(checked, truth):
    """``calculate_confidence_calibration``: rewards confident good repairs and unconfident poor ones."""
    total, count = 0.0, 0
    for interface_id, gold in truth.items():
        for rate in ("rx_rate", "tx_rate"):
            repaired, confidence = checked[interface_id][rate]
            accuracy = 1.0 - abs(gold[rate] - repaired) / max(abs(gold[rate]), abs(repaired), 1.0)
            if accuracy > 0.8:
                score = 1.0 if confidence > 0.7 else 0.5 + (confidence - 0.3) * 1.25
            elif confidence < 0.3:
                score = 0.8
            else:
                score = max(0.0, 0.8 - confidence * (1.0 - accuracy) * 2.0)
            total += max(0.0, min(1.0, score))
            count += 1
    return total / count if count else 1.0


def score_outputs(cases, outputs):
    if not isinstance(outputs, list) or len(outputs) != len(cases):
        raise CandidateError("the sandbox returned an unexpected number of repair outputs")
    rows, parts = [], []
    for index, (case, output) in enumerate(zip(cases, outputs)):
        checked = validate(output, case["truth"], index)
        part = (counter_repair(checked, case["truth"]), status_repair(checked, case["truth"]), calibration(checked, case["truth"]))
        parts.append(part)
        rows.append({"timestamp": case["timestamp"], "bug": case["bug"], "score": round(sum(w * p for w, p in zip(WEIGHTS, part)), 6)})
    means = [sum(p[k] for p in parts) / len(parts) for k in range(3)]
    score = sum(sum(w * p for w, p in zip(WEIGHTS, part)) for part in parts) / len(parts)
    return score, {"num_tests": len(parts), "counter_repair_accuracy": round(means[0], 6),
                   "status_repair_accuracy": round(means[1], 6), "confidence_calibration": round(means[2], 6), "per_row": rows}


def evaluate(module, P):
    source = common.candidate_source(module, "run_repair")
    cases = scenarios(load_rows(), seed=int(P.get("seed", 42)), count=int(P.get("rows", 30)))
    args = [[case["telemetry"], case["topology"]] for case in cases]
    ok, outputs, failure = common.call_candidate(source, "run_repair", args, loop=True, cpu_seconds=int(P.get("cpu_seconds", 60)),
                                                 wall_seconds=int(P.get("wall_seconds", 120)), memory_mb=2048)
    if not ok:
        raise CandidateError(f"run_repair failed in the sandbox ({failure['kind']}): {failure['message']}")
    return score_outputs(cases, outputs)
