"""E-graph extraction minimising DAG cost (package or_csp).

Proxy for the XLA / Enzyme-JAX e-graph extraction task of the Magellan paper,
whose e-graphs are not public: the e-graphs come from extraction-gym
(github.com/egraphs-good/extraction-gym, commit 903ba0f818b50608fe20ae9e0f03c35cb27bc50a,
MIT), data/tensat - Tensat/TASO tensor-graph rewriting of five DNN models, in the
egraph-serialize JSON format (nodes keyed by id with op, cost, eclass and
children node ids; root_eclasses).

``check_extraction`` and ``dag_cost`` port ``ExtractionResult::check`` and
``ExtractionResult::dag_cost`` (src/extract/mod.rs): every root e-class is
chosen, a chosen node belongs to its e-class, every e-class reachable from the
roots through chosen nodes is chosen, the chosen graph is acyclic, and the DAG
cost sums the chosen node's cost once per reachable e-class (summed in the same
order as the Rust code).
"""
from __future__ import annotations

import gzip
import json
import math
import time
from pathlib import Path

from arena_task_paths import data_root
from .runtime import CandidateError

DATA = data_root() / "or_csp" / "egraph"


def load_egraph(name, data_dir=None):
    path = Path(data_dir or DATA) / f"tensat-{name}.json.gz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run scripts/prepare_or_csp.py")
    with gzip.open(path, "rb") as stream:
        return json.loads(stream.read())


def plain_choices(choices):
    """A built-in dict of built-in strings with the content of ``choices``, or CandidateError.

    The extraction is checked and then costed; both must read the same mapping. A dict subclass can answer
    ``choices[c]`` differently on every call and a str subclass can compare equal to any node id, so the content is
    copied once with the base classes' own methods.
    """
    if not isinstance(choices, dict):
        raise CandidateError("extract must return a dict mapping e-class id strings to e-node id strings")
    plain = {}
    for key, value in dict.items(choices):
        if not (isinstance(key, str) and isinstance(value, str)):
            raise CandidateError("extract must return a dict mapping e-class id strings to e-node id strings")
        plain[str.__str__(key)] = str.__str__(value)
    return plain


def check_extraction(egraph, choices):
    """Raise CandidateError unless ``choices`` (e-class id -> e-node id) is a valid extraction."""
    choices = plain_choices(choices)
    nodes, roots = egraph["nodes"], egraph["root_eclasses"]
    for class_id, node_id in choices.items():
        if node_id not in nodes or nodes[node_id]["eclass"] != class_id:
            raise CandidateError(f"e-node {node_id!r} is not in e-class {class_id!r}")
    status = {}
    for root in roots:
        if root not in choices:
            raise CandidateError(f"root e-class {root!r} has no chosen e-node")
        if root in status:
            continue
        status[root] = "open"
        stack = [(root, iter(nodes[choices[root]]["children"]))]
        while stack:  # iterative DFS; reaching an open e-class closes a cycle
            class_id, children = stack[-1]
            for child in children:
                child_class = nodes[child]["eclass"]
                seen = status.get(child_class)
                if seen == "open":
                    raise CandidateError(f"the chosen e-nodes form a cycle through e-class {child_class!r}")
                if seen is None:
                    if child_class not in choices:
                        raise CandidateError(f"e-class {child_class!r} is reachable but has no chosen e-node")
                    status[child_class] = "open"
                    stack.append((child_class, iter(nodes[choices[child_class]]["children"])))
                    break
            else:
                status[class_id] = "done"
                stack.pop()
    return len(status)


def dag_cost(egraph, choices):
    """Cost of the extracted DAG: each reachable e-class's chosen e-node counted once."""
    nodes = egraph["nodes"]
    costs = {}
    todo = list(egraph["root_eclasses"])
    while todo:
        class_id = todo.pop()
        node = nodes[choices[class_id]]
        if class_id in costs:
            continue
        costs[class_id] = node["cost"]
        todo.extend(nodes[child]["eclass"] for child in node["children"])
    total = 0.0
    for value in costs.values():  # sequential, like Rust's Iterator::sum (Python's sum() compensates)
        total += value
    return total


def extraction_score(module, params, data_dir=None):
    """Geometric mean over the e-graphs of candidate DAG cost / reference DAG cost (minimise)."""
    if not callable(getattr(module, "extract", None)):
        raise CandidateError("candidate must define extract(egraph)")
    rows, log_sum = [], 0.0
    for name in params["graphs"]:
        reference = float(params["reference_dag_cost"][name])
        egraph = load_egraph(name, data_dir)  # the candidate may mutate its copy; scoring uses a fresh load
        started = time.perf_counter()
        choices = module.extract(egraph)
        elapsed = time.perf_counter() - started
        if elapsed > float(params["time_limit_s"]):
            raise CandidateError(f"extraction of {name} took {elapsed:.1f} s > {params['time_limit_s']} s")
        pristine = load_egraph(name, data_dir)
        choices = plain_choices(choices)
        reachable = check_extraction(pristine, choices)
        cost = dag_cost(pristine, choices)
        if not math.isfinite(cost) or cost <= 0:
            raise CandidateError(f"non-positive or non-finite DAG cost on {name}")
        ratio = cost / reference
        log_sum += math.log(ratio)
        rows.append({"graph": name, "dag_cost": cost, "reference_dag_cost": reference, "ratio": ratio,
                     "bottom_up_dag_cost": params.get("bottom_up_dag_cost", {}).get(name),
                     "reachable_eclasses": reachable, "seconds": round(elapsed, 3)})
    score = math.exp(log_sum / len(rows))
    return score, {"graphs": rows, "reference": params.get("reference_name", "reference extractor")}
