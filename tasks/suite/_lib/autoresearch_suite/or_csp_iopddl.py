"""IOPDDL: intra-operator parallelism (auto-sharding) contest evaluator (package or_csp).

Problem and objective of the ASPLOS 2025 / EuroSys 2025 contest on intra-operator
parallelism for distributed deep learning (github.com/asplos-contest/2025,
IOPDDL.md): pick one sharding strategy per node so that the total of node costs
and edge (resharding) costs is minimal while the memory usage summed over the
nodes live at each time step never exceeds the usage limit. ``total_cost`` is
a port of ``iopddl::Evaluate`` from github.com/google/iopddl (iopddl.cc,
commit 8a61c4359c5a330a04c8ef1780402cc254d28f29, Apache-2.0): node intervals
are half-open, an edge's strategy index is the row-major index over its nodes'
strategies, and totals use exact integers (the official code uses int128).

Contest scoring: a benchmark earns ``best_submitted_cost / cost`` points, where
the best cost is the minimum over the teams in the contest's final results
(misc/iopddl_final_results.csv); an infeasible, empty or late answer earns 0.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from arena_task_paths import data_root
from .runtime import CandidateError

DATA = data_root() / "or_csp" / "iopddl"
# Minimum cost over the teams of the final contest results (asplos-contest/2025 at
# 06267509624cea22b98cd50427aaae6d01e3b6c8, misc/iopddl_final_results.csv).
BEST_TEAM_COST = {
    "A": 6548641531, "B": 532843, "C": 84102445260, "D": 313613524736, "E": 338965335230, "F": 364364563561,
    "G": 217039, "H": 527715404257, "I": 583706456444, "J": 1360981630371, "K": 2274276676, "L": 1970993742,
    "M": 49928460991, "N": 202739586553, "O": 222880199765, "P": 662197407330, "Q": 526776842042, "R": 371899551572,
    "S": 1534613444, "T": 682895254913, "U": 2569342178, "V": 1632051, "W": 11818845467155, "X": 280461191177,
    "Y": 621176568278,
}


def load_problem(benchmark, data_dir=None):
    path = Path(data_dir or DATA) / f"asplos-2025-iopddl-{benchmark}.json.gz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run scripts/prepare_or_csp.py")
    with gzip.open(path, "rb") as stream:
        return json.loads(stream.read())["problem"]


class Scorer:
    """Compact copy of a problem, built before the candidate sees the (mutable) problem dict."""

    def __init__(self, problem):
        nodes, edges = problem["nodes"], problem["edges"]
        self.sizes = np.array([len(c) for c in nodes["costs"]], dtype=np.int64)
        if any(len(c) != len(u) for c, u in zip(nodes["costs"], nodes["usages"])):
            raise ValueError("node costs and usages disagree")
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.node_costs = np.array([c for row in nodes["costs"] for c in row], dtype=np.int64)
        self.usages = np.array([u for row in nodes["usages"] for u in row], dtype=np.int64)
        self.intervals = np.array(nodes["intervals"], dtype=np.int64).reshape(-1, 2)
        self.horizon = int(self.intervals[:, 1].max()) if len(self.intervals) else 0
        self.edge_nodes = [list(e) for e in edges["nodes"]]
        lengths = np.array([len(c) for c in edges["costs"]], dtype=np.int64)
        self.edge_offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        self.edge_flat = np.fromiter((c for row in edges["costs"] for c in row), dtype=np.int64, count=int(lengths.sum()))
        expected = [int(np.prod([self.sizes[v] for v in e])) for e in self.edge_nodes]
        if expected != lengths.tolist():
            raise ValueError("edge cost tables do not match the strategy counts")
        self.binary = all(len(e) == 2 for e in self.edge_nodes)
        self.edge_array = np.array(self.edge_nodes, dtype=np.int64).reshape(-1, 2) if self.binary else None
        self.limit = problem.get("usage_limit")
        worst = int(sum(int(max(u)) for u in nodes["usages"] if u))
        if worst >= 2 ** 62:
            raise ValueError("usage totals could overflow int64")
        self.fingerprint = self._digest()

    def _digest(self):
        digest = hashlib.sha256()
        for array in (self.sizes, self.node_costs, self.usages, self.intervals, self.edge_offsets, self.edge_flat):
            digest.update(array.tobytes())
        return digest.hexdigest()

    def check_shape(self, solution):
        if not isinstance(solution, (list, tuple)) and not (isinstance(solution, np.ndarray) and solution.ndim == 1):
            raise CandidateError("solve must return a list of strategy indices")
        values = [x.item() if isinstance(x, np.generic) else x for x in solution]
        if any(isinstance(x, bool) or not isinstance(x, int) for x in values):
            raise CandidateError("strategy indices must be integers")
        # The built-in value of each index: a subclass of int could answer the range check below as it likes
        # while numpy reads the number it really holds (and with it another node's costs and usages).
        values = [x if type(x) is int else int.__int__(x) for x in values]
        if values and len(values) != len(self.sizes):
            raise CandidateError(f"incorrect solution size: {len(values)} indices for {len(self.sizes)} nodes")
        if values and any(x < 0 or x >= int(self.sizes[i]) for i, x in enumerate(values)):
            raise CandidateError("strategy index out of range")
        return values

    def total_cost(self, values):
        """(cost, peak usage) of a well-formed solution; mirrors iopddl::Evaluate."""
        choice = np.array(values, dtype=np.int64)
        flat = self.offsets + choice
        cost = sum(self.node_costs[flat].tolist())
        if self.binary:
            u, v = self.edge_array[:, 0], self.edge_array[:, 1]
            index = choice[u] * self.sizes[v] + choice[v]
        else:
            index = np.zeros(len(self.edge_nodes), dtype=np.int64)
            for e, nodes in enumerate(self.edge_nodes):
                for node in nodes:
                    index[e] = index[e] * int(self.sizes[node]) + values[node]
        cost += sum(self.edge_flat[self.edge_offsets + index].tolist())
        delta = np.zeros(self.horizon + 1, dtype=np.int64)
        live = self.intervals[:, 1] > self.intervals[:, 0]
        np.add.at(delta, self.intervals[live, 0], self.usages[flat][live])
        np.add.at(delta, self.intervals[live, 1], -self.usages[flat][live])
        peak = int(np.cumsum(delta)[:self.horizon].max()) if self.horizon else 0
        return cost, peak


def total_cost(problem, solution):
    """Exact cost of a solution, or raise ValueError like iopddl::Evaluate's error statuses."""
    scorer = Scorer(problem)
    try:
        values = scorer.check_shape(list(solution))
    except CandidateError as error:
        raise ValueError(str(error)) from error
    if not values and len(scorer.sizes):
        raise ValueError("incorrect solution size")
    cost, peak = scorer.total_cost(values)
    if scorer.limit is not None and peak > scorer.limit:
        raise ValueError("usage limit exceeded")
    return cost


def contest_points(module, params, data_dir=None):
    """Mean contest points over the benchmarks; returns (score, details)."""
    if not callable(getattr(module, "solve", None)):
        raise CandidateError("candidate must define solve(problem, time_limit_s)")
    limit_s, grace = float(params["time_limit_s"]), float(params.get("grace_s", 1.0))
    rows = []
    for benchmark in params["benchmarks"]:
        problem = load_problem(benchmark, data_dir)
        scorer = Scorer(problem)
        started = time.perf_counter()
        solution = module.solve(problem, limit_s)
        elapsed = time.perf_counter() - started
        values = scorer.check_shape(solution)
        if scorer._digest() != scorer.fingerprint:
            raise RuntimeError("evaluator copy of the problem changed during solve")
        row = {"benchmark": benchmark, "seconds": round(elapsed, 3), "best_submitted_cost": BEST_TEAM_COST[benchmark]}
        if elapsed > limit_s + grace:
            row.update(status="late", points=0.0)
        elif not values:
            row.update(status="no_solution", points=0.0)
        else:
            cost, peak = scorer.total_cost(values)
            row.update(cost=cost, peak_usage=peak, usage_limit=scorer.limit, penalised_terms=cost // 10 ** 18)
            if scorer.limit is not None and peak > scorer.limit:
                row.update(status="usage_limit_exceeded", points=0.0)
            else:
                row.update(status="feasible", points=BEST_TEAM_COST[benchmark] / cost)
        rows.append(row)
    score = sum(r["points"] for r in rows) / len(rows)
    return score, {"benchmarks": rows, "total_points": sum(r["points"] for r in rows),
                   "time_limit_s": limit_s, "feasible": sum(r["status"] == "feasible" for r in rows)}
