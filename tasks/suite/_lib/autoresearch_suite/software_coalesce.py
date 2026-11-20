"""Optimal register coalescing: the Appel-George Optimal Coalescing Challenge (474 interference graphs).

Each instance is an interference graph from the SML/NJ compiler after optimal
spilling (Appel and George, PLDI 2001): N nodes, K = 6 registers, nodes
0..K-1 are machine registers precoloured with their own index, and every
other node interferes with fewer than K nodes, so a K-colouring always exists.
Weighted affinity (move) pairs should receive the same colour; the cost of a
colouring is the total weight of affinity pairs with different colours. The
score of a task is the mean, over its instances, of cost / W where W is the
total affinity weight of the instance (the share of move weight that is not
coalesced); lower is better.

Preparation parses the pinned archives, merges duplicate pairs (cost-neutral),
drops self-affinities (always satisfied) and recomputes, with the checker
below, the costs of the published solution sets: Grund and Hack's ILP
solutions (CPLEX optimal for all but instances 139, 304 and 387) and the
iterated-register-coalescing and optimistic-coalescing solutions of George.
"""
from __future__ import annotations

import io
import json
import re
import statistics
import tarfile
from pathlib import Path

import numpy as np

from . import software_data, software_sandbox
from .runtime import CandidateError

SOURCES = ("coalesce-graphs", "coalesce-grund", "coalesce-greedy", "coalesce-optimistic")
LARGE = 3000  # instances with at least this many nodes form the "large" task
GROUPS = ("small", "large")
DRIVER = '''

def _artask_coalesce_all(problems):
    return [coalesce(problem) for problem in problems]
'''


def _numbers(text):
    return [int(x) for x in re.sub(r"%[^\n]*", " ", text).split()]


def _members(payload):
    """{basename: text} of the regular files of a tar archive (bz2 or plain)."""
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
        return {m.name.rsplit("/", 1)[-1]: archive.extractfile(m).read().decode("ascii")
                for m in archive.getmembers() if m.isfile()}


def parse_problem(text, name):
    values = _numbers(text)
    nodes, registers = values[0], values[1]
    if (len(values) - 2) % 3:
        raise ValueError(f"{name}: incomplete triple")
    interference, affinity = set(), {}
    for i in range(2, len(values), 3):
        a, b, w = values[i:i + 3]
        if not (0 <= a < nodes and 0 <= b < nodes):
            raise ValueError(f"{name}: node out of range")
        key = (min(a, b), max(a, b))
        if w < 0:
            interference.add(key)
        elif a != b:
            affinity[key] = affinity.get(key, 0) + w
    degree = [0] * nodes
    for a, b in interference:
        degree[a] += 1
        degree[b] += 1
    if any(degree[v] >= registers for v in range(registers, nodes)) or any(a == b for a, b in interference):
        raise ValueError(f"{name}: violates the challenge's degree condition")
    return {"name": name, "nodes": nodes, "registers": registers, "interference": [list(e) for e in sorted(interference)],
            "affinities": [[a, b, w] for (a, b), w in sorted(affinity.items())]}


def parse_solution(text, problem):
    values = _numbers(text)
    if values[:2] != [problem["nodes"], problem["registers"]]:
        raise ValueError(f"{problem['name']}: solution header does not match")
    return values[2:]


def cost(problem, colours):
    """Validated cost of a colouring (raises CandidateError when it is not a legal colouring)."""
    n, k = problem["nodes"], problem["registers"]
    if not isinstance(colours, list) or len(colours) != n:
        raise CandidateError(f"instance {problem['name']}: expected a list of {n} colours")
    if not all(type(c) is int for c in colours):
        raise CandidateError(f"instance {problem['name']}: colours must be Python ints")
    col = np.asarray(colours, dtype=np.int64)
    if col.min(initial=0) < 0 or col.max(initial=0) >= k:
        raise CandidateError(f"instance {problem['name']}: colours must lie in 0..{k - 1}")
    if not np.array_equal(col[:k], np.arange(k)):
        raise CandidateError(f"instance {problem['name']}: precoloured node i < {k} must have colour i")
    edges = np.asarray(problem["interference"], dtype=np.int64).reshape(-1, 2)
    clash = np.flatnonzero(col[edges[:, 0]] == col[edges[:, 1]])
    if clash.size:
        a, b = edges[clash[0]]
        raise CandidateError(f"instance {problem['name']}: interfering nodes {int(a)} and {int(b)} share a colour")
    pairs = np.asarray(problem["affinities"], dtype=np.int64).reshape(-1, 3)
    return int(pairs[col[pairs[:, 0]] != col[pairs[:, 1]], 2].sum())


def derive(raw):
    graphs = _members(raw["coalesce-graphs"])
    grund = _members(raw["coalesce-grund"])
    greedy, optimistic = _members(raw["coalesce-greedy"]), _members(raw["coalesce-optimistic"])
    groups = {g: [] for g in GROUPS}
    for name in sorted(graphs):
        problem = parse_problem(graphs[name], name)
        status = grund[name + ".asl"].splitlines()[1].split(";")[0].strip("% ").strip()
        best = cost(problem, parse_solution(grund[name + ".asl"], problem))
        checked = int(re.search(r"Cost = (\d+)", grund[name + ".chk"]).group(1))
        if best != checked:
            raise RuntimeError(f"{name}: recomputed cost {best} differs from the published check {checked}")
        problem.update(weight=sum(w for _, _, w in problem["affinities"]), best_cost=best, best_status=status,
                       published={"iterated_register_coalescing": cost(problem, parse_solution(greedy[name], problem)),
                                  "optimistic_coalescing": cost(problem, parse_solution(optimistic[name], problem))})
        groups["large" if problem["nodes"] >= LARGE else "small"].append(problem)
    return {f"coalesce/{g}.json": json.dumps(groups[g], sort_keys=True, separators=(",", ":")).encode() for g in GROUPS}


def public_problem(problem):
    return {k: problem[k] for k in ("name", "nodes", "registers", "interference", "affinities")}


def coalesce_score(module, group, *, cpu_seconds=120):
    """Run coalesce(problem) on every instance of the group in one sandboxed child; mean uncoalesced weight share."""
    path = software_sandbox.require_entry(module, "coalesce")
    problems = software_data.load_json(f"coalesce/{group}.json")
    source = Path(path).read_text(encoding="utf-8") + DRIVER
    try:
        colourings = software_sandbox.execute(source, "_artask_coalesce_all", [[public_problem(p) for p in problems]],
                                              cpu_seconds=cpu_seconds, memory_mb=3072, filename="candidate.py")
    except software_sandbox.SandboxFailure as failure:
        raise CandidateError(f"candidate coalesce() failed in the sandbox ({failure.kind}): {failure}"[:2000]) from None
    if not isinstance(colourings, list) or len(colourings) != len(problems):
        raise CandidateError("coalesce() must return one colouring per instance")
    shares, best, optimal, total = [], [], 0, 0
    published = {k: [] for k in problems[0]["published"]}
    for problem, colours in zip(problems, colourings):
        value = cost(problem, colours)
        total += value
        optimal += value <= problem["best_cost"]
        shares.append(value / problem["weight"])
        best.append(problem["best_cost"] / problem["weight"])
        for k in published:
            published[k].append(problem["published"][k] / problem["weight"])
    score = statistics.fmean(shares)
    return score, {"instances": len(problems), "total_cost": total, "instances_at_best_known": optimal,
                   "best_known_score": statistics.fmean(best),
                   "best_known_proven_optimal": all(p["best_status"] == "optimal" for p in problems),
                   "published_heuristic_scores": {k: statistics.fmean(v) for k, v in published.items()}}
