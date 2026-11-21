"""Capacitated vehicle routing on CVRPLIB X and XL instances, scored exactly.

The candidate returns a list of routes (lists of customer ids, customers are 1..n, the depot is 0 and is not listed).
Every customer must appear exactly once and no route may exceed the capacity. Cost: sum of Euclidean distances
rounded to the nearest integer per edge (the CVRPLIB convention), depot to depot per route. The number of routes is free.
"""
from __future__ import annotations

import numpy as np

from .common import CandidateError, read_data, require


def instance(name):
    text = read_data("cvrp", name + ".vrp").decode("ascii", "replace")
    lines = [l.strip() for l in text.splitlines()]
    head = {}
    for l in lines:
        if ":" in l and not l[:1].isdigit():
            key, _, val = l.partition(":")
            head[key.strip().upper()] = val.strip()
    capacity = int(head["CAPACITY"])
    dim = int(head["DIMENSION"])
    i = lines.index("NODE_COORD_SECTION")
    coords = np.array([[float(x) for x in lines[i + 1 + k].split()[1:3]] for k in range(dim)])
    j = lines.index("DEMAND_SECTION")
    demand = np.array([int(lines[j + 1 + k].split()[1]) for k in range(dim)], dtype=np.int64)
    return {"name": name, "capacity": capacity, "coords": coords, "demand": demand, "n": dim - 1}


def route_cost(routes, inst):
    n, cap, xy, dem = inst["n"], inst["capacity"], inst["coords"], inst["demand"]
    try:
        routes = [[int(c) for c in r] for r in routes]
    except (TypeError, ValueError):
        raise CandidateError("return a list of routes, each a list of customer ids") from None
    seen = np.zeros(n + 1, dtype=np.int64)
    total = 0
    for r in routes:
        if not r:
            continue
        require(all(1 <= c <= n for c in r), f"customer ids must be in 1..{n}")
        require(int(dem[r].sum()) <= cap, "a route exceeds the vehicle capacity")
        for c in r:
            seen[c] += 1
        path = np.array([0] + r + [0])
        d = xy[path[1:]] - xy[path[:-1]]
        total += int(np.rint(np.sqrt((d * d).sum(axis=1))).sum())
    require(bool((seen[1:] == 1).all()), "every customer must be visited exactly once")
    return total, {"cost": total, "routes": sum(1 for r in routes if r), "customers": n, "verification": "exact_rounded_distance"}
