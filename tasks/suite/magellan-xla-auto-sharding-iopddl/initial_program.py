"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

import numpy as np


def solve(problem, time_limit_s):
    """Greedy start, then sweeps in which every node takes its best strategy given all neighbours
    (node cost + incident edge costs), keeping the memory limit; stops at the time limit."""
    start = time.time()
    nodes, edges = problem["nodes"], problem["edges"]
    costs, usages, intervals = nodes["costs"], nodes["usages"], nodes["intervals"]
    edge_costs, n = edges["costs"], len(costs)
    limit = problem.get("usage_limit")
    incident = [[] for _ in range(n)]
    for e, (u, v) in enumerate(edges["nodes"]):
        incident[u].append((e, v, True))
        incident[v].append((e, u, False))

    def local_cost(i, s, choice):
        total = costs[i][s]
        for e, other, first in incident[i]:
            if choice[other] >= 0:
                total += edge_costs[e][s * len(costs[other]) + choice[other]] if first else edge_costs[e][choice[other] * len(costs[i]) + s]
        return total

    choice = [-1] * n
    for i in range(n):
        choice[i] = min(range(len(costs[i])), key=lambda s: local_cost(i, s, choice))
    horizon = max(b for a, b in intervals) + 1
    level = np.zeros(horizon + 1, dtype=np.int64)
    for i in range(n):
        a, b = intervals[i]
        level[a:b] += usages[i][choice[i]]

    def fits(i, s):
        a, b = intervals[i]
        extra = usages[i][s] - usages[i][choice[i]]
        return b <= a or extra <= 0 or limit is None or int(level[a:b].max()) + extra <= limit

    improved = True
    while improved and time.time() - start < 0.8 * time_limit_s:
        improved = False
        for i in range(n):
            if time.time() - start > 0.8 * time_limit_s:
                break
            current = local_cost(i, choice[i], choice)
            best_s, best = choice[i], current
            for s in range(len(costs[i])):
                if s != choice[i]:
                    value = local_cost(i, s, choice)
                    if value < best and fits(i, s):
                        best_s, best = s, value
            if best_s != choice[i]:
                a, b = intervals[i]
                level[a:b] += usages[i][best_s] - usages[i][choice[i]]
                choice[i], improved = best_s, True
    while limit is not None and int(level.max()) > limit:  # repair: drop the largest avoidable usage at the peak
        t = int(level.argmax())
        live = [i for i in range(n) if intervals[i][0] <= t < intervals[i][1] and usages[i][choice[i]] > min(usages[i])]
        if not live:
            break
        i = max(live, key=lambda j: usages[j][choice[j]] - min(usages[j]))
        s = min(range(len(usages[i])), key=lambda k: (usages[i][k], costs[i][k]))
        a, b = intervals[i]
        level[a:b] += usages[i][s] - usages[i][choice[i]]
        choice[i] = s
    return choice
# EVOLVE-BLOCK-END
