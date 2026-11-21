"""Job-shop scheduling on fixed benchmark instances, scored by exact makespan.

Instance file: first line `n m`, then n lines of m (machine, duration) pairs. The candidate returns start times,
`starts[j][k]` for operation k of job j. Checks: nonnegative, operations of a job in order without overlap, no two
operations on one machine overlap. Score: the makespan, minimized.
"""
from __future__ import annotations

import numpy as np

from .common import CandidateError, read_data, require


def instance(name):
    lines = [l for l in read_data("jssp", name + ".txt").decode().splitlines() if l.strip()]
    n, m = (int(x) for x in lines[0].split()[:2])
    jobs = []
    for l in lines[1:1 + n]:
        v = [int(x) for x in l.split()]
        require(len(v) == 2 * m, f"{name}: a job line must hold {m} (machine, duration) pairs")
        jobs.append([(v[2 * k], v[2 * k + 1]) for k in range(m)])
    return {"name": name, "n_jobs": n, "n_machines": m, "jobs": jobs}


def makespan(starts, inst):
    jobs = inst["jobs"]
    try:
        given = [list(row) for row in starts]
        starts = [[int(x) for x in row] for row in given]
        exact = all(a == b for ra, rb in zip(given, starts) for a, b in zip(ra, rb))  # 3.9 must not be read as 3
    except (TypeError, ValueError):
        raise CandidateError("return start times: a list of n_jobs lists of integers") from None
    require(exact, "start times must be integers")
    require(len(starts) == len(jobs) and all(len(s) == len(j) for s, j in zip(starts, jobs)), "shape must be n_jobs x n_machines")
    per_machine = {}
    end_max = 0
    for j, (ops, st) in enumerate(zip(jobs, starts)):
        prev_end = 0
        for k, ((mach, dur), s) in enumerate(zip(ops, st)):
            require(s >= prev_end, f"job {j}: operation {k} starts before its predecessor ends")
            prev_end = s + dur
            per_machine.setdefault(mach, []).append((s, prev_end))
        end_max = max(end_max, prev_end)
    for mach, spans in per_machine.items():
        spans.sort()
        for a, b in zip(spans, spans[1:]):
            require(b[0] >= a[1], f"machine {mach}: two operations overlap")
    return end_max, {"makespan": end_max, "jobs": len(jobs), "machines": inst["n_machines"], "verification": "exact_schedule_check"}
