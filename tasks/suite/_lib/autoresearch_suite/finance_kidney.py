"""Kidney exchange clearing with 2- and 3-way cycles (Roth, Sonmez and Unver; Abraham, Blum and Sandholm).

Each pool is a directed compatibility graph over incompatible patient-donor
pairs: an edge a -> b means the donor of pair a can give to the patient of
pair b. A clearing is a set of vertex-disjoint directed cycles of length 2 or 3;
every pair on a cycle receives a kidney. The evaluator verifies every cycle
edge by edge and scores the number of transplants against the proven optimum.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import finance_data
from .runtime import CandidateError

TABLE = "kidney_preflib_pools.csv"
MAX_CYCLE = 3
# Proven optima (HiGHS on the cycle formulation, relative MIP gap 0); re-checked by tests/test_finance_kidney.py.
OPTIMA = {"00036-00000151": 166, "00036-00000152": 175, "00036-00000153": 158, "00036-00000154": 145,
          "00036-00000191": 351, "00036-00000192": 337, "00036-00000193": 300, "00036-00000194": 313}
PAIRS = {"00036-00000151": 256, "00036-00000152": 256, "00036-00000153": 256, "00036-00000154": 256,
         "00036-00000191": 512, "00036-00000192": 512, "00036-00000193": 512, "00036-00000194": 512}


@lru_cache(maxsize=None)
def pool(name):
    """(pairs, edges as an (E, 2) array of 0-based (donor_pair, patient_pair))."""
    keys, _, values = finance_data.table(TABLE)
    rows = np.array([k == name for k in keys])
    edges = values[rows].astype(np.int64) - 1
    edges.setflags(write=False)
    return PAIRS[name], edges


def check_clearing(name, cycles):
    """Validate a candidate clearing; return (transplants, counts of cycles by length)."""
    pairs, edges = pool(name)
    allowed = set(map(tuple, edges.tolist()))
    if not isinstance(cycles, (list, tuple)):
        raise CandidateError(f"{name}: select_exchanges must return a list of cycles")
    used, counts, transplants = set(), {2: 0, 3: 0}, 0
    for cycle in cycles:
        if not isinstance(cycle, (list, tuple, np.ndarray)):
            raise CandidateError(f"{name}: each cycle must be a sequence of pair indices")
        members = []
        for v in cycle:
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
                raise CandidateError(f"{name}: pair indices must be integers")
            members.append(int(v))
        if not 2 <= len(members) <= MAX_CYCLE:
            raise CandidateError(f"{name}: cycle {members} must have 2 to {MAX_CYCLE} pairs")
        if any(not 0 <= v < pairs for v in members) or len(set(members)) != len(members):
            raise CandidateError(f"{name}: cycle {members} has an invalid or repeated pair")
        for a, b in zip(members, members[1:] + members[:1]):
            if (a, b) not in allowed:
                raise CandidateError(f"{name}: donor of pair {a} is not compatible with the patient of pair {b}")
        if used.intersection(members):
            raise CandidateError(f"{name}: a pair appears in two cycles")
        used.update(members)
        counts[len(members)] += 1
        transplants += len(members)
    return transplants, counts


def kidney_score(select_exchanges):
    """Total transplants divided by the total proven optimum over the fixed pools (maximize; 1 is optimal)."""
    if not callable(select_exchanges):
        raise CandidateError("select_exchanges must be callable")
    rows, total, best = [], 0, 0
    for name in PAIRS:
        pairs, edges = pool(name)
        given = {"name": name, "pairs": pairs, "edges": edges.copy(), "max_cycle_length": MAX_CYCLE,
                 "time_limit_seconds": 60.0}
        transplants, counts = check_clearing(name, select_exchanges(given))
        total += transplants
        best += OPTIMA[name]
        rows.append({"pool": name, "pairs": pairs, "edges": int(len(edges)), "transplants": transplants,
                     "optimum": OPTIMA[name], "two_cycles": counts[2], "three_cycles": counts[3]})
    return total / best, {"pools": rows, "transplants": total, "optimum_transplants": best}
