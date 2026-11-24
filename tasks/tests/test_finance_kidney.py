"""Kidney exchange: clearing verification, recorded optima and scoring."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
except ImportError:  # optional: the test that solves the pools independently is skipped
    milp = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_kidney as fk  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import have_data, slow  # noqa: E402

SLOW = os.environ.get("FINANCE_SLOW_TESTS") == "1"


def greedy(pool):
    edges = set(map(tuple, pool["edges"].tolist()))
    used, cycles = set(), []
    for a, b in sorted(edges):
        if a < b and (b, a) in edges and a not in used and b not in used:
            used.update((a, b))
            cycles.append([a, b])
    return cycles


def optimum(name):
    """Independent exact solve: enumerate all 2- and 3-cycles, maximize covered pairs with HiGHS."""
    pairs, edges = fk.pool(name)
    successors = [set() for _ in range(pairs)]
    for a, b in edges.tolist():
        successors[a].add(b)
    cycles = []
    for a in range(pairs):
        for b in successors[a]:
            if b > a:
                if a in successors[b]:
                    cycles.append((a, b))
                cycles += [(a, b, c) for c in successors[b] if c > a and c != b and a in successors[c]]
    rows = [v for cycle in cycles for v in cycle]
    cols = [j for j, cycle in enumerate(cycles) for _ in cycle]
    matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(pairs, len(cycles))).tocsr()
    result = milp(-np.array([len(c) for c in cycles], float), constraints=LinearConstraint(matrix, 0, 1),
                  integrality=np.ones(len(cycles)), bounds=Bounds(0, 1), options={"mip_rel_gap": 0})
    return int(round(-result.fun)), [list(cycles[j]) for j in np.where(result.x > 0.5)[0]]


@unittest.skipUnless(have_data("kidney_preflib_pools.csv"), "run scripts/prepare_finance.py first")
class KidneyTests(unittest.TestCase):
    def test_pools_match_their_headers(self):
        for name, pairs in fk.PAIRS.items():
            n, edges = fk.pool(name)
            self.assertEqual(n, pairs)
            self.assertTrue(((edges >= 0) & (edges < pairs)).all())
            self.assertFalse((edges[:, 0] == edges[:, 1]).any())

    def test_clearing_checks(self):
        name = "00036-00000151"
        cycles = greedy({"edges": fk.pool(name)[1]})
        self.assertEqual(fk.check_clearing(name, cycles)[0], 2 * len(cycles))
        a, b = cycles[0]
        missing = next((u, v) for u in range(256) for v in range(256)
                       if u != v and (u, v) not in set(map(tuple, fk.pool(name)[1].tolist())))
        for bad in ([[a, b], [b, a]], [[a, b, a]], [list(range(4))], [[a, 300]], [[float(a), b]], [list(missing)], "x", [[a]]):
            with self.assertRaises(CandidateError):
                fk.check_clearing(name, bad)

    # Measured at 226 s on one thread for the four 256-pair pools (49 to 65 s each): runs only with ARENA_SLOW_TESTS=1.
    # FINANCE_SLOW_TESTS=1 adds the four 512-pair pools.
    @slow
    @unittest.skipUnless(milp, "scipy is not installed")
    def test_recorded_optima(self):
        names = list(fk.PAIRS) if SLOW else [n for n, p in fk.PAIRS.items() if p == 256]
        for name in names:
            value, cycles = optimum(name)
            self.assertEqual(value, fk.OPTIMA[name], name)
            self.assertEqual(fk.check_clearing(name, cycles)[0], value)

    def test_seed_score_and_data_isolation(self):
        score, details = fk.kidney_score(greedy)
        self.assertAlmostEqual(score, 1674 / 1945)
        self.assertEqual(details["optimum_transplants"], 1945)

        def vandal(pool):
            pool["edges"][:] = 0
            return []
        self.assertEqual(fk.kidney_score(vandal)[0], 0.0)
        self.assertAlmostEqual(fk.kidney_score(greedy)[0], score)


if __name__ == "__main__":
    unittest.main()
