"""Maximum Nash welfare: instance stability, the exact program against brute force, EF1 and scoring."""
from __future__ import annotations

import itertools
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_fairdiv as fd  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import needs, slow  # noqa: E402

SLOW = os.environ.get("FINANCE_SLOW_TESTS") == "1"


def round_robin(values):
    n, m = values.shape
    owner, free, turn = [0] * m, set(range(m)), 0
    while free:
        good = max(free, key=lambda g: (values[turn % n, g], -g))
        owner[good] = turn % n
        free.remove(good)
        turn += 1
    return owner


class FairDivisionTests(unittest.TestCase):
    def test_instances_are_stable_point_splits(self):
        for index, (n, m, _) in enumerate(fd.SIZES):
            values = fd.valuations(index)
            self.assertEqual(values.shape, (n, m))
            self.assertTrue((values.sum(axis=1) == fd.POINTS).all())
            self.assertTrue((values >= 0).all())

    @needs("scipy")
    def test_exact_program_matches_brute_force(self):
        for n, m, seed in ((2, 9, 11), (3, 7, 12), (3, 8, 13), (4, 6, 14)):
            values = fd.generate(n, m, seed)
            best = max(fd.nash_welfare(fd.utilities(values, np.array(owner)))
                       for owner in itertools.product(range(n), repeat=m))
            owner, welfare = fd.optimal_allocation(values)
            self.assertAlmostEqual(welfare, best, places=9)

    # Measured at 26 s on one thread: runs only with ARENA_SLOW_TESTS=1. FINANCE_SLOW_TESTS=1 adds the larger instances.
    @slow
    @needs("scipy")
    def test_recorded_optima_are_reproduced_and_envy_free_up_to_one(self):
        for index in range(len(fd.SIZES) if SLOW else 5):
            values = fd.valuations(index)
            owner, welfare = fd.optimal_allocation(values)
            self.assertAlmostEqual(welfare / fd.OPTIMUM[index], 1.0, places=11)
            self.assertTrue(fd.envy_free_up_to_one(values, owner))

    def test_scoring_and_validation(self):
        score, details = fd.fair_division_score(round_robin)
        self.assertAlmostEqual(score, 0.872, delta=0.001)
        self.assertEqual(details["ef1_instances"], 8)
        bad = [lambda v: [0] * (v.shape[1] - 1), lambda v: [v.shape[0]] * v.shape[1], lambda v: [True] * v.shape[1],
               lambda v: [0.0] * v.shape[1], lambda v: "0" * v.shape[1]]
        for candidate in bad:
            with self.assertRaises(CandidateError):
                fd.fair_division_score(candidate)
        starved, _ = fd.fair_division_score(lambda v: [0] * v.shape[1])  # one agent takes everything
        self.assertEqual(starved, 0.0)


if __name__ == "__main__":
    unittest.main()
