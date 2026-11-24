"""Market making: dynamic programming optimum, simulator consistency, references and validation."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_marketmaking as mm  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import slow  # noqa: E402


def constant(t, q, mid, params):
    depth = np.full(len(q), 1.0 / params["k"])
    return depth, depth.copy()


class MarketMakingTests(unittest.TestCase):
    def test_best_depth_solves_the_one_step_problem(self):
        regime = mm.REGIMES[0]
        grid = np.linspace(-1.0, 4.0, 200001)
        for constant_term in (-0.3, 0.0, 0.5):
            best = grid[np.argmax(mm.fill_probability(grid, regime) * (grid + constant_term))]
            self.assertAlmostEqual(float(mm._best_depth(np.array([constant_term]), regime)[0]), best, places=4)

    # Measured at 47 s on one thread: runs only with ARENA_SLOW_TESTS=1.
    @slow
    def test_dynamic_programming_value_matches_simulation(self):
        for index in range(len(mm.REGIMES)):
            values, _ = mm.simulate(mm.table_policy(index), index)
            value = mm.optimal_policy(index)[2]
            self.assertLess(abs(values.mean() - value), 3.0 * values.std() / np.sqrt(len(values)))

    def test_optimal_quotes_lean_against_inventory(self):
        bids, asks, _ = mm.optimal_policy(1)
        q = mm.REGIMES[1]["Q"]
        self.assertTrue((np.diff(bids[0, :-1]) > 0).all())  # long inventory -> quote the bid deeper
        self.assertTrue((np.diff(asks[0, 1:]) < 0).all())   # ... and the ask tighter (ask depths fall as q rises)
        self.assertAlmostEqual(bids[0, q], asks[0, q], places=10)

    # Measured at 52 s when run on its own on one thread: runs only with ARENA_SLOW_TESTS=1. The cost is the dynamic programme of the three regimes.
    @slow
    def test_scores_and_validation(self):
        seed, _ = mm.market_making_score(constant)
        approx, _ = mm.market_making_score(mm.glft_approximation)
        self.assertAlmostEqual(seed, 0.742, delta=0.003)
        self.assertGreater(approx, 0.99)
        self.assertLess(approx, 1.01)
        bad = [lambda t, q, m, p: (np.full(len(q), np.nan), np.ones(len(q))), lambda t, q, m, p: (np.ones(3), np.ones(3)),
               lambda t, q, m, p: (np.full(len(q), -10.0), np.ones(len(q))), lambda t, q, m, p: np.ones(len(q))]
        for candidate in bad:
            with self.assertRaises(CandidateError):
                mm.market_making_score(candidate)
        silent, _ = mm.market_making_score(lambda t, q, m, p: (np.full(len(q), np.inf), np.full(len(q), np.inf)))
        self.assertEqual(silent, 0.0)


if __name__ == "__main__":
    unittest.main()
