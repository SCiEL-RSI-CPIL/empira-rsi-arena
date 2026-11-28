"""School choice with ties: weak stability, DA-STB baseline, exact optimum against brute force, scoring."""
from __future__ import annotations

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_school as fs  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import needs, slow  # noqa: E402


def da_stb(problem):
    return fs.deferred_acceptance(problem, problem["lottery"])


def brute_force(problem):
    S, L = problem["preferences"].shape
    best = None
    for choice in itertools.product(range(-1, L), repeat=S):
        match = np.array([-1 if k < 0 else problem["preferences"][s, k] for s, k in enumerate(choice)])
        if (np.bincount(match[match >= 0], minlength=problem["schools"]) > problem["capacities"]).any():
            continue
        if fs.blocking_pair(problem, match) is None:
            total = int(fs.ranks(problem, match).sum())
            best = total if best is None else min(best, total)
    return best


class SchoolChoiceTests(unittest.TestCase):
    @needs("scipy")
    def test_exact_program_matches_brute_force_on_small_markets(self):
        for seed, tiers in ((1, 1), (2, 2), (3, 1), (4, 3)):
            problem = fs.generate(8, 3, 2, seed, tiers)
            self.assertEqual(fs.optimal_matching(problem)[1], brute_force(problem))

    # Measured at 27 s on one thread: runs only with ARENA_SLOW_TESTS=1.
    @slow
    @needs("scipy")
    def test_recorded_benchmarks_and_optima(self):
        for index in range(len(fs.SIZES)):
            problem = fs.instance(index)
            baseline = da_stb(problem)
            self.assertIsNone(fs.blocking_pair(problem, baseline))
            self.assertEqual(int(fs.ranks(problem, baseline).sum()), fs.DA_RANK[index])
            match, total = fs.optimal_matching(problem)
            self.assertEqual(total, fs.OPT_RANK[index])
            self.assertIsNone(fs.blocking_pair(problem, match))

    def test_blocking_pairs_and_contract_violations(self):
        problem = fs.instance(0)
        match = da_stb(problem)
        dropped = match.copy()
        dropped[np.nonzero(match >= 0)[0][0]] = -1  # leaves a seat the student wants
        self.assertIsNotNone(fs.blocking_pair(problem, dropped))
        S = problem["students"]
        off_list = match.copy()
        s = int(np.nonzero(match >= 0)[0][0])
        off_list[s] = next(c for c in range(problem["schools"]) if c not in problem["preferences"][s])
        overfull = np.full(S, int(problem["preferences"][0, 0]))
        for bad in (match[:-1], match.astype(float), np.full(S, problem["schools"]), off_list, overfull):
            with self.assertRaises(CandidateError):
                fs.validate(problem, bad)

    # Measured at 27 s on one thread: runs only with ARENA_SLOW_TESTS=1.
    @slow
    @needs("scipy")
    def test_scores(self):
        self.assertEqual(fs.school_score(da_stb)[0], 0.0)
        optimal = {fs.digest(fs.instance(i)): fs.optimal_matching(fs.instance(i))[0] for i in range(len(fs.SIZES))}
        self.assertEqual(fs.school_score(lambda p: optimal[fs.digest(p)])[0], 1.0)
        with self.assertRaises(CandidateError):
            fs.school_score(lambda p: np.full(p["students"], -1))  # empty matching wastes every seat


if __name__ == "__main__":
    unittest.main()
