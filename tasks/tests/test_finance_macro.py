"""FRED-MD forecasting: data layout, targets, benchmark, information set and validation."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import finance_macro as fm  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import have_data  # noqa: E402


@unittest.skipUnless(have_data("fredmd_2024_10.csv"), "run scripts/prepare_finance.py first")
class MacroTests(unittest.TestCase):
    def test_layout_and_targets(self):
        months, names, tcodes, panel = fm.load()
        self.assertEqual((int(months[0]), int(months[-1]), len(names)), (195901, 202408, 126))
        self.assertEqual(int(tcodes[names.index("INDPRO")]), 5)
        ip = panel[:, names.index("INDPRO")]
        t = int(np.nonzero(months == 200001)[0][0])
        self.assertAlmostEqual(fm.ahead(ip, "growth", t, 12), 100 * np.log(ip[t + 12] / ip[t]))
        u = panel[:, names.index("UNRATE")]
        self.assertAlmostEqual(fm.ahead(u, "difference", t, 6), u[t + 6] - u[t])

    def test_benchmark_seed_scores_one_and_sees_only_the_past(self):
        seen = []

        def spy(history, horizons):
            seen.append((int(history["yyyymm"][-1]), history["panel"].shape[0]))
            self.assertEqual(history["panel"].shape[0], len(history["yyyymm"]))
            return fm.ar_bic(history, horizons)
        for target in fm.KINDS:
            score, details = fm.macro_score(spy, target)
            self.assertEqual(score, 1.0)
            self.assertEqual(details["scored_forecasts_by_horizon"], [394, 389, 383])
        self.assertEqual(seen[0][0], 199001)
        self.assertEqual(max(m for m, _ in seen), 202308)

    def test_invalid_forecasts(self):
        for bad in (lambda h, z: np.zeros(2), lambda h, z: np.full(3, np.nan), lambda h, z: np.full(3, 5e3), lambda h, z: "1"):
            with self.assertRaises(CandidateError):
                fm.macro_score(bad, "UNRATE")


if __name__ == "__main__":
    unittest.main()
