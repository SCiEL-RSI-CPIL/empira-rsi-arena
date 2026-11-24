"""Photonic band-gap task: plane-wave solver checks against analytic bands and a published MPB result."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

NEEDS_SCIPY = unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is not installed")
if importlib.util.find_spec("scipy"):  # the evaluator module imports scipy.linalg when it is loaded
    from autoresearch_suite import phys_photonic as ph  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402


def rod_cell(radius=0.2, size=32):
    x = (np.arange(size) + 0.5) / size - 0.5
    return ((x[:, None] ** 2 + x[None, :] ** 2) < radius ** 2).astype(int)


@NEEDS_SCIPY
class SolverTests(unittest.TestCase):
    def test_uniform_medium_gives_folded_free_photon_bands(self):
        uniform = np.ones((32, 32), int)
        k = np.array([0.3 * np.pi, 0.1 * np.pi])
        frequencies = ph.tm_bands(uniform, [k], 6)[0]
        g = np.arange(-3, 4)
        exact = sorted(np.hypot(k[0] + 2 * np.pi * a, k[1] + 2 * np.pi * b) / (2 * np.pi * np.sqrt(12.0))
                       for a in g for b in g)[:6]
        self.assertTrue(np.allclose(frequencies, exact, atol=1e-10))

    def test_fourier_coefficients_of_uniform_and_half_filled_cells(self):
        coefficients = ph.fourier_coefficients(np.ones((32, 32), int))
        centre = coefficients.shape[0] // 2
        self.assertAlmostEqual(coefficients[centre, centre].real, 12.0, places=12)
        self.assertAlmostEqual(float(np.abs(coefficients).sum() - abs(coefficients[centre, centre])), 0.0, places=10)
        half = np.zeros((32, 32), int)
        half[:16] = 1
        self.assertAlmostEqual(ph.fourier_coefficients(half)[centre, centre].real, 6.5, places=12)

    def test_rod_crystal_reproduces_the_published_mpb_gap(self):
        # MPB Python tutorial: rods of epsilon 12, radius 0.2a, TM gap between bands 1 and 2 of 38.95 %
        # (band edges 0.2826 and 0.4193). The local pixelated rod differs by a fraction of a point.
        frequencies = ph.tm_bands(rod_cell(), ph.brillouin_samples(), 2)
        value, top, bottom = ph.gap_ratio(frequencies, 1)
        self.assertAlmostEqual(value, 0.3895, delta=0.005)
        self.assertAlmostEqual(top, 0.2826, delta=0.002)
        self.assertAlmostEqual(bottom, 0.4193, delta=0.003)

    def test_plane_wave_truncation_is_converged(self):
        cell = rod_cell(0.27)
        coarse = ph.gap_ratio(ph.tm_bands(cell, ph.brillouin_samples(4), 5, cutoff=8), 4)[0]
        fine = ph.gap_ratio(ph.tm_bands(cell, ph.brillouin_samples(4), 5, cutoff=12), 4)[0]
        self.assertLess(abs(coarse - fine), 0.002)


@NEEDS_SCIPY
class ContractTests(unittest.TestCase):
    def test_symmetric_cell_is_accepted(self):
        self.assertEqual(ph.check_cell(rod_cell().tolist()).shape, (32, 32))

    def test_invalid_cells_are_rejected(self):
        asymmetric = rod_cell()
        asymmetric[0, 1] = 1
        for bad in (asymmetric, np.full((32, 32), 2), np.ones((31, 31), int), np.full((32, 32), 0.5),
                    [[0] * 32] * 31):
            with self.assertRaises(CandidateError):
                ph.check_cell(np.asarray(bad).tolist())


if __name__ == "__main__":
    unittest.main()
