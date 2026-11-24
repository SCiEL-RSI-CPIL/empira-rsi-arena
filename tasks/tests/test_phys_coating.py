"""Antireflection coating task: transfer-matrix reflectance against analytic and recursive formulas."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import phys_coating as c  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402


def airy_recursive(indices, thicknesses, wavelength, ambient=1.0, substrate=1.52):
    """Independent check: Rouard's method, combining interface Fresnel coefficients from the substrate outwards."""
    media = [ambient] + list(indices) + [substrate]
    r = (media[-2] - media[-1]) / (media[-2] + media[-1])
    for k in range(len(indices), 0, -1):
        phase = np.exp(2j * 2 * np.pi * media[k] * thicknesses[k - 1] / wavelength)
        interface = (media[k - 1] - media[k]) / (media[k - 1] + media[k])
        r = (interface + r * phase) / (1 + interface * r * phase)
    return abs(r) ** 2


class ReflectanceTests(unittest.TestCase):
    def test_bare_substrate_matches_fresnel(self):
        self.assertAlmostEqual(float(c.reflectance([], [], [550.0])[0]), ((1 - 1.52) / 2.52) ** 2, places=14)

    def test_quarter_wave_layer_matches_the_closed_form(self):
        n = 1.38
        value = float(c.reflectance([n], [550.0 / (4 * n)], [550.0])[0])
        self.assertAlmostEqual(value, ((1.52 - n ** 2) / (1.52 + n ** 2)) ** 2, places=12)

    def test_half_wave_layer_is_absentee(self):
        value = float(c.reflectance([2.35], [550.0 / (2 * 2.35)], [550.0])[0])
        self.assertAlmostEqual(value, ((1 - 1.52) / 2.52) ** 2, places=12)

    def test_matrix_method_agrees_with_rouard_recursion(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            layers = int(rng.integers(1, 12))
            indices = rng.choice([1.38, 2.35], size=layers)
            widths = rng.uniform(0, 300, size=layers)
            for wavelength in (400.0, 523.0, 800.0):
                matrix = float(c.reflectance(indices, widths, [wavelength])[0])
                self.assertAlmostEqual(matrix, airy_recursive(indices, widths, wavelength), places=12)

    def test_reflectance_stays_in_the_unit_interval(self):
        curve = c.reflectance([2.35, 1.38] * 5, [80.0, 120.0] * 5, np.linspace(400, 800, 101))
        self.assertTrue(np.all(curve >= 0) and np.all(curve <= 1))


class ConstraintTests(unittest.TestCase):
    def test_valid_design_is_accepted(self):
        indices, widths = c.check_design((["L", "H"], [100.0, 50.0]), 20, 1000, 400)
        self.assertEqual(list(indices), [1.38, 2.35])
        self.assertEqual(list(widths), [100.0, 50.0])

    def test_invalid_designs_are_rejected(self):
        for output in ((["L"], [500.0]), (["X"], [10.0]), (["L", "H"], [10.0]), (["L"], [-1.0]),
                       (["L"] * 21, [1.0] * 21), (["L"] * 3, [400.0, 400.0, 300.0]), ([], []),
                       (["L"], [float("nan")]), "LH"):
            with self.assertRaises(CandidateError):
                c.check_design(output, 20, 1000, 400)


if __name__ == "__main__":
    unittest.main()
