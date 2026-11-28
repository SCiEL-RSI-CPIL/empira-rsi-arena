"""Two-stream radiative transfer: the reference solver against Monte Carlo and shooting, and the seed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

try:
    from scipy.linalg import expm
except ImportError:  # optional: the tests that integrate with scipy are skipped
    expm = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import earth_common as ec  # noqa: E402
from autoresearch_suite import earth_twostream as ts  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import module_from, seed_source  # noqa: E402

# The seed is the package's: the generator module that defines it in the source repository is not in this library.
TWOSTREAM_SEED = seed_source("science-earth-two-stream-radiative-transfer")


def monte_carlo(tau, omega, g, mu0, photons, seed=3):
    """Independent photon random walk in a slab (HG scattering, black surface)."""
    rng = np.random.default_rng(seed)
    z, mu, w = np.zeros(photons), np.full(photons, mu0), np.ones(photons)
    alive = np.ones(photons, bool)
    up = down = 0.0
    while alive.any():
        idx = np.flatnonzero(alive)
        z[idx] += -np.log(rng.random(len(idx))) * mu[idx]
        bottom, top = idx[z[idx] >= tau], idx[z[idx] <= 0]
        down, up = down + w[bottom].sum(), up + w[top].sum()
        alive[bottom] = alive[top] = False
        idx = np.flatnonzero(alive)
        w[idx] *= omega
        u = rng.random(len(idx))
        cos = (1 + g * g - ((1 - g * g) / (1 - g + 2 * g * u)) ** 2) / (2 * g)
        sin, phi, m = np.sqrt(np.maximum(0, 1 - cos ** 2)), 2 * np.pi * rng.random(len(idx)), mu[idx]
        mu[idx] = m * cos + np.sqrt(np.maximum(0, 1 - m * m)) * sin * np.cos(phi)
    return up / photons, down / photons


def shooting(tau, omega, mu0, g1, g2, g3):
    """Two-stream fluxes by the matrix exponential of the augmented linear system (valid for small tau)."""
    a = np.array([[g1, -g2, -g3 * omega], [g2, -g1, (1 - g3) * omega], [0.0, 0.0, -1.0 / mu0]])
    e = expm(a * tau)
    up0 = -e[0, 2] / e[0, 0]
    down = e[1, 0] * up0 + e[1, 2]
    return up0 / mu0, down / mu0 + np.exp(-tau / mu0)


class SolverTests(unittest.TestCase):
    def test_reference_conserves_energy_and_converges(self):
        r, d, s = ts.discrete_ordinates(2.0, 1.0, 0.85, 0.4)
        self.assertAlmostEqual(r + d + s, 1.0, places=7)
        coarse = ts.discrete_ordinates(3.0, 0.95, 0.7, 0.6, n=24)
        fine = ts.discrete_ordinates(3.0, 0.95, 0.7, 0.6, n=48)
        self.assertLess(max(abs(a - b) for a, b in zip(coarse, fine)), 1e-7)

    def test_reference_agrees_with_monte_carlo(self):
        r, d, s = ts.discrete_ordinates(1.5, 0.98, 0.6, 0.7)
        mc_r, mc_t = monte_carlo(1.5, 0.98, 0.6, 0.7, photons=120000)
        self.assertLess(abs(mc_r - r), 0.006)
        self.assertLess(abs(mc_t - d - s), 0.006)

    @unittest.skipUnless(expm, "scipy is not installed")
    def test_closed_form_two_stream_matches_shooting(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            g2 = rng.uniform(-0.3, 1.0)
            g1 = abs(g2) + rng.uniform(0.0, 1.0)
            g3, omega, mu0, tau = rng.uniform(0, 1), rng.uniform(0.5, 1.0), rng.uniform(0.1, 1.0), rng.uniform(0.01, 2.0)
            r, t = ts.two_stream(np.array([tau]), np.array([omega]), np.array([mu0]), np.array([g1]), np.array([g2]), np.array([g3]))
            r_ref, t_ref = shooting(tau, min(omega, ts.OMEGA_CAP), mu0, np.sqrt(g2 ** 2 + max(g1 ** 2 - g2 ** 2, ts.K_FLOOR ** 2)), g2, g3)
            self.assertAlmostEqual(float(r[0]), r_ref, places=8)
            self.assertAlmostEqual(float(t[0]), t_ref, places=8)

    def test_conservative_two_stream_conserves_energy(self):
        g = np.array([0.0, 0.5, 0.9])
        r, t = ts.classic("delta_eddington", np.ones(3), g, np.full(3, 0.5), np.full(3, 50.0))
        np.testing.assert_allclose(r + t, 1.0, atol=1e-6)


@unittest.skipUnless((ec.data_dir() / ts.DATA).is_file(), "run scripts/prepare_earth.py first")
class HeldOutTests(unittest.TestCase):
    def test_seed_is_delta_eddington(self):
        value, details = ts.score(module_from(TWOSTREAM_SEED))
        self.assertAlmostEqual(value, details["delta_eddington_mean_error"], places=6)
        self.assertEqual(details["cases"], ts.CASES)

    def test_invalid_coefficients_are_rejected(self):
        bad = TWOSTREAM_SEED + "\n\n_seed = two_stream_coefficients\n\ndef two_stream_coefficients(omega, g, mu0):\n    a, b, c, f = _seed(omega, g, mu0)\n    return a, b, c + 2.0, f\n"
        with self.assertRaises(CandidateError):
            ts.score(module_from(bad))
        nan = "import numpy as np\n\ndef two_stream_coefficients(omega, g, mu0):\n    return [np.full(len(g), np.nan)] * 4\n"
        with self.assertRaises(CandidateError):
            ts.score(module_from(nan))


if __name__ == "__main__":
    unittest.main()
