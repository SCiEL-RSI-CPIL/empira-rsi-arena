"""Inverse design of a two-dimensional photonic crystal for a large TM band gap.

The candidate draws one unit cell of a square-lattice photonic crystal on a 32 x 32 pixel
grid (air, epsilon 1, or dielectric, epsilon 12) with the full square symmetry. For
TM polarization (electric field along the rods) the evaluator solves the Helmholtz
eigenproblem -(nabla + ik)^2 E = (omega/c)^2 epsilon E with a plane-wave Galerkin basis
(289 waves; the Fourier coefficients of the pixel map are computed exactly), sweeps the
boundary and interior of the irreducible Brillouin zone and scores the gap-midgap ratio
between TM bands 4 and 5, the quantity maximized in band-gap topology optimization
(Cox and Dobson 1999; Sigmund and Jensen 2003; Men et al. 2010).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .bio_physics_common import as_int_array, run_sandboxed
from .runtime import require

PIXELS, EPS_LOW, EPS_HIGH, CUTOFF = 32, 1.0, 12.0, 8


def fourier_coefficients(pixels, eps_low=EPS_LOW, eps_high=EPS_HIGH, cutoff=CUTOFF):
    """Exact Fourier coefficients eps(m, n), |m|, |n| <= 2 * cutoff, of the piecewise-constant pixel map."""
    size = pixels.shape[0]
    eps = np.where(pixels > 0, eps_high, eps_low).astype(float)
    transform = np.fft.fft2(eps) / size ** 2
    index = np.arange(-2 * cutoff, 2 * cutoff + 1)
    m, n = np.meshgrid(index, index, indexing="ij")
    phase = np.exp(-1j * np.pi * (m + n) / size)  # pixel centres instead of corners
    form = np.sinc(m / size) * np.sinc(n / size)  # Fourier transform of one square pixel
    return transform[m % size, n % size] * phase * form


def tm_bands(pixels, kpoints, bands, eps_low=EPS_LOW, eps_high=EPS_HIGH, cutoff=CUTOFF):
    """Lowest TM frequencies omega a / (2 pi c) at each k (rows) for the pixel map."""
    coefficients = fourier_coefficients(pixels, eps_low, eps_high, cutoff)
    g = np.arange(-cutoff, cutoff + 1)
    gm, gn = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
    epsilon = coefficients[gm[:, None] - gm[None, :] + 2 * cutoff, gn[:, None] - gn[None, :] + 2 * cutoff]
    rows = []
    for kx, ky in kpoints:
        stiffness = np.diag((kx + 2 * np.pi * gm) ** 2 + (ky + 2 * np.pi * gn) ** 2).astype(complex)
        values = sla.eigh(stiffness, epsilon, eigvals_only=True, subset_by_index=[0, bands - 1])
        rows.append(np.sqrt(np.clip(values.real, 0.0, None)) / (2 * np.pi))
    return np.array(rows)


def brillouin_samples(per_edge=8):
    """Gamma-X-M-Gamma boundary of the irreducible zone plus interior points of the triangle."""
    gamma, x, m = np.array([0.0, 0.0]), np.array([np.pi, 0.0]), np.array([np.pi, np.pi])
    points = [a + t * (b - a) for a, b in ((gamma, x), (x, m), (m, gamma))
              for t in np.linspace(0.0, 1.0, per_edge, endpoint=False)]
    points += [np.array([np.pi * u, np.pi * v]) for u, v in ((0.5, 0.25), (0.75, 0.25), (0.75, 0.5), (0.25, 0.125),
                                                              (0.875, 0.625), (0.625, 0.375))]
    return points


def gap_ratio(frequencies, lower_band):
    """Gap-midgap ratio between bands lower_band and lower_band + 1 (1-based); negative when they overlap."""
    top = float(frequencies[:, lower_band - 1].max())
    bottom = float(frequencies[:, lower_band].min())
    return 2.0 * (bottom - top) / (bottom + top), top, bottom


def check_cell(cell):
    pixels = as_int_array(cell, shape=(PIXELS, PIXELS), name="cell", low=0, high=1)
    require(np.array_equal(pixels, pixels[::-1, :]) and np.array_equal(pixels, pixels[:, ::-1])
            and np.array_equal(pixels, pixels.T), "cell: must have the square (C4v) symmetry about the cell centre")
    return pixels


DRIVER = '''

def _bp_photonic_driver(problem):
    _bp_seed()
    return _bp_plain(design(problem))
'''


def score(module, P):
    lower = int(P.get("lower_band", 4))
    problem = {"pixels": PIXELS, "eps_low": EPS_LOW, "eps_high": EPS_HIGH, "lower_band": lower, "polarization": "TM"}
    cell = run_sandboxed(module, DRIVER, "_bp_photonic_driver", [problem], required=("design",),
                         cpu_seconds=int(P.get("cpu_seconds", 120)), memory_mb=2048)
    pixels = check_cell(cell)
    frequencies = tm_bands(pixels, brillouin_samples(), lower + 1)
    value, top, bottom = gap_ratio(frequencies, lower)
    return value, {"gap_midgap_ratio": value, "band_top_frequency": top, "band_bottom_frequency": bottom,
                   "dielectric_fill_fraction": float(pixels.mean()), "lower_band": lower,
                   "frequency_units": "omega a / (2 pi c)"}
