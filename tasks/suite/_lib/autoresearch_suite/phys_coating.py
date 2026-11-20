"""Broadband antireflection coatings designed under the exact thin-film transfer-matrix model.

A coating is a stack of homogeneous, lossless, non-dispersive layers of two materials
(low index 1.38, high index 2.35) on a glass substrate (index 1.52) in air. At normal
incidence the characteristic-matrix method (Abeles; Macleod, Thin-Film Optical Filters)
gives the reflectance exactly. The candidate proposes the layer sequence; the evaluator
checks the manufacturing constraints and averages the reflectance over 400-800 nm.
"""
from __future__ import annotations

import numpy as np

from .bio_physics_common import as_array, run_sandboxed
from .runtime import require

AMBIENT, SUBSTRATE = 1.0, 1.52
MATERIALS = {"L": 1.38, "H": 2.35}


def reflectance(indices, thicknesses_nm, wavelengths_nm, ambient=AMBIENT, substrate=SUBSTRATE):
    """Normal-incidence reflectance of a lossless multilayer (layers listed from the ambient side)."""
    wavelengths = np.asarray(wavelengths_nm, float)
    m11 = np.ones_like(wavelengths, dtype=complex)
    m12 = np.zeros_like(wavelengths, dtype=complex)
    m21 = np.zeros_like(wavelengths, dtype=complex)
    m22 = np.ones_like(wavelengths, dtype=complex)
    for n, d in zip(indices, thicknesses_nm):
        delta = 2 * np.pi * n * d / wavelengths
        c, s = np.cos(delta), np.sin(delta)
        a11, a12, a21, a22 = c, 1j * s / n, 1j * n * s, c
        m11, m12, m21, m22 = (m11 * a11 + m12 * a21, m11 * a12 + m12 * a22,
                              m21 * a11 + m22 * a21, m21 * a12 + m22 * a22)
    b = m11 + m12 * substrate
    c = m21 + m22 * substrate
    r = (ambient * b - c) / (ambient * b + c)
    return np.abs(r) ** 2


DRIVER = '''

def _bp_coating_driver(problem):
    _bp_seed()
    return _bp_plain(design(problem))
'''


def check_design(design_output, max_layers, max_total_nm, max_layer_nm):
    """Validate (materials, thicknesses) and return the index and thickness arrays."""
    require(isinstance(design_output, (list, tuple)) and len(design_output) == 2,
            "design() must return (materials, thicknesses)")
    materials, thicknesses = design_output
    require(isinstance(materials, (list, tuple)) and all(m in MATERIALS for m in materials),
            "materials: expected a list of 'L' and 'H' labels")
    require(1 <= len(materials) <= max_layers, f"materials: between 1 and {max_layers} layers")
    widths = as_array(thicknesses, shape=(len(materials),), name="thicknesses", low=0.0, high=max_layer_nm)
    require(widths.sum() <= max_total_nm + 1e-9, f"thicknesses: total physical thickness above {max_total_nm} nm")
    return np.array([MATERIALS[m] for m in materials]), widths


def score(module, P):
    band = (float(P["band_nm"][0]), float(P["band_nm"][1]))
    wavelengths = np.linspace(band[0], band[1], int(P.get("points", 401)))
    problem = {"ambient_index": AMBIENT, "substrate_index": SUBSTRATE, "materials": dict(MATERIALS),
               "band_nm": list(band), "max_layers": int(P["max_layers"]), "max_total_nm": float(P["max_total_nm"]),
               "max_layer_nm": float(P["max_layer_nm"])}
    answer = run_sandboxed(module, DRIVER, "_bp_coating_driver", [problem], required=("design",),
                           cpu_seconds=int(P.get("cpu_seconds", 60)), memory_mb=1024)
    indices, widths = check_design(answer, problem["max_layers"], problem["max_total_nm"], problem["max_layer_nm"])
    curve = reflectance(indices, widths, wavelengths)
    value = float(100.0 * curve.mean())
    return value, {"layers": int(len(widths)), "total_thickness_nm": float(widths.sum()),
                   "mean_reflectance_percent": value, "max_reflectance_percent": float(100.0 * curve.max()),
                   "bare_substrate_percent": float(100.0 * ((AMBIENT - SUBSTRATE) / (AMBIENT + SUBSTRATE)) ** 2)}
