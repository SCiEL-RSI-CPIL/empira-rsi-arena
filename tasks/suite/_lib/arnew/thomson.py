"""Thomson problem: n unit charges on the sphere, minimise sum over pairs of 1 / distance.

The verifier normalises the candidate's 3D vectors to unit length (zero vectors are rejected) and sums 1/r over all pairs.
Records are the energies listed in the Wikipedia table (9 decimals), so a result counts as an improvement only when it is
lower by more than 1e-8. The Cambridge Cluster Database lists no lower energy for any of these sizes (compared 2026-11-15).
"""
from __future__ import annotations

import numpy as np

from .common import CandidateError, require

IMPROVEMENT_TOL = 1e-8


def energy(points):
    p = np.asarray(points, dtype=float)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    iu = np.triu_indices(len(p), 1)
    return float(np.sum(1.0 / d[iu]))


def check(points, n):
    try:
        p = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        raise CandidateError("return an array of shape (n, 3)") from None
    require(p.shape == (n, 3), f"return {n} points with 3 coordinates each")
    require(bool(np.isfinite(p).all()), "coordinates must be finite")
    require(bool((np.linalg.norm(p, axis=1) > 1e-12).all()), "a point is the zero vector")
    e = energy(p)
    require(np.isfinite(e), "two points coincide")
    return e, {"energy": e, "n": n, "verification": "exact_coulomb_energy_of_normalised_points"}
