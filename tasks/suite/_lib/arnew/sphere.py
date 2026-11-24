"""Spherical codes / Tammes problem: n points on the unit sphere S^(d-1) maximising the minimum pairwise angle.

The verifier normalises the candidate's vectors to unit length (zero vectors are rejected) and returns the minimum angle in
degrees.

The record of a task is the minimum angle of the code listed for its size in Henry Cohn's table of spherical codes
(spherical-codes.org, dimensions 3, 4, 5; read on the date in catalog/source-lock.json), computed by this verifier from the
table's coordinates. The table is kept up to date; Sloane's tables (neilsloane.com/packings, 1994-96), the first source of
this family, are behind it for about a third of the sizes and are kept only for comparison (scripts/polish_sphere_records.py).
An improvement must exceed the record by more than IMPROVEMENT_TOL.
"""
from __future__ import annotations

import numpy as np

from .common import CandidateError, read_data, require, unit_rows


def min_angle(points):
    p = unit_rows(np.asarray(points, dtype=float))
    g = p @ p.T
    np.fill_diagonal(g, -1.0)
    return float(np.degrees(np.arccos(np.clip(g.max(), -1.0, 1.0))))


IMPROVEMENT_TOL = 1e-5  # degrees. A margin against numerical slack in a record. The largest slack seen: Sloane's codes after a local
                        # re-optimisation stay up to 7.3e-6 degrees below what Cohn's table lists as the same code (22 sizes). Re-optimising
                        # the table's own coordinates gained less than 1e-13 where it was tried (scripts/polish_sphere_records.py)


def points(dim, n):
    """Sloane's published coordinates for n points in dimension dim, as an (n, dim) array."""
    values = []
    for line in read_data("sphere", f"pack.{dim}.{n}.txt").decode().splitlines():
        parts = line.split()
        try:
            if len(parts) != 1:
                raise ValueError  # a header line such as "3 124 1" in pack.3.124, or prose
            values.append(float(parts[0]))
        except ValueError:
            if values:
                break  # coordinates are one number per line; stop at the first blank or text line after them (pack.5.14 appends notes)
    values = np.array(values)
    require(len(values) == n * dim, f"pack.{dim}.{n}.txt has {len(values)} numbers, expected {n * dim}")
    return values.reshape(n, dim)


def table_points(dim, n):
    """The coordinates that Cohn's table of spherical codes serves for n points in dimension dim, as an (n, dim) array."""
    rows = [line.split(",") for line in read_data("sphere", f"cohn.{dim}.{n}.csv").decode().splitlines() if line.strip()]
    try:
        values = np.array(rows, dtype=float)
    except ValueError:
        raise CandidateError(f"cohn.{dim}.{n}.csv is not a table of numbers") from None
    require(values.shape == (n, dim), f"cohn.{dim}.{n}.csv has shape {values.shape}, expected {(n, dim)}")
    return values


def instance(dim, n):
    """Minimum angle (degrees) of Sloane's published coordinates as they are."""
    return min_angle(points(dim, n))


def check(points, dim, n):
    try:
        p = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        raise CandidateError("return an array of shape (n, dim)") from None
    require(p.shape == (n, dim), f"return {n} points with {dim} coordinates each")
    require(bool(np.isfinite(p).all()), "coordinates must be finite")
    norms = np.linalg.norm(p, axis=1)
    require(bool((norms > 1e-12).all()), "a point is the zero vector")
    angle = min_angle(p)
    return angle, {"min_angle_degrees": angle, "n": n, "dimension": dim, "verification": "exact_min_angle_of_normalised_points"}
