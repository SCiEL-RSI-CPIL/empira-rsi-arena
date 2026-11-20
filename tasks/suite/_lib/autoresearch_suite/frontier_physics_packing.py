"""Packings of equal circles and spheres in fixed containers (frontier physics collection).

The candidate returns only the N centres. For a container C the largest common
radius of pairwise disjoint balls centred there and contained in C is

    r = min( min_{i<j} |c_i - c_j| / 2 ,  min_i clearance_C(c_i) )

with

    disk, ball:      clearance(c) = 1 - |c|              (container radius 1, centred at the origin)
    square, cube:    clearance(c) = min_k (1/2 - |c_k|)  (side 1, centred at the origin)

These are the conventions of E. Specht's record tables (www.packomania.com):
a published centre file is scored without any transformation. Any set of
distinct centres strictly inside the container defines a valid packing, so the
validity test has no tolerance: a centre on or outside the boundary, or two
coincident centres, give r <= 0 and are rejected. The score is r computed in
IEEE double precision (maximize).

Tolerance policy for comparisons with a record. Published configurations are
given with 30 or 12 decimals and reproduce their tabulated radii to within
1.1e-12 here (validation/human-results/frontier_physics.json). A score above the
record by at most ``IMPROVEMENT_MARGIN`` (1e-11, absolute) is reported as a tie,
not as a new record; any claimed improvement should be confirmed in higher
precision before it is announced.
"""
from __future__ import annotations

import math

import numpy as np

from .runtime import CandidateError, require

IMPROVEMENT_MARGIN = 1e-11
CONTACT_TOLERANCE = 1e-9  # relative; diagnostic contact counts only
CONTAINERS = {"disk": 2, "square": 2, "cube": 3, "ball": 3}


def centres(candidate, n, dim):
    """Strictly validated float copy of an (n, dim) array of centres."""
    try:
        x = np.array(candidate, dtype=float)
    except (TypeError, ValueError):
        raise CandidateError(f"construct() must return {n} centres as an ({n}, {dim}) numeric array") from None
    require(x.shape == (n, dim), f"expected an array of shape ({n}, {dim}); got {x.shape}")
    require(bool(np.all(np.isfinite(x))), "centres contain NaN or infinite values")
    return x


def clearance(x, container):
    """Distance from each centre to the container boundary (negative outside)."""
    if container in ("disk", "ball"):
        return 1.0 - np.sqrt(np.einsum("ij,ij->i", x, x))
    return 0.5 - np.max(np.abs(x), axis=1)


def pair_distances(x):
    """Distances |c_i - c_j| for i < j, from coordinate differences (no Gram-matrix cancellation)."""
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    return dist[np.triu_indices(len(x), 1)]


def density(radius, n, container):
    """Fraction of the container volume covered by the n balls."""
    if container == "disk":
        return n * radius ** 2
    if container == "square":
        return n * math.pi * radius ** 2
    if container == "ball":
        return n * radius ** 3
    return n * 4.0 / 3.0 * math.pi * radius ** 3


def record_comparison(score, published, direction, margin):
    """Position of a score relative to a published record under an explicit tie margin."""
    if published is None:
        return {}
    published = float(published)
    gain = (score - published) if direction == "maximize" else (published - score)
    status = "above_record" if gain > margin else "below_record" if gain < -margin else "ties_record"
    return {"best_known": published, "gain_over_best_known": gain, "record_margin": margin, "record_status": status}


def packing_radius(candidate, n, container, published=None):
    """Largest common radius of n disjoint balls centred at the candidate's centres; maximize."""
    if container not in CONTAINERS:
        raise ValueError(f"unknown container {container!r}")
    x = centres(candidate, n, CONTAINERS[container])
    edge = clearance(x, container)
    outside = int(np.sum(edge <= 0.0))
    require(outside == 0, f"{outside} centre(s) lie on or outside the container")
    pairs = pair_distances(x)
    half = float(pairs.min()) / 2.0
    require(half > 0.0, "two centres coincide")
    wall = float(edge.min())
    radius = min(half, wall)
    details = {
        "container": container,
        "n": n,
        "radius": radius,
        "half_min_distance": half,
        "min_wall_clearance": wall,
        "limited_by": "pair" if half <= wall else "wall",
        "density": density(radius, n, container),
        "pair_contacts": int(np.sum(pairs / 2.0 <= radius * (1.0 + CONTACT_TOLERANCE))),
        "wall_contacts": int(np.sum(edge <= radius * (1.0 + CONTACT_TOLERANCE))),
    }
    details.update(record_comparison(radius, published, "maximize", IMPROVEMENT_MARGIN))
    return radius, details
