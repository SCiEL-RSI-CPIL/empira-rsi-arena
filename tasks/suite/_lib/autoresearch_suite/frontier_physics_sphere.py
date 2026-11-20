"""Point configurations on the unit sphere S^2: covering radius and logarithmic energy.

Candidates return N nonzero vectors in R^3. Each vector is divided by its
length and the normalised configuration is the one scored; the largest
deviation of the returned lengths from 1 is reported. Points must be distinct
(chordal separation at least ``MIN_SEPARATION``).

Covering radius (minimize, in degrees). The covering radius of X is
max_{y in S^2} min_i angle(y, x_i). When the origin lies strictly inside the
convex hull of X, the farthest points y are the vertices of the spherical
Voronoi diagram, i.e. the circumcentres of the triangular facets of the hull
(outward facet normals), and the covering radius is the largest facet
circumradius. Facets come from Qhull (scipy.spatial.ConvexHull, triangulated
output). The evaluator recomputes every facet normal from its three vertices
and checks the empty-cap property of every facet against all N points; a facet
whose cap contains another point by more than ``EMPTY_CAP_TOLERANCE`` radians
means the hull is numerically inconsistent, and the candidate is rejected
instead of scored. Configurations inside a closed hemisphere (covering radius
at least 90 degrees) are rejected.

Logarithmic energy (minimize). E = sum_{i<j} log(1 / |x_i - x_j|) with the
natural logarithm, the energy of Smale's 7th problem; the pair terms are
summed with math.fsum.

Tolerance policy for comparisons with a record: covering radii in the
Hardin-Sloane-Smith table are rounded up to 1e-7 degrees, so a covering is
reported as a new record only when it is below the table value by more than
1e-7 degrees; log energies are compared with a margin of 1e-9 (absolute,
about 1e-12 relative for the instances used).
"""
from __future__ import annotations

import math

import numpy as np

from .frontier_physics_packing import record_comparison
from .runtime import CandidateError, require

MIN_SEPARATION = 1e-9
EMPTY_CAP_TOLERANCE = 1e-10
HULL_MARGIN = 1e-9
COVERING_MARGIN_DEGREES = 1e-7
LOG_ENERGY_MARGIN = 1e-9


def unit_vectors(candidate, n):
    """Strictly validated, normalised (n, 3) configuration and the largest length deviation."""
    try:
        x = np.array(candidate, dtype=float)
    except (TypeError, ValueError):
        raise CandidateError(f"construct() must return {n} points as an ({n}, 3) numeric array") from None
    require(x.shape == (n, 3), f"expected an array of shape ({n}, 3); got {x.shape}")
    require(bool(np.all(np.isfinite(x))), "points contain NaN or infinite values")
    lengths = np.sqrt(np.einsum("ij,ij->i", x, x))
    require(bool(np.all(lengths > 1e-12)), "every point must be a nonzero vector (it is projected to the unit sphere)")
    require(bool(np.all(np.isfinite(lengths))), "point coordinates are too large")
    u = x / lengths[:, None]
    deviation = float(np.max(np.abs(lengths - 1.0)))
    return u, deviation


def chord_lengths(u):
    diff = u[:, None, :] - u[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    return dist[np.triu_indices(len(u), 1)]


def _angles(normals, points):
    """Angle between matching rows of two (m, 3) unit-vector arrays, via atan2 for accuracy."""
    cross = np.linalg.norm(np.cross(normals, points), axis=1)
    dot = np.einsum("ij,ij->i", normals, points)
    return np.arctan2(cross, dot)


def covering_radius(candidate, n, published=None):
    """Covering radius of the normalised configuration in degrees; minimize."""
    from scipy.spatial import ConvexHull
    from scipy.spatial import QhullError

    u, deviation = unit_vectors(candidate, n)
    require(n >= 4, "a covering needs at least four points")
    chords = chord_lengths(u)
    require(float(chords.min()) >= MIN_SEPARATION, f"points must be distinct (closest chord {float(chords.min()):.3e})")
    try:
        hull = ConvexHull(u)
    except QhullError as error:
        raise CandidateError(f"convex hull failed (degenerate configuration): {str(error).splitlines()[0]}") from None
    margin = float(-hull.equations[:, 3].max())
    require(margin > HULL_MARGIN, "the points lie in a closed hemisphere, so the covering radius is at least 90 degrees")
    tri = u[hull.simplices]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    require(bool(np.all(lengths > 0.0)), "degenerate hull facet")
    normals /= lengths[:, None]
    flip = np.einsum("ij,ij->i", normals, tri[:, 0]) < 0.0
    normals[flip] *= -1.0
    radii = np.max(np.stack([_angles(normals, tri[:, k]) for k in range(3)]), axis=0)
    # Empty-cap check: no point may be closer to a facet circumcentre than the facet's own vertices.
    nearest = np.arccos(np.clip(np.max(normals @ u.T, axis=1), -1.0, 1.0))
    violation = float(np.max(radii - nearest))
    require(violation <= EMPTY_CAP_TOLERANCE,
            f"numerically inconsistent hull (a facet cap contains another point by {violation:.3e} rad); perturb the configuration")
    worst = int(np.argmax(radii))
    value = math.degrees(float(radii[worst]))
    details = {
        "n": n,
        "covering_radius_degrees": value,
        "covering_radius_radians": float(radii[worst]),
        "facets": int(len(hull.simplices)),
        "deepest_hole_direction": [float(v) for v in normals[worst]],
        "hull_origin_margin": margin,
        "empty_cap_violation": max(violation, 0.0),
        "min_chord": float(chords.min()),
        "max_length_deviation": deviation,
    }
    details.update(record_comparison(value, published, "minimize", COVERING_MARGIN_DEGREES))
    return value, details


def log_energy(candidate, n, published=None):
    """Logarithmic energy sum_{i<j} -log|x_i - x_j| of the normalised configuration; minimize."""
    u, deviation = unit_vectors(candidate, n)
    require(n >= 2, "the energy needs at least two points")
    chords = chord_lengths(u)
    closest = float(chords.min())
    require(closest >= MIN_SEPARATION, f"points must be distinct (closest chord {closest:.3e})")
    value = math.fsum((-np.log(chords)).tolist())
    details = {
        "n": n,
        "log_energy": value,
        "normalized_pair_energy": value / (n * (n - 1) / 2.0),
        "min_chord": closest,
        "max_chord": float(chords.max()),
        "max_length_deviation": deviation,
    }
    details.update(record_comparison(value, published, "minimize", LOG_ENERGY_MARGIN))
    return value, details
