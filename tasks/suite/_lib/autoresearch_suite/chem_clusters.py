"""Global optimisation of Morse, Sutton-Chen and Dzugutov clusters against published putative minima.

Morse clusters (reduced units, pair well depth 1, equilibrium distance 1)::

    E = sum_{i<j} exp(rho*(1 - r_ij)) * (exp(rho*(1 - r_ij)) - 2)

Sutton-Chen clusters (reduced units, energy scale 1, lattice constant 1)::

    E = sum_i [ 1/2 sum_{j != i} r_ij^-n  -  c * sqrt(rho_i) ],   rho_i = sum_{j != i} r_ij^-m

with (n, m, c) = (12, 6, 144.41) for Ag/Rh, (9, 6, 39.432) for Ni/Cu and
(10, 8, 34.408) for Au/Pt (Sutton and Chen 1990), summed over all pairs
without a cutoff, as in Doye and Wales (1998).

Dzugutov clusters (a pair potential with a maximum near 1.6 that favours
polytetrahedral order, a model of glass- and quasicrystal-forming metals)::

    V(r) = A (r^-m - B) exp(c / (r - a)) [r < a] + B exp(d / (r - b)) [r < b]

with A = 5.82, B = 1.28, a = 1.87, b = 1.94, c = 1.10, d = 0.27, m = 16.

References are the putative global minima tabulated by Doye and Wales:
Morse energies from Table 1 of "Structural consequences of the range of the
interatomic potential" (J. Chem. Soc. Faraday Trans. 93, 4233, 1997; arXiv
cond-mat/9709201, six decimals) and Sutton-Chen energies from the table of
"Global minima for transition metal clusters described by Sutton-Chen
potentials" (New J. Chem. 22, 733, 1998; arXiv cond-mat/9711038, four
decimals) and Dzugutov energies from "Global optimization and the energy
landscapes of Dzugutov clusters" (Doye, Wales and Simdyankin, Faraday Discuss.
118, 159, 2001; arXiv cond-mat/0011018, six decimals). tests/test_chem_clusters.py reproduces several of them from
constructed geometries. The Cambridge Cluster Database, which may list later
improvements, was not reachable when this collection was prepared; sizes with
improvements reported in the literature that we could read (Sutton-Chen 12-6
N=68 and 10-8 N=58, Lopez-Castro et al. 2024) are not used.
"""
from __future__ import annotations

import numpy as np

from .runtime import CandidateError, require

SUTTON_CHEN = {"12-6": (12, 6, 144.41), "9-6": (9, 6, 39.432), "10-8": (10, 8, 34.408)}
DZUGUTOV = {"A": 5.82, "B": 1.28, "a": 1.87, "b": 1.94, "c": 1.10, "d": 0.27, "m": 16.0}

# (potential parameter, number of atoms, published putative global minimum, structure noted in the source)
MORSE_INSTANCES = (
    (3.0, 80, -690.577890, "80A, C_3"),
    (6.0, 38, -157.477108, "38E, C_5v"),
    (6.0, 75, -351.472365, "75C, D_5h Marks decahedron"),
    (10.0, 38, -145.849817, "38D, O_h truncated octahedron"),
    (10.0, 50, -200.691615, "50C, C_s"),
    (10.0, 80, -344.911381, "80D, C_1"),
)
SUTTON_CHEN_INSTANCES = (
    ("12-6", 38, -35419.9804, "O_h, fcc"),
    ("12-6", 75, -73421.0521, "D_5h, decahedral"),
    ("9-6", 59, -14202.4032, "T_d, close-packed"),
    ("9-6", 80, -19560.8966, "C_s, close-packed"),
    ("10-8", 55, -14814.5225, "C_1, fcc"),
    ("10-8", 79, -21547.5979, "O_h, fcc"),
)
DZUGUTOV_INSTANCES = (
    (None, 38, -78.212691, "D_6h"),
    (None, 45, -92.914141, "C_2"),
    (None, 50, -104.366190, "C_1"),
    (None, 55, -115.796712, "C_s"),
    (None, 57, -120.619381, "T_d"),
    (None, 60, -127.515076, "C_s"),
)


def _pairs(x):
    d = x[:, None, :] - x[None, :, :]
    r = np.sqrt(np.sum(d * d, axis=-1))
    return d, r


def morse_energy_gradient(positions, rho):
    """Energy and gradient (n, 3) of a Morse cluster."""
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    d, r = _pairs(x)
    np.fill_diagonal(r, 1.0)
    e = np.exp(rho * (1.0 - r))
    np.fill_diagonal(e, 0.0)
    energy = 0.5 * float(np.sum(e * (e - 2.0)))
    dvdr = -2.0 * rho * e * (e - 1.0)
    gradient = np.sum((dvdr / r)[..., None] * d, axis=1)
    return energy, gradient


def sutton_chen_energy_gradient(positions, kind):
    """Energy and gradient (n, 3) of a Sutton-Chen cluster (kind '12-6', '9-6' or '10-8')."""
    n_exp, m_exp, c = SUTTON_CHEN[kind]
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    d, r = _pairs(x)
    np.fill_diagonal(r, np.inf)
    rep = r ** -n_exp
    att = r ** -m_exp
    density = np.sum(att, axis=1)
    if np.any(density <= 0.0):
        raise CandidateError("an isolated atom has zero electron density")
    root = np.sqrt(density)
    energy = 0.5 * float(np.sum(rep)) - c * float(np.sum(root))
    inv = 1.0 / root
    dedr = -n_exp * rep / r + 0.5 * c * m_exp * (inv[:, None] + inv[None, :]) * att / r
    gradient = np.sum((dedr / r)[..., None] * d, axis=1)
    return energy, gradient


def dzugutov_energy_gradient(positions, parameter=None):
    """Energy and gradient (n, 3) of a Dzugutov cluster (``parameter`` is ignored)."""
    p = DZUGUTOV
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    d, r = _pairs(x)
    np.fill_diagonal(r, 10.0)
    v = np.zeros_like(r)
    dv = np.zeros_like(r)
    inner = r < p["a"]
    ri = r[inner]
    ei = np.exp(p["c"] / (ri - p["a"]))
    v[inner] += p["A"] * (ri ** -p["m"] - p["B"]) * ei
    dv[inner] += p["A"] * ei * (-p["m"] * ri ** (-p["m"] - 1.0) - (ri ** -p["m"] - p["B"]) * p["c"] / (ri - p["a"]) ** 2)
    outer = r < p["b"]
    ro = r[outer]
    eo = np.exp(p["d"] / (ro - p["b"]))
    v[outer] += p["B"] * eo
    dv[outer] -= p["B"] * eo * p["d"] / (ro - p["b"]) ** 2
    np.fill_diagonal(v, 0.0)
    np.fill_diagonal(dv, 0.0)
    return 0.5 * float(np.sum(v)), np.sum((dv / r)[..., None] * d, axis=1)


ENERGY_FUNCTIONS = {"morse": morse_energy_gradient, "sutton_chen": sutton_chen_energy_gradient, "dzugutov": dzugutov_energy_gradient}


def relax(positions, potential, parameter=None, maxiter=20000):
    """Local L-BFGS relaxation (available to candidates); returns (energy, positions)."""
    from scipy.optimize import minimize
    function = ENERGY_FUNCTIONS[potential]

    def objective(flat):
        energy, gradient = function(flat.reshape(-1, 3), parameter)
        return energy, gradient.ravel()

    x0 = np.asarray(positions, dtype=float).ravel()
    result = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "gtol": 1e-10, "ftol": 1e-16})
    return float(result.fun), result.x.reshape(-1, 3)


def checked_positions(value, n, min_distance):
    try:
        x = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise CandidateError("construct() must return an (n, 3) array of numbers") from None
    require(x.shape == (n, 3), f"expected {n} atoms as rows (x, y, z); got shape {x.shape}")
    require(bool(np.all(np.isfinite(x))), "positions contain NaN or infinite values")
    _, r = _pairs(x)
    closest = float(np.min(r[np.triu_indices(n, 1)]))
    require(closest >= min_distance, f"two atoms are {closest:.4f} apart; the minimum allowed separation is {min_distance}")
    return x, closest


def cluster_set_score(construct, potential):
    """Mean energy excess over the published minima in percent of |E_published| (lower is better; < 0 beats them)."""
    if potential == "morse":
        instances = [{"potential": "morse", "rho": rho, "n": n} for rho, n, _, _ in MORSE_INSTANCES]
        references = MORSE_INSTANCES
        energy_of = lambda x, inst: morse_energy_gradient(x, inst["rho"])[0]  # noqa: E731
        min_distance, tolerance = 0.5, 1e-5
    elif potential == "sutton_chen":
        instances = [{"potential": "sutton_chen", "kind": kind, "n": n, "n_exp": SUTTON_CHEN[kind][0],
                      "m_exp": SUTTON_CHEN[kind][1], "c": SUTTON_CHEN[kind][2]} for kind, n, _, _ in SUTTON_CHEN_INSTANCES]
        references = SUTTON_CHEN_INSTANCES
        energy_of = lambda x, inst: sutton_chen_energy_gradient(x, inst["kind"])[0]  # noqa: E731
        min_distance, tolerance = 0.35, 1e-3
    elif potential == "dzugutov":
        instances = [{"potential": "dzugutov", "n": n, **DZUGUTOV} for _, n, _, _ in DZUGUTOV_INSTANCES]
        references = DZUGUTOV_INSTANCES
        energy_of = lambda x, inst: dzugutov_energy_gradient(x)[0]  # noqa: E731
        min_distance, tolerance = 0.5, 1e-5
    else:
        raise ValueError(f"unknown potential {potential}")
    rows, excess = [], []
    for instance, (_, n, reference, structure) in zip(instances, references):
        x, closest = checked_positions(construct(dict(instance)), n, min_distance)
        energy = float(energy_of(x, instance))
        gap = energy - reference
        excess.append(100.0 * gap / abs(reference))
        rows.append({**instance, "energy": energy, "published_minimum": reference, "gap": gap,
                     "excess_percent": excess[-1], "matched": bool(gap <= tolerance), "below_published": bool(gap < -tolerance),
                     "published_structure": structure, "closest_pair": closest})
    return float(np.mean(excess)), {"instances": rows, "matched": sum(r["matched"] for r in rows),
                                    "instance_count": len(rows), "match_tolerance": tolerance,
                                    "total_gap": float(sum(r["gap"] for r in rows))}
