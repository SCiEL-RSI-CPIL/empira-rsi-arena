"""Two-stream closures for solar radiative transfer, scored against an accurate multi-stream solution.

A homogeneous plane-parallel layer (optical depth tau, single-scattering albedo omega,
Henyey-Greenstein phase function with asymmetry g) over a black surface is lit by a collimated
beam with cosine zenith angle mu0. Two-stream schemes (Meador and Weaver 1980) approximate the
diffuse fluxes with coefficients gamma1, gamma2, gamma3 (and gamma4 = 1 - gamma3), optionally
after delta-scaling with a forward-peak fraction f. The candidate supplies these coefficients as
a function of (omega, g, mu0) only; the evaluator solves the two-stream equations exactly and
compares the layer reflectance and total transmittance with a 96-stream discrete-ordinate
solution (delta-M scaled, double-Gauss quadrature) computed by ``derive``.
"""
from __future__ import annotations

import numpy as np

from . import earth_common as ec
from . import earth_sandbox
from .runtime import require

RAW_SOURCES = ()
DATA = "twostream/reference_flux_cases.json.gz"
DERIVED = {DATA: "03bf0bcd7b8c5751ebe0ce41680ae9f65fcecfbd89dc2fa64086c7f049f5b430"}
CASES, SEED = 12000, 20261010
STREAMS = 48  # per hemisphere
OMEGA_CAP = 1.0 - 1e-10  # conservative scattering is evaluated at this albedo by both solvers
K_FLOOR = 1e-6  # smallest two-stream eigenvalue used by the closed-form solution


def legendre_table(order, x):
    """P_0..P_order evaluated at x (shape (order + 1, len(x)))."""
    x = np.asarray(x, float)
    table = np.zeros((order + 1, len(x)))
    table[0] = 1.0
    if order > 0:
        table[1] = x
    for n in range(1, order):
        table[n + 1] = ((2 * n + 1) * x * table[n] - n * table[n - 1]) / (n + 1)
    return table


def discrete_ordinates(tau, omega, g, mu0, n=STREAMS):
    """Reflectance, diffuse and direct transmittance of a delta-M scaled HG layer over a black surface."""
    omega = min(omega, OMEGA_CAP)
    nodes, weights = np.polynomial.legendre.leggauss(n)
    mu, w = 0.5 * (nodes + 1.0), 0.5 * weights  # double-Gauss on [0, 1]
    order = 2 * n - 1
    f = g ** (2 * n)  # delta-M forward fraction
    moments = (g ** np.arange(order + 1) - f) / (1.0 - f)
    tau_s, omega_s = (1.0 - omega * f) * tau, omega * (1.0 - f) / (1.0 - omega * f)
    angles = np.concatenate([mu, -mu])
    weights2 = np.concatenate([w, w])
    pl = legendre_table(order, angles)
    coeffs = (2 * np.arange(order + 1) + 1) * moments
    phase = (pl.T * coeffs) @ pl  # P(mu_i, mu_j), m = 0
    beam = (pl.T * coeffs) @ legendre_table(order, [-mu0])[:, 0]
    # mu_i dI_i/dtau = I_i - (omega/2) sum_j w_j P_ij I_j - (omega/(4 pi)) P(mu_i, -mu0) exp(-tau/mu0), unit beam irradiance
    a = (np.eye(2 * n) - 0.5 * omega_s * phase * weights2) / angles[:, None]
    q = 0.25 / np.pi * omega_s * beam / angles
    shift = 1.0 / mu0
    values, vectors = np.linalg.eig(a)
    values, vectors = values.real, vectors.real
    if np.min(np.abs(values + shift)) < 1e-8:  # resonance with an eigenvalue: nudge the beam angle
        shift *= 1.0 + 1e-7
    z = np.linalg.solve(a + shift * np.eye(2 * n), q)  # particular solution Z exp(-shift tau)
    positive = values > 0
    # modes: positive eigenvalues decay away from the bottom, negative ones away from the top
    decay = np.exp(-np.abs(values) * tau_s)
    scale = np.where(positive, decay, 1.0)  # value of each mode at tau = 0
    scale_bottom = np.where(positive, 1.0, decay)  # value at tau = tau_s
    top_rows, bottom_rows = slice(n, 2 * n), slice(0, n)  # downward intensities at top, upward at bottom
    system = np.vstack([vectors[top_rows] * scale, vectors[bottom_rows] * scale_bottom])
    rhs = np.concatenate([-z[top_rows], -z[bottom_rows] * np.exp(-shift * tau_s)])
    c = np.linalg.solve(system, rhs)
    top = vectors @ (c * scale) + z
    bottom = vectors @ (c * scale_bottom) + z * np.exp(-shift * tau_s)
    up = 2 * np.pi * np.sum(w * mu * top[:n])
    down = 2 * np.pi * np.sum(w * mu * bottom[n:])
    direct = np.exp(-tau_s / mu0)
    return up / mu0, down / mu0, direct


def two_stream(tau, omega, mu0, gamma1, gamma2, gamma3):
    """Reflectance and total transmittance of the Meador-Weaver two-stream equations (vectorised)."""
    omega = np.minimum(omega, OMEGA_CAP)
    gamma4 = 1.0 - gamma3
    # conservative closures have gamma1 = |gamma2| (k = 0): raise gamma1 so that k >= 1e-6 (absorption change ~1e-10)
    k = np.sqrt(np.maximum(gamma1 ** 2 - gamma2 ** 2, K_FLOOR ** 2))
    gamma1 = np.sqrt(gamma2 ** 2 + k ** 2)
    inv_mu0 = 1.0 / mu0
    det = inv_mu0 ** 2 - k ** 2
    inv_mu0 = np.where(np.abs(det) < 1e-9 * inv_mu0 ** 2, inv_mu0 * (1.0 + 1e-6), inv_mu0)
    det = inv_mu0 ** 2 - k ** 2
    z1 = omega * (-gamma3 * (gamma1 - inv_mu0) - gamma2 * gamma4) / det
    z2 = omega * (-gamma4 * (gamma1 + inv_mu0) - gamma2 * gamma3) / det
    big_gamma = gamma2 / (gamma1 + k)
    e, d = np.exp(-k * tau), np.exp(-tau * inv_mu0)
    b = (-z2 + big_gamma * e * z1 * d) / (1.0 - big_gamma ** 2 * e ** 2)
    a = -z1 * d - big_gamma * e * b
    up_top = a * e + b * big_gamma + z1
    down_bottom = a * big_gamma + b * e + z2 * d
    return up_top / mu0, down_bottom / mu0 + d


def scheme(omega, g, mu0, f, gamma1, gamma2, gamma3, tau):
    """Delta-scale with forward fraction f, then solve the two-stream equations."""
    tau_s, omega_s = (1.0 - omega * f) * tau, omega * (1.0 - f) / (1.0 - omega * f)
    return two_stream(tau_s, omega_s, mu0, gamma1(omega_s), gamma2(omega_s), gamma3)


def classic(name, omega, g, mu0, tau):
    """Published closures (Meador and Weaver 1980, Table 1; Zdunkowski et al. 1980)."""
    f = g ** 2 if name.startswith("delta") else np.zeros_like(g)
    gs = (g - f) / (1.0 - f)
    root3 = np.sqrt(3.0)
    if name.endswith("eddington"):
        return scheme(omega, g, mu0, f, lambda w: (7 - w * (4 + 3 * gs)) / 4, lambda w: -(1 - w * (4 - 3 * gs)) / 4, (2 - 3 * gs * mu0) / 4, tau)
    if name.endswith("quadrature"):
        return scheme(omega, g, mu0, f, lambda w: root3 * (2 - w * (1 + gs)) / 2, lambda w: root3 * w * (1 - gs) / 2, (1 - root3 * gs * mu0) / 2, tau)
    return scheme(omega, g, mu0, f, lambda w: (8 - w * (5 + 3 * gs)) / 4, lambda w: 3 * w * (1 - gs) / 4, (2 - 3 * gs * mu0) / 4, tau)


def sample_cases(count=CASES, seed=SEED):
    """Random layers, rounded so that the stored cases are exact decimal values."""
    rng = np.random.default_rng(seed)
    tau = np.round(10 ** rng.uniform(-2, 2, count), 5)
    conservative = rng.random(count) < 0.25
    omega = np.where(conservative, 1.0, np.round(1.0 - 10 ** rng.uniform(-5, np.log10(0.5), count), 9))
    g = np.round(rng.uniform(0.0, 0.95, count), 6)
    mu0 = np.round(rng.uniform(0.05, 1.0, count), 6)
    return np.maximum(tau, 0.01), omega, g, mu0


def derive(paths):
    tau, omega, g, mu0 = sample_cases()
    rows = [discrete_ordinates(*case) for case in zip(tau, omega, g, mu0)]
    reflect, diffuse, direct = (np.array(c) for c in zip(*rows))
    return {DATA: {"method": f"discrete ordinates, {STREAMS} streams per hemisphere, delta-M, double-Gauss", "seed": SEED,
                   "tau": tau.tolist(), "omega": omega.tolist(), "g": g.tolist(), "mu0": mu0.tolist(),
                   "reflectance": np.round(reflect, 7).tolist(), "transmittance": np.round(diffuse + direct, 7).tolist()}}


def load():
    raw = ec.load_derived(DATA, DERIVED[DATA])
    return {key: np.array(raw[key], float) for key in ("tau", "omega", "g", "mu0", "reflectance", "transmittance")}


def errors(data, reflect, transmit):
    return np.abs(reflect - data["reflectance"]) + np.abs(transmit - data["transmittance"])


def score(module, cpu_seconds=60):
    data = load()
    unique = np.unique(np.column_stack([data["omega"], data["g"], data["mu0"]]), axis=0)
    out = earth_sandbox.call(module, "two_stream_coefficients", [unique[:, 0], unique[:, 1], unique[:, 2]], cpu_seconds=cpu_seconds)
    require(isinstance(out, (list, tuple)) and len(out) == 4, "two_stream_coefficients must return (gamma1, gamma2, gamma3, f)")
    gamma1, gamma2, gamma3, f = (ec.as_array(v, (len(unique),), name) for v, name in zip(out, ("gamma1", "gamma2", "gamma3", "f")))
    require(np.all(gamma1 > 0) and np.all(gamma1 >= np.abs(gamma2) * (1 - 1e-12)), "need gamma1 > 0 and gamma1 >= |gamma2|")
    require(np.all((gamma3 >= 0) & (gamma3 <= 1)), "gamma3 must lie in [0, 1]")
    require(np.all((f >= 0) & (f < 1)), "the delta-scaling fraction f must lie in [0, 1)")
    index = {tuple(row): i for i, row in enumerate(unique)}
    pick = np.array([index[(w, gg, m)] for w, gg, m in zip(data["omega"], data["g"], data["mu0"])])
    omega, tau, mu0 = data["omega"], data["tau"], data["mu0"]
    fs = f[pick]
    tau_s, omega_s = (1.0 - omega * fs) * tau, omega * (1.0 - fs) / (1.0 - omega * fs)
    reflect, transmit = two_stream(tau_s, omega_s, mu0, gamma1[pick], gamma2[pick], gamma3[pick])
    require(np.all(np.isfinite(reflect)) and np.all(np.isfinite(transmit)), "two-stream solution is not finite for these coefficients")
    err = errors(data, reflect, transmit)
    details = {"cases": int(len(err)), "mean_abs_reflectance_error": round(float(np.mean(np.abs(reflect - data["reflectance"]))), 6),
               "mean_abs_transmittance_error": round(float(np.mean(np.abs(transmit - data["transmittance"]))), 6),
               "max_case_error": round(float(err.max()), 6), "p95_case_error": round(float(np.quantile(err, 0.95)), 6),
               "conservative_cases_mean_error": round(float(err[omega == 1.0].mean()), 6)}
    for name in ("eddington", "quadrature", "pifm", "delta_eddington", "delta_quadrature", "delta_pifm"):
        r, t = classic(name, data["omega"], data["g"], data["mu0"], data["tau"])
        details[f"{name}_mean_error"] = round(float(np.mean(errors(data, r, t))), 6)
    return float(np.mean(err)), details
