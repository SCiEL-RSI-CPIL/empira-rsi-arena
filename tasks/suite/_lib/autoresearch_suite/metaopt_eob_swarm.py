"""Swarm and evolution-strategy members of the EoB algorithm pool (local implementations).

Written from the published update rules: global-best PSO with constriction
coefficients (Clerc and Kennedy, 2002), ring-topology local-best PSO (Kennedy and
Mendes, 2002), CLPSO (Liang et al., 2006), DMS-PSO without its local search
(Liang and Suganthan, 2005), CMA-ES (Hansen, The CMA Evolution Strategy: A
Tutorial, 2016) and sep-CMA-ES (Ros and Hansen, 2008). The PSO variants stand in
for the paper's PSO, GLPSO, sDMS-PSO and SAHLPSO and are not those exact
algorithms. Same interface as ``metaopt_eob_de``.
"""
from __future__ import annotations

import math

import numpy as np

CHI, PHI = 0.7298, 1.49618


def _swarm(budget, rng, n):
    lb, ub, d = budget.lb, budget.ub, budget.dim
    vmax = 0.2 * (ub - lb)
    X = rng.uniform(lb, ub, (n, d))
    V = rng.uniform(-vmax, vmax, (n, d))
    fx = budget(X)
    return X, V, fx, X.copy(), fx.copy(), vmax


def _move(budget, X, V, vmax):
    V = np.clip(V, -vmax, vmax)
    X = np.clip(X + V, budget.lb, budget.ub)
    return X, V


def pso_gbest(budget, rng, n=40):
    X, V, fx, P, fp, vmax = _swarm(budget, rng, n)
    while budget.remaining > 0:
        g = P[np.argmin(fp)]
        V = CHI * V + PHI * rng.random(X.shape) * (P - X) + PHI * rng.random(X.shape) * (g - X)
        X, V = _move(budget, X, V, vmax)
        fx = budget(X)
        better = fx < fp
        P[better], fp[better] = X[better], fx[better]


def pso_ring(budget, rng, n=40):
    X, V, fx, P, fp, vmax = _swarm(budget, rng, n)
    idx = np.arange(n)
    neighbours = np.stack([(idx - 1) % n, idx, (idx + 1) % n], axis=1)
    while budget.remaining > 0:
        local = P[neighbours[idx, np.argmin(fp[neighbours], axis=1)]]
        V = CHI * V + PHI * rng.random(X.shape) * (P - X) + PHI * rng.random(X.shape) * (local - X)
        X, V = _move(budget, X, V, vmax)
        fx = budget(X)
        better = fx < fp
        P[better], fp[better] = X[better], fx[better]


def clpso(budget, rng, n=40, gap=7, c=1.49445, w0=0.9, w1=0.2):
    X, V, fx, P, fp, vmax = _swarm(budget, rng, n)
    d = budget.dim
    pc = 0.05 + 0.45 * (np.exp(10 * np.arange(n) / (n - 1)) - 1) / (math.exp(10) - 1)
    stale = np.zeros(n, dtype=int)

    def exemplar(i):
        a, b = rng.integers(0, n, (2, d))
        tournament = np.where(fp[a] < fp[b], a, b)
        source = np.where(rng.random(d) < pc[i], tournament, i)
        if np.all(source == i):
            source[rng.integers(0, d)] = tournament[rng.integers(0, d)]
        return source

    sources = np.stack([exemplar(i) for i in range(n)])
    start = budget.remaining
    while budget.remaining > 0:
        w = w0 - (w0 - w1) * (start - budget.remaining) / start
        E = P[sources, np.arange(d)[None, :]]
        V = w * V + c * rng.random(X.shape) * (E - X)
        X, V = _move(budget, X, V, vmax)
        fx = budget(X)
        better = fx < fp
        P[better], fp[better] = X[better], fx[better]
        stale = np.where(better, 0, stale + 1)
        for i in np.flatnonzero(stale >= gap):
            sources[i], stale[i] = exemplar(i), 0


def dms_pso(budget, rng, n_swarms=10, size=3, regroup=5, c=1.49445, w=0.729):
    n = n_swarms * size
    X, V, fx, P, fp, vmax = _swarm(budget, rng, n)
    groups = rng.permutation(n).reshape(n_swarms, size)
    generation = 0
    while budget.remaining > 0:
        member = np.empty(n, dtype=int)
        for g in groups:
            member[g] = g[np.argmin(fp[g])]
        V = w * V + c * rng.random(X.shape) * (P - X) + c * rng.random(X.shape) * (P[member] - X)
        X, V = _move(budget, X, V, vmax)
        fx = budget(X)
        better = fx < fp
        P[better], fp[better] = X[better], fx[better]
        generation += 1
        if generation % regroup == 0:
            groups = rng.permutation(n).reshape(n_swarms, size)


def _cma_constants(d):
    lam = 4 + int(3 * math.log(d))
    mu = lam // 2
    w = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
    cs = (mueff + 2) / (d + mueff + 5)
    c1 = 2 / ((d + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (d + 1)) - 1) + cs
    chi_n = math.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))
    return lam, mu, w, mueff, cc, cs, c1, cmu, damps, chi_n


def cma_es(budget, rng, separable=False):
    d, lb, ub = budget.dim, budget.lb, budget.ub
    lam, mu, w, mueff, cc, cs, c1, cmu, damps, chi_n = _cma_constants(d)
    if separable:  # Ros and Hansen (2008): larger learning rates for the diagonal model
        c1, cmu = c1 * (d + 2) / 3, min(1 - c1 * (d + 2) / 3, cmu * (d + 2) / 3)
    mean = rng.uniform(lb, ub, d)
    sigma = 0.3 * (ub - lb)
    pc, ps = np.zeros(d), np.zeros(d)
    diag = np.ones(d)
    C, B, D = np.eye(d), np.eye(d), np.ones(d)
    generation = 0
    while budget.remaining > 0:
        Z = rng.standard_normal((lam, d))
        Y = Z * np.sqrt(diag) if separable else (Z * D) @ B.T
        X = np.clip(mean + sigma * Y, lb, ub)
        Y = (X - mean) / sigma  # repaired steps enter the update
        f = budget(X)
        order = np.argsort(f)[:mu]
        y_w = w @ Y[order]
        mean = mean + sigma * y_w
        inv_sqrt = (1 / np.sqrt(diag)) * y_w if separable else B @ ((B.T @ y_w) / D)
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * inv_sqrt
        generation += 1
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * generation)) / chi_n < 1.4 + 2 / (d + 1)
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w
        Yo = Y[order]
        if separable:
            diag = (1 - c1 - cmu) * diag + c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * diag) + cmu * (w @ Yo ** 2)
            diag = np.maximum(diag, 1e-20)
        else:
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (Yo.T * w) @ Yo
            C = (C + C.T) / 2
            vals, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(vals, 1e-20))
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
        sigma = min(max(sigma, 1e-12), 10 * (ub - lb))
        if not np.all(np.isfinite(mean)):
            mean = rng.uniform(lb, ub, d)


def sep_cma_es(budget, rng):
    cma_es(budget, rng, separable=True)


ALGORITHMS = {"pso_gbest": pso_gbest, "pso_ring": pso_ring, "clpso": clpso, "dms_pso": dms_pso,
              "cma_es": cma_es, "sep_cma_es": sep_cma_es}
