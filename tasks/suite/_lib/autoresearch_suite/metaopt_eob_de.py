"""Differential-evolution members of the EoB algorithm pool (local implementations).

Written from the published update rules: DE/rand/1/bin (Storn and Price, 1997),
jDE (Brest et al., 2006), JADE (Zhang and Sanderson, 2009) and L-SHADE (Tanabe
and Fukunaga, 2014). They stand in for the paper's Vanilla DE, JDE21, MadDE and
NL-SHADE-LBC and are not those exact algorithms. Every optimizer takes a
``Budget`` (vectorised objective with an evaluation cap) and a NumPy Generator,
clips trial points to the box and returns nothing; the budget keeps the best value.
"""
from __future__ import annotations

import math

import numpy as np


class Budget:
    """Vectorised objective with a hard evaluation cap; rows beyond the cap get +inf and are not evaluated."""

    def __init__(self, function, dim, lb, ub, max_evals):
        self.function, self.dim, self.lb, self.ub, self.max_evals = function, dim, lb, ub, max_evals
        self.used, self.best = 0, math.inf

    @property
    def remaining(self):
        return self.max_evals - self.used

    def __call__(self, X):
        X = np.clip(np.asarray(X, dtype=float).reshape(-1, self.dim), self.lb, self.ub)
        out = np.full(len(X), np.inf)
        n = min(len(X), max(self.remaining, 0))
        if n:
            values = self.function(X[:n])
            out[:n] = values
            self.used += n
            self.best = min(self.best, float(np.min(values)))
        return out


def _distinct(rng, n, k, exclude):
    """k distinct indices in range(n) per row, all different from exclude[row]."""
    rows = np.empty((len(exclude), k), dtype=int)
    for i, e in enumerate(exclude):
        pool = rng.permutation(n)
        rows[i] = pool[pool != e][:k]
    return rows


def _binomial(rng, x, v, cr):
    n, d = x.shape
    mask = rng.random((n, d)) < np.reshape(cr, (-1, 1))
    mask[np.arange(n), rng.integers(0, d, n)] = True
    return np.where(mask, v, x)


def de_rand1_bin(budget, rng, pop_size=50, F=0.5, CR=0.9):
    lb, ub, d = budget.lb, budget.ub, budget.dim
    X = rng.uniform(lb, ub, (pop_size, d))
    fx = budget(X)
    while budget.remaining > 0:
        r = _distinct(rng, pop_size, 3, np.arange(pop_size))
        V = X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]])
        U = np.clip(_binomial(rng, X, V, CR), lb, ub)
        fu = budget(U)
        better = fu <= fx
        X[better], fx[better] = U[better], fu[better]


def jde(budget, rng, pop_size=50, tau1=0.1, tau2=0.1):
    lb, ub, d = budget.lb, budget.ub, budget.dim
    X = rng.uniform(lb, ub, (pop_size, d))
    fx = budget(X)
    Fs, CRs = np.full(pop_size, 0.5), np.full(pop_size, 0.9)
    while budget.remaining > 0:
        newF = np.where(rng.random(pop_size) < tau1, 0.1 + 0.9 * rng.random(pop_size), Fs)
        newCR = np.where(rng.random(pop_size) < tau2, rng.random(pop_size), CRs)
        r = _distinct(rng, pop_size, 3, np.arange(pop_size))
        V = X[r[:, 0]] + newF[:, None] * (X[r[:, 1]] - X[r[:, 2]])
        U = np.clip(_binomial(rng, X, V, newCR), lb, ub)
        fu = budget(U)
        better = fu <= fx
        X[better], fx[better], Fs[better], CRs[better] = U[better], fu[better], newF[better], newCR[better]


def _cauchy_f(rng, loc):
    """Cauchy(loc, 0.1) scale factors, regenerated while non-positive, truncated at 1."""
    loc = np.asarray(loc, dtype=float)
    F = loc + 0.1 * np.tan(np.pi * (rng.random(loc.shape) - 0.5))
    bad = F <= 0
    while bad.any():
        F[bad] = loc[bad] + 0.1 * np.tan(np.pi * (rng.random(int(bad.sum())) - 0.5))
        bad = F <= 0
    return np.minimum(F, 1.0)


def _pbest_mutation(rng, X, fx, archive, F, p):
    n = len(X)
    top = max(2, int(round(p * n)))
    order = np.argsort(fx)
    pbest = X[order[rng.integers(0, top, n)]]
    r1 = _distinct(rng, n, 1, np.arange(n))[:, 0]
    union = np.vstack([X, archive]) if len(archive) else X
    r2 = np.empty(n, dtype=int)
    for i in range(n):
        while True:
            j = int(rng.integers(0, len(union)))
            if j != i and j != r1[i]:
                break
        r2[i] = j
    return X + F[:, None] * (pbest - X) + F[:, None] * (X[r1] - union[r2])


def jade(budget, rng, pop_size=50, p=0.05, c=0.1):
    lb, ub, d = budget.lb, budget.ub, budget.dim
    X = rng.uniform(lb, ub, (pop_size, d))
    fx = budget(X)
    mu_f, mu_cr, archive = 0.5, 0.5, np.empty((0, d))
    while budget.remaining > 0:
        CR = np.clip(rng.normal(mu_cr, 0.1, pop_size), 0, 1)
        F = _cauchy_f(rng, np.full(pop_size, mu_f))
        V = _pbest_mutation(rng, X, fx, archive, F, p)
        U = np.clip(_binomial(rng, X, V, CR), lb, ub)
        fu = budget(U)
        better = fu < fx
        if better.any():
            archive = np.vstack([archive, X[better]])
            if len(archive) > pop_size:
                archive = archive[rng.permutation(len(archive))[:pop_size]]
            mu_cr = (1 - c) * mu_cr + c * float(np.mean(CR[better]))
            mu_f = (1 - c) * mu_f + c * float(np.sum(F[better] ** 2) / np.sum(F[better]))
        keep = fu <= fx
        X[keep], fx[keep] = U[keep], fu[keep]


def lshade(budget, rng, init_factor=18, min_size=4, memory=6, p=0.11, arc_rate=2.6):
    lb, ub, d = budget.lb, budget.ub, budget.dim
    n_init = init_factor * d
    X = rng.uniform(lb, ub, (n_init, d))
    fx = budget(X)
    m_f, m_cr, k = np.full(memory, 0.5), np.full(memory, 0.5), 0
    archive = np.empty((0, d))
    max_evals = budget.max_evals
    while budget.remaining > 0:
        n = len(X)
        idx = rng.integers(0, memory, n)
        CR = np.where(np.isnan(m_cr[idx]), 0.0, np.clip(rng.normal(np.nan_to_num(m_cr[idx]), 0.1), 0, 1))
        F = _cauchy_f(rng, m_f[idx])
        V = _pbest_mutation(rng, X, fx, archive, F, p)
        U = np.clip(_binomial(rng, X, V, CR), lb, ub)
        fu = budget(U)
        better = fu < fx
        if better.any():
            archive = np.vstack([archive, X[better]])
            cap = int(round(arc_rate * n))
            if len(archive) > cap:
                archive = archive[rng.permutation(len(archive))[:cap]]
            w = (fx[better] - fu[better])
            w = w / w.sum() if np.isfinite(w.sum()) and w.sum() > 0 else np.full(better.sum(), 1.0 / better.sum())
            sf, scr = F[better], CR[better]
            m_f[k] = float(np.sum(w * sf ** 2) / max(np.sum(w * sf), 1e-12))
            m_cr[k] = np.nan if np.isnan(m_cr[k]) or np.max(scr) == 0 else float(np.sum(w * scr ** 2) / max(np.sum(w * scr), 1e-12))
            k = (k + 1) % memory
        keep = fu <= fx
        X[keep], fx[keep] = U[keep], fu[keep]
        target = int(round((min_size - n_init) / max_evals * budget.used + n_init))
        if target < n:
            order = np.argsort(fx)[:max(target, min_size)]
            X, fx = X[order], fx[order]
            cap = int(round(arc_rate * len(X)))
            if len(archive) > cap:
                archive = archive[rng.permutation(len(archive))[:cap]]


ALGORITHMS = {"de_rand1_bin": de_rand1_bin, "jde": jde, "jade": jade, "lshade": lshade}
