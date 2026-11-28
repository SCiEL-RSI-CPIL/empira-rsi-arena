"""Sixteen exploratory-landscape-analysis features (local implementation).

Feature groups follow Mersmann et al. (2011) and Kerschke and Trautmann (2019):
y-distribution, linear/quadratic meta-models, local rank correlation, dispersion,
nearest-better clustering, information content and fitness-distance correlation. Every feature
is invariant to adding a constant to y and to multiplying y by a positive
constant, so a function's bias and scale do not matter. The definitions are this
suite's own and are not numerically identical to flacco/pflacco.
"""
from __future__ import annotations

import math

import numpy as np

NAMES = ("distr_skewness", "distr_kurtosis", "meta_lin_r2", "meta_quad_r2", "meta_quad_cond", "meta_interact_r2",
         "meta_full_quad_r2", "local_rank_cor", "disp_ratio_02", "disp_ratio_10", "disp_ratio_25", "nbc_sd_ratio",
         "nbc_mean_ratio", "nbc_fitness_cor", "ic_h_max", "fdc")
STD_FLOOR = 0.01


def sample(dim, lb, ub, n, seed):
    """Latin-hypercube sample in [lb, ub]^dim (fixed seed)."""
    rng = np.random.default_rng(seed)
    cells = np.stack([rng.permutation(n) for _ in range(dim)], axis=1)
    return lb + (ub - lb) * (cells + rng.random((n, dim))) / n


def _adj_r2(design, y):
    n, p = design.shape
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot
    return 1.0 - (1.0 - r2) * (n - 1) / max(n - p, 1), coef


def _slog(v):
    """Signed log1p, which keeps heavy-tailed moments on a comparable scale."""
    return math.copysign(math.log1p(abs(v)), v)


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float(np.sum(ra ** 2) * np.sum(rb ** 2)))
    return float(np.sum(ra * rb) / denom) if denom > 0 else 0.0


def _information_content(Xn, y, rng_seed=7):
    """Max entropy H(eps) of the up/neutral/down symbol sequence along a nearest-neighbour tour."""
    n = len(y)
    dist = np.linalg.norm(Xn[:, None, :] - Xn[None, :, :], axis=2)
    visited = np.zeros(n, dtype=bool)
    current = int(np.random.default_rng(rng_seed).integers(0, n))
    order = [current]
    visited[current] = True
    for _ in range(n - 1):
        row = np.where(visited, np.inf, dist[current])
        current = int(np.argmin(row))
        visited[current] = True
        order.append(current)
    order = np.array(order)
    steps = np.linalg.norm(np.diff(Xn[order], axis=0), axis=1)
    psi = np.diff(y[order]) / np.maximum(steps, 1e-12)
    top = float(np.max(np.abs(psi)))
    if top == 0:
        return 0.0
    best = 0.0
    for eps in np.concatenate(([0.0], top * np.logspace(-5, 0, 51))):
        s = np.where(psi < -eps, 0, np.where(psi > eps, 2, 1))
        pairs = s[:-1] * 3 + s[1:]
        counts = np.bincount(pairs, minlength=9).astype(float)
        counts[[0, 4, 8]] = 0.0  # only changes of symbol contribute (Munoz et al., 2015)
        p = counts / max(len(pairs), 1)
        p = p[p > 0]
        best = max(best, float(-np.sum(p * np.log(p) / math.log(6))))
    return best


def features(X, y, lb, ub):
    """Return the 16 features (float array, order NAMES) of a sample X (n x d) with values y (n,)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, d = X.shape
    span = float(np.ptp(y))
    yn = (y - y.min()) / span
    Xn = (X - lb) / (ub - lb)
    centred = yn - yn.mean()
    sd = float(np.sqrt(np.mean(centred ** 2)))
    skew = _slog(float(np.mean(centred ** 3) / sd ** 3))
    kurt = _slog(float(np.mean(centred ** 4) / sd ** 4 - 3.0))
    ones = np.ones((n, 1))
    lin_r2, _ = _adj_r2(np.hstack([ones, Xn]), yn)
    quad_r2, quad_coef = _adj_r2(np.hstack([ones, Xn, Xn ** 2]), yn)
    q = np.abs(quad_coef[1 + d:])
    quad_cond = float(np.log10((q.max() + 1e-12) / (q.min() + 1e-12)))
    iu = np.triu_indices(d, 1)
    products = (Xn[:, :, None] * Xn[:, None, :])[:, iu[0], iu[1]]
    inter_r2, _ = _adj_r2(np.hstack([ones, Xn, products]), yn)
    full_r2, _ = _adj_r2(np.hstack([ones, Xn, Xn ** 2, products]), yn)
    dist = np.linalg.norm(Xn[:, None, :] - Xn[None, :, :], axis=2)
    mean_all = float(dist[np.triu_indices(n, 1)].mean())
    order = np.argsort(y, kind="stable")
    disp = []
    for frac in (0.02, 0.10, 0.25):
        k = max(2, int(round(frac * n)))
        sub = dist[np.ix_(order[:k], order[:k])]
        disp.append(float(sub[np.triu_indices(k, 1)].mean() / mean_all))
    off = dist + np.diag(np.full(n, np.inf))
    nn = off.min(axis=1)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    better = rank[None, :] < rank[:, None]
    nb = np.where(better, off, np.inf).min(axis=1)
    has = np.isfinite(nb)
    nbc_sd = float(np.std(nn[has]) / max(np.std(nb[has]), 1e-12))
    nbc_mean = float(np.mean(nn[has]) / max(np.mean(nb[has]), 1e-12))
    nbc_cor = _spearman(nb[has], y[has])
    ranks = rank / (n - 1.0)
    neighbours = np.argsort(off, axis=1)[:, :5]
    local = float(np.corrcoef(ranks, ranks[neighbours].mean(axis=1))[0, 1])
    ic = _information_content(Xn, yn)
    fdc = _spearman(dist[order[0]], y)
    return np.array([skew, kurt, lin_r2, quad_r2, quad_cond, inter_r2, full_r2, local, *disp,
                     nbc_sd, nbc_mean, nbc_cor, ic, fdc])
