"""Maximum Nash welfare allocation of indivisible goods (Caragiannis et al.; Cole and Gkatzelis).

Spliddit-style instances: every agent splits 1000 points over the goods
(additive valuations). An allocation gives each good to one agent; its Nash
welfare is the geometric mean of the agents' utilities (zero if any agent gets
nothing). The score is the mean ratio to the proven optimum, which is found by
the integer-utility mixed-integer program of Caragiannis et al. (exact because
utilities are integers).
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math

import numpy as np

from . import finance_sandbox
from .runtime import CandidateError

ENTRY = "allocate"
POINTS = 1000
SIZES = [(4, 16, 101), (5, 20, 102), (6, 24, 103), (8, 32, 104), (10, 40, 105), (12, 48, 106), (15, 60, 107), (15, 60, 108)]
# SHA-256 of the generated valuation matrices (guards against a changed random stream).
INSTANCE_SHA256 = {
    0: "298032a7ecd59566d68525ddb0c70b0c0096c3978bb3f45ecd8e257a2519f470",
    1: "78d62d9423f953f9b5244f2205901eef80fea8f1c4b1eaf2d837b1c56c40b727",
    2: "863f5b292a31dd6787b1ecb62d9e9e257360b38a8fc8e57e4291873530d7724a",
    3: "e71c52e76f86de1552058678f7a7839e9db9cc5c243738a60adaf87bfe2613aa",
    4: "799de88e94c4bd4e3e904f1b35532139fd19d152b4b72e348ec95b2c25e6d00d",
    5: "8112a050c842a3dfa36d417aadab0776b9981764db7ae8c6ffdfd4aca94abfde",
    6: "b60175a40879b7f4d36135fee6d7d683724100a7322f783c5283ec2b3a727a8a",
    7: "b7f40a89bed20b770f4cad7292ffdae95506f5e755515e067a0ca16cf045d2b1",
}
# Proven optimal Nash welfare (geometric mean utility) per instance; re-checked by tests/test_finance_fairdiv.py.
OPTIMUM = {
    0: 484.59095976159176,
    1: 343.4006415761669,
    2: 347.63336598907983,
    3: 279.6722677232401,
    4: 230.44700446829972,
    5: 186.89403527561217,
    6: 176.99994793012144,
    7: 162.66996335540532,
}


def generate(n, m, seed):
    """Integer valuations (n x m) whose rows sum to POINTS: common good quality times idiosyncratic taste."""
    rng = np.random.RandomState(seed)  # legacy generator: stream frozen across NumPy versions
    quality = rng.lognormal(0.0, 0.8, m)
    raw = quality[None, :] * rng.lognormal(0.0, 0.6, (n, m)) * (rng.random_sample((n, m)) > 0.15)
    raw[:, 0] += 1e-9  # every agent values something
    share = POINTS * raw / raw.sum(axis=1, keepdims=True)
    values = np.floor(share).astype(np.int64)
    for i in range(n):  # largest remainders so that every row sums exactly to POINTS
        extra = POINTS - int(values[i].sum())
        values[i, np.argsort(-(share[i] - values[i]), kind="stable")[:extra]] += 1
    return values


@lru_cache(maxsize=None)
def valuations(index):
    n, m, seed = SIZES[index]
    values = generate(n, m, seed)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    if digest != INSTANCE_SHA256[index]:
        raise RuntimeError(f"fair-division instance {index} changed (SHA-256 {digest}); the NumPy random stream differs")
    values.setflags(write=False)
    return values


def utilities(values, owner):
    n, m = values.shape
    return np.array([int(values[i, owner == i].sum()) for i in range(n)])


def nash_welfare(u):
    return 0.0 if (u <= 0).any() else math.exp(float(np.mean(np.log(u))))


def envy_free_up_to_one(values, owner):
    """EF1: any envy disappears after removing some single good from the envied bundle."""
    n = values.shape[0]
    for i in range(n):
        mine = values[i, owner == i].sum()
        for j in range(n):
            bundle = values[i, owner == j]
            if j != i and bundle.size and bundle.sum() - bundle.max() > mine:
                return False
    return True


def optimal_allocation(values):
    """Exact MNW by the integer-utility MILP (log replaced by its secants between consecutive integers)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    n, m = values.shape
    nx = n * m
    slopes = np.log(np.arange(2, POINTS + 1)) - np.log(np.arange(1, POINTS))
    rows, cols, data, upper = [], [], [], []
    r = 0
    for g in range(m):
        rows += [r] * n
        cols += [i * m + g for i in range(n)]
        data += [1.0] * n
        r += 1
    assign_rows = r
    for i in range(n):
        for k, s in enumerate(slopes, start=1):  # z_i - s*u_i <= log k - s*k
            rows += [r] * (m + 1)
            cols += [i * m + g for g in range(m)] + [nx + i]
            data += list(-s * values[i]) + [1.0]
            upper.append(math.log(k) - s * k)
            r += 1
    matrix = coo_matrix((data, (rows, cols)), shape=(r, nx + n)).tocsr()
    lower = np.r_[np.ones(assign_rows), np.full(r - assign_rows, -np.inf)]
    upper = np.r_[np.ones(assign_rows), upper]
    positive = coo_matrix((values.ravel().astype(float), (np.repeat(np.arange(n), m), np.arange(nx))), shape=(n, nx + n))
    result = milp(np.r_[np.zeros(nx), -np.ones(n)], integrality=np.r_[np.ones(nx), np.zeros(n)],
                  constraints=[LinearConstraint(matrix, lower, upper), LinearConstraint(positive, 1, np.inf)],
                  bounds=Bounds(np.r_[np.zeros(nx), np.full(n, -1.0)], np.r_[np.ones(nx), np.full(n, math.log(POINTS) + 1)]),
                  options={"mip_rel_gap": 1e-12})
    if result.status != 0:
        raise RuntimeError(f"MNW reference solve failed: {result.message}")
    owner = np.argmax(result.x[:nx].reshape(n, m), axis=0)
    return owner, nash_welfare(utilities(values, owner))


def validate(raw, n, m):
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)) or len(raw) != m:
        raise CandidateError(f"allocate must return {m} agent indices, one per good")
    owner = []
    for agent in raw:
        if isinstance(agent, (bool, np.bool_)) or not isinstance(agent, (int, np.integer)) or not 0 <= int(agent) < n:
            raise CandidateError(f"every good needs an integer agent index in [0, {n})")
        owner.append(int(agent))
    return np.array(owner)


def allocations():
    """Scoring loop: one candidate call per instance; the optimal welfare stays in this process."""
    owners = []
    for index, (n, m, _) in enumerate(SIZES):
        owners.append(validate((yield finance_sandbox.step(ENTRY, valuations(index).copy())), n, m))
    return owners


def fair_division_score(candidate):
    """Mean of Nash welfare / optimal Nash welfare over the fixed instances (maximize; 1 is optimal).

    ``candidate`` is the candidate module (its allocate runs in the sandbox) or a trusted callable.
    """
    rows, ratios = [], []
    owners = finance_sandbox.run(candidate, allocations(), ENTRY)
    for index, (n, m, _) in enumerate(SIZES):
        values = valuations(index)
        owner = owners[index]
        u = utilities(values, owner)
        ratio = nash_welfare(u) / OPTIMUM[index]
        ratios.append(ratio)
        rows.append({"agents": n, "goods": m, "nash_welfare": nash_welfare(u), "optimum": OPTIMUM[index], "ratio": ratio,
                     "min_utility": int(u.min()), "ef1": envy_free_up_to_one(values, owner)})
    return float(np.mean(ratios)), {"instances": rows, "ef1_instances": sum(r["ef1"] for r in rows)}
