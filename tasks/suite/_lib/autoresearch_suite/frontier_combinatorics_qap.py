"""Quadratic assignment on fixed QAPLIB instances, scored exactly.

The instance file is read from the prepared data directory and its SHA-256
digest is checked against the value pinned in the task parameters. The QAPLIB
convention is used: the first matrix A and the second matrix B of the file,
and the cost of a permutation p (facility i placed at location p[i]) is
sum_{i,j} A[i, j] * B[p[i], p[j]], computed with exact integer arithmetic.
"""
from __future__ import annotations

import hashlib

import numpy as np

from arena_task_paths import data_root
from .frontier_combinatorics_designs import integer, sequence
from .runtime import require


def data_dir():
    return data_root() / "frontier_combinatorics"


def instance(name, sha256):
    """Return (A, B) for a pinned QAPLIB instance, verifying the file digest."""
    path = data_dir() / f"{name}.dat"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run python3 scripts/prepare_frontier_combinatorics.py")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path.name} has SHA-256 {digest}, expected {sha256}")
    return parse(raw.decode("ascii"))


def parse(text):
    """Parse the QAPLIB format: n, then the n x n matrices A and B."""
    tokens = text.split()
    n = int(tokens[0])
    values = np.array([int(x) for x in tokens[1:]], dtype=np.int64)
    if len(values) != 2 * n * n:
        raise ValueError(f"expected {2 * n * n} matrix entries, found {len(values)}")
    return values[:n * n].reshape(n, n), values[n * n:].reshape(n, n)


def objective(permutation, flow, distance):
    """Validate a permutation of 0..n-1 and return its exact QAP cost."""
    n = len(flow)
    permutation = sequence(permutation, "permutation")
    require(len(permutation) == n, f"return a permutation of length n = {n}")
    p = [integer(x, "location") for x in permutation]
    require(sorted(p) == list(range(n)), f"the assignment must be a permutation of 0..{n - 1}")
    # int64 is exact here: n^2 * max|A| * max|B| stays far below 2^63 for the pinned instances.
    bound = n * n * int(np.abs(flow).max()) * int(np.abs(distance).max())
    require(bound < 2 ** 62, "instance magnitudes exceed the exact int64 range")
    index = np.asarray(p, dtype=np.int64)
    cost = int((flow * distance[np.ix_(index, index)]).sum())
    return cost, {"cost": cost, "n": n, "verification": "exact_integer_objective"}
