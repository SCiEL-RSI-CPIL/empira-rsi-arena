"""Quadratic assignment on QAPLIB instances; same contract and exact objective as the existing QAPLIB tasks.

Records and lower bounds: the maintained QAPLIB table (scripts/registry_misc.py says where one value is later than QAPLIB's).
"""
from __future__ import annotations

from .common import read_data  # importing common first puts the old suite library on sys.path
from autoresearch_suite import frontier_combinatorics_qap as _old  # noqa: E402


def instance(name):
    return _old.parse(read_data("qap", name + ".dat").decode("ascii"))


objective = _old.objective
