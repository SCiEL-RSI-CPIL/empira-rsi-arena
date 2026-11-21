"""Shared helpers: finds the original suite library (for `runtime`) and the prepared-data directory.

The evaluators of this suite reuse `autoresearch_suite.runtime`, so that results have the same envelope (combined_score,
score, direction, valid, error). In Arena it sits next to this package in suite/_lib. Prepared data lives outside the
task library, in `<ARENA_TASK_CACHE_DIR>/data/arnew/<family>/` (override with ARNEW_DATA).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _find_old_lib():
    for cand in (os.environ.get("ARTASK_LIB"), str(Path(__file__).resolve().parents[1])):
        if cand and (Path(cand) / "autoresearch_suite" / "runtime.py").is_file():
            return cand
    raise ImportError("autoresearch_suite not found; set ARTASK_LIB to the suite/_lib directory")


_OLD = _find_old_lib()
if _OLD not in sys.path:
    sys.path.insert(0, _OLD)

from autoresearch_suite import runtime  # noqa: E402
from autoresearch_suite.runtime import CandidateError, require  # noqa: E402,F401


def data_dir(family):
    configured = os.environ.get("ARNEW_DATA")
    if configured:
        return Path(configured) / family
    from arena_task_paths import cache_path

    return cache_path("data", "arnew", family)


def candidate_sandbox():
    """The old suite's sandbox for candidate calls (autoresearch_suite.chem_sandbox): one call in a fresh child process that receives only its
    arguments and cannot read the evaluator's memory. The old suite hides its own data and library from that process; this adds ours (the
    held-out labels live under data/), once."""
    from autoresearch_suite import chem_sandbox

    if not getattr(chem_sandbox, "_arnew_roots_hidden", False):
        old = chem_sandbox.hidden_roots
        chem_sandbox.hidden_roots = lambda: old() + [str(data_dir("any").parent), str(Path(__file__).resolve().parent)]
        chem_sandbox._arnew_roots_hidden = True
    return chem_sandbox


def read_data(family, name):
    """Bytes of a prepared data file; checked against data/<family>/MANIFEST.json when that lists it."""
    folder = data_dir(family)
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run python3 tasks/scripts/new_open_problems/prepare_new_data.py --family {family}")
    raw = path.read_bytes()
    manifest = folder / "MANIFEST.json"
    if manifest.is_file():
        want = json.loads(manifest.read_text()).get(name)
        if want and hashlib.sha256(raw).hexdigest() != want:
            raise ValueError(f"{name} does not match the SHA-256 in {manifest}")
    return raw


def as_int_list(value, what):
    try:
        out = [int(x) for x in value]
    except (TypeError, ValueError):
        raise CandidateError(f"{what} must be a sequence of integers") from None
    return out
