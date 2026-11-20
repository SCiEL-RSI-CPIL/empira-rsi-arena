"""Prepared data for the software-engineering tasks (written by scripts/prepare_software.py).

Every data path of the software_* evaluators is resolved through ``data_dir()``.
Evaluators never use the network; a missing file is reported with the command
that prepares it.
"""
from __future__ import annotations

import json

from arena_task_paths import data_root
from .runtime import CandidateError  # noqa: F401 - re-exported for evaluator modules


def data_dir():
    return data_root() / "software"


class MissingData(RuntimeError):
    """Raised when prepared data is absent (an installation problem, not a candidate error)."""


def path(name):
    target = data_dir() / name
    if not target.exists():
        raise MissingData(f"missing prepared data {target}; run python3 scripts/prepare_software.py")
    return target


def load_json(name):
    return json.loads(path(name).read_text(encoding="utf-8"))
