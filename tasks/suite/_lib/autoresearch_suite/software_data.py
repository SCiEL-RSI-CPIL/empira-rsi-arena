"""Prepared data for the software-engineering tasks (written by scripts/prepare_software.py).

Every data path of the software_* evaluators is resolved through ``data_dir()``.
Evaluators never use the network; a missing file is reported with the command
that prepares it.
"""
from __future__ import annotations

import json
import os
import time

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


class Unfinished(RuntimeError):
    """Raised when a derivation stops at its time limit; its finished steps are kept and the next run continues."""


class Steps:
    """Runs the named steps of a long derivation and keeps each result, so that the work can be done in several runs.

    ``steps(key, compute)`` returns the stored result of the step, or computes and stores it. With a ``directory``
    the results are JSON files that a later run reads back; with a ``deadline`` (a time.monotonic() value) no new
    step starts after it and ``Unfinished`` is raised instead. Every result passes through JSON, so a derivation
    gives the same bytes whether it ran in one piece or in several."""

    def __init__(self, directory=None, deadline=None):
        self.directory, self.deadline, self.done = directory, deadline, 0

    def __call__(self, key, compute):
        target = None if self.directory is None else self.directory / f"{key}.json"
        if target is not None and target.is_file():
            self.done += 1
            return json.loads(target.read_text(encoding="utf-8"))
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise Unfinished(f"time limit reached after {self.done} finished steps (next: {key})")
        text = json.dumps(compute(), sort_keys=True, separators=(",", ":"))
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".part")
            partial.write_text(text, encoding="utf-8")
            os.replace(partial, target)  # a step is either complete on disk or absent
        self.done += 1
        return json.loads(text)
