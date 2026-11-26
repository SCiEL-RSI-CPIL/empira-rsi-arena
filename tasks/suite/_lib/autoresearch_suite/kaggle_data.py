"""Paths and file layout of the prepared data of the kaggle package (MLE-bench recreations).

``scripts/prepare_kaggle.py`` downloads the pinned public sources (catalog/kaggle-data-lock.json) and
writes, per task, ``<data_dir>/<task>/public/*`` (what the candidate may read: training data with labels
and unlabelled test inputs) and ``<data_dir>/<task>/private/*`` (held-out labels, read only by the
evaluator). ``data_dir()`` is ``data/kaggle`` and the raw downloads are ``kaggle-raw`` of the task cache
(arena_task_paths: ARENA_TASK_CACHE_DIR, default ~/.cache/empira/tasks), outside the task library.
Evaluators never use the network.
"""
from __future__ import annotations

from arena_task_paths import cache_path, data_root

TASKS = ("jigsaw", "noisyoffice", "pcam", "nomad", "openvaccine", "patent")


def data_dir():
    return data_root() / "kaggle"


def raw_cache_dir():
    """Raw downloads of scripts/prepare_kaggle.py (contain labels; the sandbox denies reading them)."""
    return cache_path("kaggle-raw")


def public_dir(task):
    return data_dir() / task / "public"


def private_dir(task):
    return data_dir() / task / "private"


def require_files(task, public=(), private=()):
    """Paths of prepared files; a missing file is an installation problem (run the prepare script)."""
    paths = [public_dir(task) / name for name in public] + [private_dir(task) / name for name in private]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError(f"prepared data missing ({', '.join(missing)}); run python3 scripts/prepare_kaggle.py --only {task}")
    return paths


def isolation(task):
    """Read permissions of the sandboxed child: everything prepared or raw is denied except the task's public files."""
    return {"deny_read": [str(data_dir()), str(raw_cache_dir())], "allow_read": [str(public_dir(task))]}
