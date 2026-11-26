#!/usr/bin/env python3
"""Download the pinned public data of the kaggle package (MLE-bench recreations) and derive the task files.

Every raw source in catalog/kaggle-data-lock.json is streamed from its pinned URL into kaggle-raw of the
task cache (ARENA_TASK_CACHE_DIR, default ~/.cache/empira/tasks), verified by size and SHA-256, and turned
into the per-task files under data/kaggle of the task cache by suite/_lib/autoresearch_suite/kaggle_derive*.py;
derived files are verified against the lock as well. A source already placed at
kaggle-raw/<source id>/<filename> is verified and used. The repository redistributes none of these
files; evaluators read the prepared files and never use the network. Raw files contain held-out labels:
the candidate sandbox denies reading kaggle-raw.

Preparation needs packages that evaluation does not: pyarrow (nomad, patent), h5py (pcam) and Pillow
(noisyoffice).

    python3 scripts/prepare_kaggle.py                    # all tasks (2.2 GB of downloads, 1.6 GB of them PCam)
    python3 scripts/prepare_kaggle.py --only nomad patent
    python3 scripts/prepare_kaggle.py --update-lock      # maintainers: record new hashes
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
from autoresearch_suite import kaggle_data, kaggle_derive, kaggle_derive_images  # noqa: E402

LOCK = ROOT / "catalog" / "kaggle-data-lock.json"
DERIVE = {"jigsaw": kaggle_derive.jigsaw, "noisyoffice": kaggle_derive_images.noisyoffice, "pcam": kaggle_derive_images.pcam,
          "nomad": kaggle_derive.nomad, "openvaccine": kaggle_derive.openvaccine, "patent": kaggle_derive.patent}
# what a derivation imports beyond numpy and pandas (module, distribution)
NEEDS = {"noisyoffice": [("PIL", "Pillow")], "pcam": [("h5py", "h5py")], "nomad": [("pyarrow", "pyarrow")], "patent": [("pyarrow", "pyarrow")]}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


USER_AGENT = {"User-Agent": "pinned-data-fetch/1"}  # generic: names no project or person
RANGED_MIN_BYTES = 100 << 20  # Zenodo serves about 0.2 MB/s per connection; large files use parallel ranges


def stream(url, partial):
    request = urllib.request.Request(url, headers=USER_AGENT)
    with urllib.request.urlopen(request, timeout=600) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 22)


def ranged(url, size, partial, connections):
    """Parallel HTTP range requests into a preallocated file; raises if the server ignores ranges."""
    chunk = -(-size // connections)
    with open(partial, "wb") as out:
        out.truncate(size)

    def part(i):
        start, end = i * chunk, min(size, (i + 1) * chunk) - 1
        request = urllib.request.Request(url, headers={**USER_AGENT, "Range": f"bytes={start}-{end}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    if response.status != 206:
                        raise RuntimeError("server ignored the range request")
                    payload = response.read()
                if len(payload) != end - start + 1:
                    raise OSError(f"short range {len(payload)} of {end - start + 1} bytes")
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        with open(partial, "r+b") as out:
            out.seek(start)
            out.write(payload)
    with ThreadPoolExecutor(connections) as pool:
        list(pool.map(part, range(connections)))


def fetch(source, update, connections=6):
    """Local path of one verified raw download (cached under kaggle-raw/<id>/ of the task cache)."""
    target = kaggle_data.raw_cache_dir() / source["id"] / source["filename"]
    if target.is_file() and (update or (target.stat().st_size == source.get("size") and file_sha256(target) == source.get("sha256"))):
        if update:
            source.update(sha256=file_sha256(target), size=target.stat().st_size)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    size = source.get("size")
    if isinstance(size, int) and size >= RANGED_MIN_BYTES and connections > 1:
        try:
            ranged(source["url"], size, partial, connections)
        except (OSError, RuntimeError) as error:
            print(f"{source['id']}: parallel ranges failed ({error}); streaming instead", flush=True)
            size = None
    if not (isinstance(size, int) and size >= RANGED_MIN_BYTES and connections > 1):
        for attempt in range(4):
            try:
                stream(source["url"], partial)
                break
            except OSError as error:
                if attempt == 3:
                    raise RuntimeError(f"download failed for {source['id']}: {error}") from error
                time.sleep(5 * (attempt + 1))
    digest, size = file_sha256(partial), partial.stat().st_size
    if update:
        source.update(sha256=digest, size=size)
    elif digest != source.get("sha256") or size != source.get("size"):
        raise RuntimeError(f"{source['id']}: SHA-256/size mismatch ({digest}, {size} bytes); the pinned file changed")
    partial.replace(target)
    return target


def prepare(tasks, update=False, connections=6):
    lock = json.loads(LOCK.read_text())
    sources = {s["id"]: s for s in lock["sources"]}
    recorded = {f["path"]: f for f in lock["files"]}
    out = kaggle_data.data_dir()
    report = {}
    for task in tasks:
        missing = [name for module, name in NEEDS.get(task, []) if importlib.util.find_spec(module) is None]
        if missing:
            raise RuntimeError(f"{task}: preparing this task needs the Python package(s) {', '.join(missing)} "
                               "(used only by the preparation, not by the evaluator)")
        task_sources = [s for s in lock["sources"] if s["task"] == task and s.get("role", "data") == "data"]
        raw = {s["id"]: fetch(sources[s["id"]], update, connections) for s in task_sources}
        files, summary = DERIVE[task](raw)
        for relative, payload in sorted(files.items()):
            digest = hashlib.sha256(payload).hexdigest()
            entry = recorded.get(relative)
            if update:
                entry = recorded.setdefault(relative, {"path": relative})
                entry.update(task=task, derived_from=[s["id"] for s in task_sources], sha256=digest, size=len(payload))
            elif entry is None or entry["sha256"] != digest:
                raise RuntimeError(f"{relative}: derived file differs from the lock ({digest})")
        for relative, payload in sorted(files.items()):  # written only when every file of the task is verified
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        report[task] = {"files": len(files), **summary}
        print(f"{task}: {json.dumps(report[task])[:400]}", flush=True)
    if update:
        lock["files"] = sorted(recorded.values(), key=lambda f: f["path"])
        LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=kaggle_data.TASKS, help="prepare only these tasks")
    parser.add_argument("--update-lock", action="store_true", help="record the hashes of downloads and derived files")
    parser.add_argument("--connections", type=int, default=6, help="parallel range requests for files over 100 MB (1 = one stream)")
    args = parser.parse_args(argv)
    prepare(args.only or list(kaggle_data.TASKS), update=args.update_lock, connections=args.connections)
    return 0


if __name__ == "__main__":
    sys.exit(main())
