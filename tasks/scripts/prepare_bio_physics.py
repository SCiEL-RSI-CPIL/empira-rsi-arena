#!/usr/bin/env python3
"""Fetch the pinned public sources of the biology and physics research tasks and derive their data.

Raw downloads are cached in bio_physics-raw under the task cache and checked against the SHA-256 in
catalog/bio_physics-data-lock.json. Each task module (suite/_lib/autoresearch_suite/bio_*.py,
phys_*.py) turns the raw sources into small gzip-compressed JSON files with its ``derive``
function; those files are written to data/bio_physics/ in the task cache and checked against the
hashes recorded both in the lock and in the module. Evaluators read only the derived files
and never use the network.

    python3 scripts/prepare_bio_physics.py                      # every task
    python3 scripts/prepare_bio_physics.py --only phys_spinglass
    python3 scripts/prepare_bio_physics.py --record             # maintainer: record observed hashes

Licences are recorded per source in the lock; sources whose publisher states none are marked
"not stated" and are never redistributed by this repository (the files stay in the task cache of
whoever runs this script, outside the task library; see ARENA_TASK_CACHE_DIR).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "catalog" / "bio_physics-data-lock.json"
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from arena_task_paths import cache_path  # noqa: E402
from autoresearch_suite import bio_physics_common as common  # noqa: E402

RAW = cache_path("bio_physics-raw")

MODULES = ("phys_spinglass", "phys_nuclear", "bio_align", "bio_rna", "bio_flu")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, attempts=4):
    destination.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(attempts):
        name = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "autoresearch-tasks-bio-physics-data/1"})
            with urllib.request.urlopen(request, timeout=600) as response, \
                    tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                name = stream.name
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    stream.write(block)
            os.replace(name, destination)
            return
        except OSError as error:  # transient network errors and TLS resets: retry with backoff
            last = error
            if name:
                Path(name).unlink(missing_ok=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def fetch(source, record=False):
    """Return the cached raw file for a lock entry, downloading and verifying it when needed."""
    path = RAW / source["name"]
    expected = source.get("sha256")
    live = str(source.get("pinning", "")).startswith("live")
    if not path.is_file() or (expected and sha256(path) != expected and not live):
        download(source["url"], path)
    observed = sha256(path)
    if record:
        source["sha256"], source["size_bytes"] = observed, path.stat().st_size
    elif observed != expected:
        if not live:
            raise RuntimeError(f"{source['name']}: SHA-256 {observed} differs from the lock ({expected})")
        print(f"note: live source {source['name']} changed since it was locked; the derived-file hashes decide",
              file=sys.stderr)
    return path


def prepare(only=None, record=False):
    lock = json.loads(LOCK.read_text())
    sources = {s["name"]: s for s in lock["sources"]}
    derived = {d["path"]: d for d in lock["derived"]}
    report = []
    for module_name in MODULES:
        if only and module_name not in only:
            continue
        module = importlib.import_module("autoresearch_suite." + module_name)
        missing = [name for name in module.RAW_SOURCES if name not in sources]
        if missing:
            raise RuntimeError(f"{module_name}: sources missing from the lock: {missing}")
        paths = {name: fetch(sources[name], record) for name in module.RAW_SOURCES}
        for name, payload in module.derive(paths).items():
            data = common.canonical_bytes(payload)
            digest = hashlib.sha256(data).hexdigest()
            entry = derived.setdefault(name, {"path": name, "module": module_name, "sources": list(module.RAW_SOURCES)})
            if record:
                entry.update(module=module_name, sources=list(module.RAW_SOURCES), sha256=digest, size_bytes=len(data))
                if module.DERIVED.get(name) != digest:
                    print(f"update {module_name}.DERIVED[{name!r}] = {digest!r}", file=sys.stderr)
            else:
                for label, expected in (("lock", entry.get("sha256")), (module_name, module.DERIVED.get(name))):
                    if digest != expected:
                        raise RuntimeError(f"derived {name}: SHA-256 {digest} differs from {label} ({expected})")
            target = common.data_dir() / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            report.append({"path": name, "sha256": digest, "bytes": len(data)})
    if record:
        lock["sources"] = sorted(sources.values(), key=lambda s: s["name"])
        lock["derived"] = sorted(derived.values(), key=lambda d: d["path"])
        LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="task modules to prepare (default: all)")
    parser.add_argument("--record", action="store_true", help="write observed hashes into the lock")
    args = parser.parse_args()
    print(json.dumps(prepare(args.only, args.record), indent=2))


if __name__ == "__main__":
    main()
