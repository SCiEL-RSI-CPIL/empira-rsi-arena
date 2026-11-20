#!/usr/bin/env python3
"""Download the pinned public subjects of the software-engineering tasks and derive their data files.

Every raw source listed in catalog/software-data-lock.json is fetched from its
pinned URL, verified by SHA-256 and cached in software-raw under the task cache. Each task
family module (suite/_lib/autoresearch_suite/software_*.py with ``SOURCES``
and ``derive``) turns the raw bytes into small files under
data/software in the task cache, which are verified against the lock as well. The
repository redistributes none of these files; evaluators read the prepared
files and never use the network.

    python3 scripts/prepare_software.py                # fetch, derive, verify all families
    python3 scripts/prepare_software.py --only cit     # one family
    python3 scripts/prepare_software.py --update-lock  # maintainers: record new hashes
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
from arena_task_paths import cache_path  # noqa: E402
from autoresearch_suite import software_data  # noqa: E402

LOCK = ROOT / "catalog" / "software-data-lock.json"
RAW = cache_path("software-raw")
FAMILIES = ("cit", "quixbugs", "coalesce", "tcp", "pymod", "invariants", "superopt", "sbfl", "logparse", "pbe")


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def fetch(source, update):
    """Verified payload of one raw source (cached by id)."""
    cached = RAW / source["id"]
    payload = cached.read_bytes() if cached.is_file() else None
    if payload is None or (not update and sha256(payload) != source.get("sha256")):
        request = urllib.request.Request(source["url"], headers={"User-Agent": "autoresearch-tasks software data (pinned)"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = response.read()
                break
            except OSError as error:
                if attempt == 3:
                    raise RuntimeError(f"download failed for {source['id']}: {error}") from error
                time.sleep(5 * (attempt + 1))
    if update:
        source.update(sha256=sha256(payload), size=len(payload))
    elif sha256(payload) != source.get("sha256"):
        raise RuntimeError(f"{source['id']}: SHA-256 mismatch ({sha256(payload)}); the pinned file changed")
    RAW.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(payload)
    return payload


def prepare(families, update=False):
    lock = json.loads(LOCK.read_text())
    sources = {s["id"]: s for s in lock["sources"]}
    recorded = {f["path"]: f for f in lock["files"]}
    out = software_data.data_dir()
    report = []
    for family in families:
        module = importlib.import_module("autoresearch_suite.software_" + family)
        raw = {sid: fetch(sources[sid], update) for sid in module.SOURCES}
        for relative, payload in sorted(module.derive(raw).items()):
            entry = recorded.get(relative)
            if update:
                entry = recorded.setdefault(relative, {"path": relative})
                entry.update(family=family, derived_from=list(module.SOURCES), sha256=sha256(payload), size=len(payload))
            elif entry is None or entry["sha256"] != sha256(payload):
                raise RuntimeError(f"{relative}: derived file differs from the lock ({sha256(payload)})")
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            report.append({"path": relative, "size": len(payload)})
    if update:
        lock["files"] = sorted(recorded.values(), key=lambda f: f["path"])
        LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", choices=FAMILIES, help="prepare only this family (repeatable)")
    parser.add_argument("--update-lock", action="store_true", help="record the hashes of the downloaded and derived files")
    args = parser.parse_args()
    report = prepare(args.only or FAMILIES, update=args.update_lock)
    print(json.dumps({"files": len(report), "bytes": sum(r["size"] for r in report), "directory": str(software_data.data_dir())}))


if __name__ == "__main__":
    main()
