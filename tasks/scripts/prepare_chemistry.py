#!/usr/bin/env python3
"""Download the pinned public chemistry and materials data and derive the evaluator files.

Every raw source listed in catalog/chemistry-data-lock.json is fetched from its
pinned URL (GitHub files at fixed commits; Matbench files from the Materials
Project host, whose SHA-256 equals the hash recorded by matminer), verified by
SHA-256 and cached in chemistry-raw under the task cache. The derived files are
written to data/chemistry in the task cache and verified against the lock. The repository does
not redistribute any of these files; evaluators read the prepared files and
never use the network. Standard library only.

    python3 scripts/prepare_chemistry.py                  # fetch, derive, verify
    python3 scripts/prepare_chemistry.py --only matbench  # derived files whose path contains "matbench"
    python3 scripts/prepare_chemistry.py --update-lock    # maintainers: record new hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
from arena_task_paths import cache_path, data_root  # noqa: E402
from autoresearch_suite import chem_sources  # noqa: E402

LOCK = ROOT / "catalog" / "chemistry-data-lock.json"
RAW = cache_path("chemistry-raw")
OUT = data_root() / "chemistry"


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def fetch(source, update):
    """Return the verified payload of one raw source (cached in chemistry-raw under the task cache)."""
    cached = RAW / source["id"]
    payload = cached.read_bytes() if cached.is_file() else None
    if payload is None or (not update and sha256(payload) != source.get("sha256")):
        request = urllib.request.Request(source["url"], headers={"User-Agent": "autoresearch-tasks chemistry data (pinned)"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="substring of derived file paths to prepare")
    parser.add_argument("--update-lock", action="store_true", help="record hashes of the current downloads")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    sources = {s["id"]: s for s in lock["sources"]}
    derived = {d["path"]: d for d in lock["derived"]}
    mismatch = set(derived) ^ set(chem_sources.DERIVERS)
    if mismatch:
        raise SystemExit(f"lock and derivations disagree: {sorted(mismatch)}")
    report = []
    for path, entry in derived.items():
        if args.only not in path:
            continue
        needed, derive = chem_sources.DERIVERS[path]
        if sorted(needed) != sorted(entry["sources"]):
            raise SystemExit(f"{path}: lock lists sources {entry['sources']}, deriver needs {list(needed)}")
        payloads = {sid: fetch(sources[sid], args.update_lock) for sid in needed}
        data = derive(payloads)
        if args.update_lock:
            entry.update(sha256=sha256(data), size=len(data))
        elif sha256(data) != entry.get("sha256"):
            raise SystemExit(f"{path}: derived SHA-256 {sha256(data)} does not match the lock")
        target = OUT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        report.append({"file": path, "sha256": sha256(data), "bytes": len(data), "sources": list(needed)})
    if args.update_lock:
        LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "receipt.json").write_text(json.dumps({"lock": str(LOCK.relative_to(ROOT)), "files": report}, indent=2) + "\n")
    print(json.dumps({"prepared": len(report), "directory": str(OUT)}))


if __name__ == "__main__":
    main()
