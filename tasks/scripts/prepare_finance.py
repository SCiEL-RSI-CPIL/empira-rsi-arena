#!/usr/bin/env python3
"""Download the pinned public finance data and derive the small evaluator tables.

Every raw source listed in catalog/finance-data-lock.json is fetched from its
pinned URL (Internet Archive snapshots for files that their publishers
overwrite), verified by SHA-256 and cached in finance-raw under the task cache.
The derived CSV tables are written to data/finance in the task cache and verified against the
lock. The repository does not redistribute any of these files; evaluators read
the prepared tables and never use the network.

    python3 scripts/prepare_finance.py                 # fetch, derive, verify
    python3 scripts/prepare_finance.py --only ind10    # tables whose name contains "ind10"
    python3 scripts/prepare_finance.py --update-lock   # maintainers: record new hashes
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
from arena_task_paths import cache_path, data_root  # noqa: E402
from autoresearch_suite import finance_sources  # noqa: E402

LOCK = ROOT / "catalog" / "finance-data-lock.json"
RAW = cache_path("finance-raw")
OUT = data_root() / "finance"


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def fetch(source, update):
    """Return the verified payload of one raw source (cached)."""
    cached = RAW / source["id"]
    payload = cached.read_bytes() if cached.is_file() else None
    if payload is None or (not update and sha256(payload) != source.get("sha256")):
        request = urllib.request.Request(source["url"], headers={"User-Agent": "autoresearch-tasks finance data (pinned)"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    payload = response.read()
                break
            except OSError as error:  # transient archive errors
                if attempt == 3:
                    raise RuntimeError(f"download failed for {source['id']}: {error}") from error
                time.sleep(5 * (attempt + 1))
        if payload[:2] == b"\x1f\x8b" and source.get("content_encoding") == "gzip":
            payload = gzip.decompress(payload)  # archived HTTP transfer encoding
    if update:
        source.update(sha256=sha256(payload), size=len(payload))
    elif sha256(payload) != source.get("sha256"):
        raise RuntimeError(f"{source['id']}: SHA-256 mismatch ({sha256(payload)}); the pinned file changed")
    RAW.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="substring of derived table names to prepare")
    parser.add_argument("--update-lock", action="store_true", help="record hashes of the current downloads")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text())
    sources = {s["id"]: s for s in lock["sources"]}
    derived = {d["path"]: d for d in lock["derived"]}
    unknown = set(derived) ^ set(finance_sources.DERIVERS)
    if unknown:
        raise SystemExit(f"lock and derivations disagree: {sorted(unknown)}")
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for name, entry in derived.items():
        if args.only not in name:
            continue
        payloads = {sid: fetch(sources[sid], args.update_lock) for sid in entry["sources"]}
        text = finance_sources.DERIVERS[name](payloads)
        data = text.encode("utf-8")
        if args.update_lock:
            entry.update(sha256=sha256(data), size=len(data))
        elif sha256(data) != entry.get("sha256"):
            raise SystemExit(f"{name}: derived SHA-256 {sha256(data)} does not match the lock")
        (OUT / name).write_bytes(data)
        report.append({"table": name, "sha256": sha256(data), "bytes": len(data), "sources": entry["sources"]})
    if args.update_lock:
        LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    (OUT / "receipt.json").write_text(json.dumps({"lock": str(LOCK.relative_to(ROOT)), "tables": report}, indent=2) + "\n")
    print(json.dumps({"prepared": len(report), "directory": str(OUT)}))


if __name__ == "__main__":
    main()
