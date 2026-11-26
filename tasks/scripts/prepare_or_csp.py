#!/usr/bin/env python3
"""Fetch the public data of the or_csp tasks into data/or_csp/ of the task cache and verify the lock.

Every file comes from a pinned public revision and is checked against the SHA-256
and size recorded in catalog/or_csp-data-lock.json. Entries with
"stored_as": "gzip" are kept gzip-compressed; their SHA-256 and size refer to the
uncompressed source bytes, which are re-hashed after decompression. Never run
during scoring.

    python3 scripts/prepare_or_csp.py            # download what is missing, verify everything
    python3 scripts/prepare_or_csp.py --check    # verify only (no network)
    python3 scripts/prepare_or_csp.py --only iopddl/   # one group of lock paths

The task cache is ARENA_TASK_CACHE_DIR (default ~/.cache/empira/tasks).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from arena_task_paths import data_root  # noqa: E402

LOCK = ROOT / "catalog" / "or_csp-data-lock.json"
DATA = data_root() / "or_csp"


def source_bytes(path, entry):
    """The bytes the lock describes: the file itself, or its decompressed content."""
    if entry.get("stored_as") == "gzip":
        with gzip.open(path, "rb") as stream:
            return stream.read()
    return Path(path).read_bytes()


def verified(path, entry):
    if not path.is_file():
        return False
    try:
        data = source_bytes(path, entry)
    except OSError:
        return False
    return len(data) == entry["size_bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]


def fetch(url, destination, expected_sha, expected_size, stored_as=None):
    request = urllib.request.Request(url, headers={"User-Agent": "autoresearch-tasks-pinned-data"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read(expected_size + 1)
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError(f"{url}: download does not match the locked size/SHA-256")
    if stored_as == "gzip":
        payload = gzip.compress(payload, compresslevel=9, mtime=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            name = stream.name
            stream.write(payload)
        os.chmod(name, 0o644)
        os.replace(name, destination)
        name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def prepare(check_only=False, only=None):
    lock = json.loads(LOCK.read_text())
    report = []
    for entry in lock["files"]:
        if only and not entry["path"].startswith(only):
            continue
        path = DATA / entry["path"]
        ok = verified(path, entry)
        if not ok and not check_only:
            fetch(entry["url"], path, entry["sha256"], entry["size_bytes"], entry.get("stored_as"))
            ok, state = True, "downloaded_verified"
        else:
            state = "verified" if ok else "missing_or_mismatch"
        report.append({"path": entry["path"], "status": state})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only, do not download")
    parser.add_argument("--only", help="limit to lock paths with this prefix (e.g. iopddl/)")
    args = parser.parse_args()
    report = prepare(args.check, args.only)
    bad = [r for r in report if r["status"] == "missing_or_mismatch"]
    print(json.dumps({"files": len(report), "problems": bad}, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
