#!/usr/bin/env python3
"""Fetch the QAPLIB instances of the frontier_combinatorics tasks into data/frontier_combinatorics/ of the task cache.

The instance files come from the QAPLIB deposit on Edinburgh DataShare
(https://doi.org/10.7488/ds/3428, CC BY 4.0). The archive qapdata.tar.gz is
downloaded once into frontier_combinatorics-raw/ of the task cache and checked against the
SHA-256 and size in catalog/frontier_combinatorics-data-lock.json; the three
instance files are read from it member by member (nothing else is extracted)
and checked against their own pinned digests. Never run during scoring.

    python3 scripts/prepare_frontier_combinatorics.py            # download what is missing, verify everything
    python3 scripts/prepare_frontier_combinatorics.py --check    # verify only (no network)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "catalog" / "frontier_combinatorics-data-lock.json"
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from arena_task_paths import cache_path, data_root  # noqa: E402

DATA = data_root() / "frontier_combinatorics"
RAW = cache_path("frontier_combinatorics-raw")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(payload)
        name = stream.name
    os.chmod(name, 0o644)
    os.replace(name, path)


def archive(source, check_only):
    """Bytes of a locked source archive, downloading it into the raw cache when allowed."""
    path = RAW / source["cache_name"]
    if path.is_file():
        data = path.read_bytes()
        if len(data) == source["size_bytes"] and digest(data) == source["sha256"]:
            return data
    if check_only:
        return None
    request = urllib.request.Request(source["url"], headers={"User-Agent": "autoresearch-tasks-pinned-data"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(source["size_bytes"] + 1)
    if len(data) != source["size_bytes"] or digest(data) != source["sha256"]:
        raise ValueError(f"{source['url']}: download does not match the locked size/SHA-256")
    write_atomic(path, data)
    return data


def member(data, name):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        info = tar.getmember(name)
        if not info.isfile():
            raise ValueError(f"{name} is not a regular file in the archive")
        return tar.extractfile(info).read()


def prepare(check_only=False):
    lock = json.loads(LOCK.read_text())
    sources = {s["name"]: s for s in lock["sources"]}
    archives, report = {}, []
    for entry in lock["files"]:
        path = DATA / entry["path"]
        if path.is_file():
            data = path.read_bytes()
            if len(data) == entry["size_bytes"] and digest(data) == entry["sha256"]:
                report.append({"path": entry["path"], "state": "verified"})
                continue
        name = entry["source"]
        if name not in archives:
            archives[name] = archive(sources[name], check_only)
        if archives[name] is None:
            report.append({"path": entry["path"], "state": "missing"})
            continue
        payload = member(archives[name], entry["member"])
        if len(payload) != entry["size_bytes"] or digest(payload) != entry["sha256"]:
            raise ValueError(f"{entry['member']} in {name} does not match the locked size/SHA-256")
        write_atomic(path, payload)
        report.append({"path": entry["path"], "state": "extracted_verified"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify the prepared files without downloading")
    args = parser.parse_args()
    report = prepare(check_only=args.check)
    print(json.dumps(report, indent=2))
    return 1 if any(row["state"] == "missing" for row in report) else 0


if __name__ == "__main__":
    sys.exit(main())
