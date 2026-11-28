#!/usr/bin/env python3
"""Fetch the pinned public sources of the earth research tasks and derive their data files.

Raw downloads are cached in earth-raw under the task cache and checked against the SHA-256 in
catalog/earth-data-lock.json. Each task module (suite/_lib/autoresearch_suite/earth_*.py)
derives small gzip-compressed JSON files from the raw sources; these are written
to data/earth/ in the task cache and checked against the hashes recorded both in the lock
and in the module. Evaluators read only the derived files and never use the network.

    python3 scripts/prepare_earth.py                 # all tasks
    python3 scripts/prepare_earth.py --only earth_cyclone
    python3 scripts/prepare_earth.py --record        # maintainer: record observed hashes in the lock

    python3 scripts/prepare_earth.py --only earth_icedrift --budget 60   # stop cleanly after about a minute; rerun to continue

Sources marked "pinning": "live" are files that the publisher keeps appending to
(monthly indices); for them the raw hash in the lock documents the retrieved
version and the derived-file hash (covering only the fixed period the task uses)
is the binding check. A source may also name an "archive_url": an immutable copy of
exactly the locked bytes (for example a web-archive capture). It is tried before the
publisher's URL whenever the cached file is not the locked version.

A module whose derivation takes minutes (earth_icedrift: one plane fit per buoy-day)
defines PARTS, derive_part() and assemble(). Its parts are computed one at a time and
kept in earth-work under the task cache, so an interrupted or time-limited run resumes
where it stopped; the assembled file is the same as from an uninterrupted derive().
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "catalog" / "earth-data-lock.json"
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from arena_task_paths import cache_path  # noqa: E402
from autoresearch_suite import earth_common  # noqa: E402

RAW = cache_path("earth-raw")
WORK = cache_path("earth-work")

MODULES = ("earth_cyclone", "earth_enso", "earth_quakes", "earth_sdm", "earth_twostream", "earth_gmst", "earth_aftershocks", "earth_nitrate", "earth_snow", "earth_waves", "earth_radii", "earth_homog", "earth_co2", "earth_icedrift", "earth_solar", "earth_mc")


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
            request = urllib.request.Request(url, headers={"User-Agent": "autoresearch-tasks-earth-data/1"})
            with urllib.request.urlopen(request, timeout=300) as response, \
                    tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                name = stream.name
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    stream.write(block)
            os.replace(name, destination)
            return
        except OSError as error:  # network errors and resets: retry with backoff
            last = error
            if name:
                Path(name).unlink(missing_ok=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def fetch(source, record=False):
    """Return the cached raw file for a lock entry, downloading and verifying it if needed."""
    path = RAW / source["name"]
    expected = source.get("sha256")
    live = source.get("pinning") == "live"
    archive = source.get("archive_url")
    if archive and not record and expected and not (path.is_file() and sha256(path) == expected):
        try:  # the archived copy is the locked version; the live file may have moved on
            download(archive, path)
        except RuntimeError as error:
            print(f"note: {error}; trying the publisher's URL", file=sys.stderr)
    if not path.is_file() or (expected and sha256(path) != expected and not live):
        download(source["url"], path)
    observed = sha256(path)
    if record:
        source["sha256"], source["size_bytes"] = observed, path.stat().st_size
    elif observed != expected:
        if not live:
            raise RuntimeError(f"{source['name']}: SHA-256 {observed} differs from the lock ({expected})")
        print(f"note: live source {source['name']} changed since it was locked; the derived-file hashes decide", file=sys.stderr)
    return path


class Unfinished(RuntimeError):
    """A stepwise derivation used its time budget; the finished parts are kept for the next run."""


def derive(module, module_name, paths, deadline=None):
    """(payloads, work folder or None) of a module; modules with PARTS are derived part by part and can resume."""
    if not hasattr(module, "PARTS"):
        return module.derive(paths), None
    key = hashlib.sha256()  # parts are valid only for these raw bytes and this code
    key.update(Path(module.__file__).read_bytes())
    for name in sorted(paths):
        key.update(f"{name}:{sha256(paths[name])}\n".encode())
    folder = WORK / module_name / key.hexdigest()[:20]
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    for index, part in enumerate(module.PARTS):
        target = folder / f"{part}.json"
        if not target.is_file():
            if deadline is not None and time.monotonic() > deadline:
                raise Unfinished(f"{module_name}: {index} of {len(module.PARTS)} parts are done; run the same command again to continue")
            with tempfile.NamedTemporaryFile("w", dir=folder, delete=False) as stream:
                json.dump(module.derive_part(paths, part), stream, allow_nan=False)
            os.replace(stream.name, target)
        parts.append(json.loads(target.read_text()))  # always the stored form, so a resumed run assembles the same bytes
    return module.assemble(parts), folder


def prepare(only=None, record=False, budget=None, lock_path=None):
    lock_path = Path(lock_path) if lock_path else LOCK
    lock = json.loads(lock_path.read_text())
    deadline = time.monotonic() + budget if budget else None
    sources = {s["name"]: s for s in lock["sources"]}
    derived = {d["path"]: d for d in lock["derived"]}
    report = []
    for module_name in MODULES:
        if only and module_name not in only:
            continue
        module = importlib.import_module("autoresearch_suite." + module_name)
        if record:  # new raw sources are declared next to the code that derives from them
            for spec in getattr(module, "SOURCE_SPECS", ()):
                sources.setdefault(spec["name"], dict(spec))
        missing = [name for name in module.RAW_SOURCES if name not in sources]
        if missing:
            raise RuntimeError(f"{module_name}: raw sources missing from the lock: {missing}")
        paths = {name: fetch(sources[name], record) for name in module.RAW_SOURCES}
        payloads, work = derive(module, module_name, paths, deadline)
        for name, payload in payloads.items():
            data = earth_common.canonical_bytes(payload)
            digest = hashlib.sha256(data).hexdigest()
            entry = derived.setdefault(name, {"path": name, "module": module_name, "sources": list(module.RAW_SOURCES)})
            if record:
                entry.update(sha256=digest, size_bytes=len(data))
                if module.DERIVED.get(name) != digest:
                    print(f"update {module_name}.DERIVED[{name!r}] = {digest!r}", file=sys.stderr)
            else:
                for label, expected in (("lock", entry.get("sha256")), (module_name, module.DERIVED.get(name))):
                    if digest != expected:
                        raise RuntimeError(f"derived {name}: SHA-256 {digest} differs from {label} ({expected})")
            target = earth_common.data_dir() / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            report.append({"path": name, "sha256": digest, "bytes": len(data)})
        if work is not None:  # the derived file is written and verified; no parts of this module are needed any more
            shutil.rmtree(work.parent, ignore_errors=True)
    if record:
        lock["sources"] = list(sources.values())
        lock["derived"] = sorted(derived.values(), key=lambda d: d["path"])
        lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="task modules to prepare (default: all)")
    parser.add_argument("--record", action="store_true", help="write observed hashes into the lock")
    parser.add_argument("--budget", type=float, help="seconds after which no new part of a stepwise derivation is started")
    parser.add_argument("--lock", help="lock file to use instead of catalog/earth-data-lock.json")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.only, args.record, args.budget, args.lock), indent=2))
    except Unfinished as pause:
        print(pause, file=sys.stderr)
        raise SystemExit(75) from None  # EX_TEMPFAIL: not an error, more steps remain


if __name__ == "__main__":
    main()
