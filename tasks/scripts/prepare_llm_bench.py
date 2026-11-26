#!/usr/bin/env python3
"""Download and verify the public data used by the llm_bench evaluators.

    python scripts/prepare_llm_bench.py                 # every prepared file
    python scripts/prepare_llm_bench.py addsub.json     # selected files only
    python scripts/prepare_llm_bench.py --verify-only   # check existing files

Sources are pinned URLs (Git commits, Hugging Face revisions) listed in
``suite/_lib/autoresearch_suite/llm_bench_sources.py``.  Each download must
match the SHA-256 and size recorded in ``catalog/llm_bench-data-lock.json``;
the derived files, written to ``data/llm_bench`` in the task cache
(``ARENA_TASK_CACHE_DIR``), are hash-checked as well.  Raw downloads are kept
under ``llm-bench-sources`` in the same cache; ``--models`` uses the Hugging
Face cache.  ``crave_test.json`` and ``arxivmath.json`` are read from parquet
files and need ``pyarrow``.
``--update-lock`` (maintainers only) records the hashes of the current sources.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from arena_task_paths import cache_path, data_root  # noqa: E402
from autoresearch_suite import llm_bench_sources as sources  # noqa: E402

LOCK = ROOT / "catalog" / "llm_bench-data-lock.json"
DATA = data_root() / "llm_bench"
CACHE = cache_path("llm-bench-sources")


def content_digest(path, name):
    """SHA-256 of a prepared file's content (decompressed for .gz files, whose bytes depend on the zlib build)."""
    if name.endswith(".gz"):
        return hashlib.sha256(gzip.decompress(Path(path).read_bytes())).hexdigest()
    return sha256_file(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    last = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "autoresearch-tasks-llm-bench"})
            with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as handle:
                for block in iter(lambda: response.read(1 << 20), b""):
                    handle.write(block)
            partial.replace(target)
            return
        except OSError as error:
            last = error
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def tree_digest(directory, files):
    """SHA-256 over the sorted (path, file SHA-256) lines of a multi-file source."""
    lines = [f"{rel} {sha256_file(directory / rel)}" for rel in sorted(files)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def fetch_tree(name, spec, lock_row, update):
    directory = CACHE / name
    files = spec["files"]
    if lock_row and all((directory / rel).is_file() for rel in files) and tree_digest(directory, files) == lock_row["sha256"]:
        return directory
    print(f"downloading {name}: {len(files)} files under {spec['base']}", flush=True)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda rel: download(f"{spec['base']}/{rel}", directory / rel), files))
    if not update:
        if lock_row is None:
            raise SystemExit(f"{name} is not in the lock; run with --update-lock")
        if tree_digest(directory, files) != lock_row["sha256"] or len(files) != lock_row["files"]:
            raise SystemExit(f"{name}: downloaded files differ from the lock")
    return directory


def source_lock_entry(name, path):
    info = sources.SOURCES[name]
    if "files" in info:
        return {"url": info["base"], "files": len(info["files"]), "sha256": tree_digest(path, info["files"]),
                "size": sum((path / rel).stat().st_size for rel in info["files"]), "license": info["license"]}
    row = {"url": info["url"], "sha256": sha256_file(path), "size": path.stat().st_size, "license": info["license"]}
    if "ranges" in info:
        row["ranges"] = len(info["ranges"])
    return row


def fetch_ranges(name, spec, lock_row, update):
    """Byte ranges (complete JSONL lines) of a large file, concatenated in offset order."""
    path = CACHE / name
    if lock_row and path.is_file() and sha256_file(path) == lock_row["sha256"]:
        return path
    print(f"downloading {name}: {len(spec['ranges'])} byte ranges of {spec['url']}", flush=True)
    chunks = []
    for offset, length in sorted(spec["ranges"]):
        for attempt in range(4):
            try:
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "autoresearch-tasks-llm-bench", "Range": f"bytes={offset}-{offset + length - 1}"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    chunk = response.read()
                if response.status != 206 or len(chunk) != length:
                    raise OSError(f"range request returned status {response.status} and {len(chunk)} bytes")
                chunks.append(chunk)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(chunks))
    if not update and (lock_row is None or sha256_file(path) != lock_row["sha256"]):
        raise SystemExit(f"{name}: downloaded ranges differ from the lock")
    return path


def fetch(name, spec, lock_row, update):
    if "files" in spec:
        return fetch_tree(name, spec, lock_row, update)
    if "ranges" in spec:
        return fetch_ranges(name, spec, lock_row, update)
    path = CACHE / name
    if lock_row and path.is_file() and sha256_file(path) == lock_row["sha256"]:
        return path
    print(f"downloading {name}: {spec['url']}", flush=True)
    download(spec["url"], path)
    if not update:
        if lock_row is None:
            raise SystemExit(f"{name} is not in the lock; run with --update-lock")
        actual = sha256_file(path)
        if actual != lock_row["sha256"] or path.stat().st_size != lock_row["size"]:
            raise SystemExit(f"{name}: downloaded file differs from the lock ({actual})")
    return path


def fetch_models():
    """Download the pinned local models into the Hugging Face cache and check their weight hashes."""
    from huggingface_hub import hf_hub_download
    for name, spec in sources.MODELS.items():
        paths = {rel: hf_hub_download(spec["repo"], rel, revision=spec["revision"]) for rel in spec["files"]}
        for rel, digest in spec["weights"].items():
            if sha256_file(paths[rel]) != digest:
                raise SystemExit(f"{name}: {rel} differs from the pinned SHA-256")
        print(f"model   {name}: {spec['repo']}@{spec['revision'][:12]} ({len(paths)} files verified)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="derived file names (default: all)")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--update-lock", action="store_true")
    parser.add_argument("--models", action="store_true", help="also fetch the local models of the LMX tasks (about 600 MB)")
    args = parser.parse_args(argv)
    if args.models:
        fetch_models()
    lock = json.loads(LOCK.read_text()) if LOCK.is_file() else {"schema_version": "1.0", "package": "llm_bench", "sources": {}, "derived": {}}
    wanted = args.files or sorted(sources.DERIVED)
    unknown = [name for name in wanted if name not in sources.DERIVED]
    if unknown:
        raise SystemExit("unknown prepared files: " + ", ".join(unknown))
    failures = []
    for name in wanted:
        target = DATA / name
        locked = lock["derived"].get(name)
        if args.verify_only:
            ok = target.is_file() and locked and content_digest(target, name) == locked["sha256"]
            print(("ok      " if ok else "MISSING ") + name)
            if not ok:
                failures.append(name)
            continue
        spec = sources.DERIVED[name]
        paths = {}
        for source in spec["sources"]:
            row = lock["sources"].get(source)
            paths[source] = fetch(source, sources.SOURCES[source], row, args.update_lock)
            if args.update_lock:
                lock["sources"][source] = source_lock_entry(source, paths[source])
        try:
            payload = spec["derive"](paths)
        except ImportError as error:  # a reader that only some files need (pyarrow for the parquet sources, pandas)
            raise SystemExit(f"{name}: its sources are downloaded and verified, but deriving it needs the Python package "
                             f"'{error.name}', which is not installed") from None
        digest = hashlib.sha256(payload).hexdigest()
        if args.update_lock:
            lock["derived"][name] = {"sha256": digest, "size": len(payload), "sources": list(spec["sources"])}
        elif not locked or locked["sha256"] != digest:
            raise SystemExit(f"{name}: derived content differs from the lock ({digest})")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0) if name.endswith(".gz") else payload)
        print(f"wrote   {name} ({len(payload)} bytes)")
    if args.update_lock:
        lock["sources"] = dict(sorted(lock["sources"].items()))
        lock["derived"] = dict(sorted(lock["derived"].items()))
        LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
        print(f"updated {LOCK.relative_to(ROOT)}")
    if failures:
        raise SystemExit("missing or modified prepared files: " + ", ".join(failures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
