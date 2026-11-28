#!/usr/bin/env python3
"""Prepare the data of the metaopt evaluators (BIG-bench scaling-law task).

    python scripts/prepare_metaopt.py                # fetch (if needed), verify, derive
    python scripts/prepare_metaopt.py --verify-only  # check the prepared file against the lock
    python scripts/prepare_metaopt.py --update-lock  # maintainers: record the current hashes

Source: the BIG-bench evaluation records ``bigbench/benchmark_tasks/*/results/scores_*.json``
of google/BIG-bench at commit 092b196c1f8f14a54bbc62f24759d43bde46dd3b (Apache-2.0),
fetched with a sparse, blob-filtered ``git`` checkout into ``metaopt-raw/BIG-bench`` under the task
cache (about 130 MB). Git checks every blob against its object id; the lock additionally records
a SHA-256 over the sorted ``path blob-id`` lines of the fetched files.

Derived file ``data/metaopt/bigbench_mcg.csv.gz`` in the task cache: one row per score entry whose
``preferred_score`` is ``multiple_choice_grade``, with the model's training steps, training
batch size, total, non-embedding and FLOP-matched non-embedding parameters, number of shots
and decoding temperature; entries of models that state no decoding temperature are dropped.
The lock records the SHA-256 of the decompressed CSV.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from arena_task_paths import cache_path  # noqa: E402
from autoresearch_suite import metaopt_lasr  # noqa: E402

REPO = "https://github.com/google/BIG-bench.git"
COMMIT = "092b196c1f8f14a54bbc62f24759d43bde46dd3b"
PATTERN = "bigbench/benchmark_tasks/*/results/scores_*.json"
CACHE = cache_path("metaopt-raw") / "BIG-bench"
LOCK = ROOT / "catalog" / "metaopt-data-lock.json"
OUT = metaopt_lasr.data_dir() / metaopt_lasr.DATA_FILE


def git(*args, cwd=CACHE):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def fetch():
    if (CACHE / ".git").is_dir() and git("rev-parse", "HEAD").strip() == COMMIT:
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    if not (CACHE / ".git").is_dir():
        git("init", "-q")
        git("remote", "add", "origin", REPO)
    git("config", "core.sparseCheckout", "true")
    git("sparse-checkout", "set", "--no-cone", PATTERN)
    print(f"fetching {REPO} at {COMMIT} (blob filter, sparse checkout of {PATTERN})", flush=True)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    git("checkout", "-q", "--detach", COMMIT)


def source_files():
    """Sorted (path, blob id, size) of the score files at the pinned commit.

    ``ls-tree`` without ``-l`` reads tree objects only; asking it for sizes would make the
    partial clone download every blob of the task directories.
    """
    rows = []
    for line in git("ls-tree", "-r", COMMIT, "--", "bigbench/benchmark_tasks").splitlines():
        meta, path = line.split("\t", 1)
        _, kind, blob = meta.split()
        if kind == "blob" and "/results/scores_" in path and path.endswith(".json"):
            local = CACHE / path
            if not local.is_file():
                raise SystemExit(f"missing checked-out file {path}; delete {CACHE} and run again")
            rows.append((path, blob, local.stat().st_size))
    return sorted(rows)


def tree_digest(files):
    return hashlib.sha256("\n".join(f"{p} {b}" for p, b, _ in files).encode()).hexdigest()


def derive(files):
    """Return (CSV text, statistics) from the checked-out score files."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(metaopt_lasr.COLUMNS)
    stats = {"files": len(files), "mcg_entries": 0, "rows": 0, "dropped_no_temperature": 0, "dropped_missing_value": 0}
    for path, _, _ in files:
        record = json.loads((CACHE / path).read_text(encoding="utf-8"))
        model, task = record.get("model", {}), record.get("task", {}).get("task_name", path.split("/")[2])
        temperature = (model.get("decoding_params") or {}).get("temperature")
        for entry in record.get("scores", []):
            if entry.get("preferred_score") != "multiple_choice_grade":
                continue
            stats["mcg_entries"] += 1
            if temperature is None:
                stats["dropped_no_temperature"] += 1
                continue
            values = [model.get("training_steps"), model.get("training_batch_size"), model.get("total_params"),
                      model.get("non_embedding_params"), model.get("flop_matched_non_embedding_params"),
                      entry.get("number_of_shots"), temperature, (entry.get("score_dict") or {}).get("multiple_choice_grade")]
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in values):
                stats["dropped_missing_value"] += 1
                continue
            writer.writerow([task, entry.get("subtask_description", ""), model.get("model_family", ""), model.get("model_name", ""),
                             *(repr(float(v)) if isinstance(v, float) else str(v) for v in values)])
            stats["rows"] += 1
    return out.getvalue(), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--update-lock", action="store_true")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text()) if LOCK.is_file() else {}
    prepared = lock.get("prepared", {}).get(metaopt_lasr.DATA_FILE, {})
    if args.verify_only:
        digest = hashlib.sha256(gzip.decompress(OUT.read_bytes())).hexdigest() if OUT.is_file() else None
        ok = digest is not None and digest == prepared.get("sha256")
        print(f"{OUT.name}: {'ok' if ok else 'MISSING OR MISMATCH'}")
        return 0 if ok else 1
    fetch()
    files = source_files()
    digest = tree_digest(files)
    source = lock.get("sources", {}).get("bigbench-results", {})
    if not args.update_lock and source and source.get("sha256") != digest:
        raise SystemExit("BIG-bench score files do not match the locked tree digest")
    text, stats = derive(files)
    content = hashlib.sha256(text.encode()).hexdigest()
    if not args.update_lock and prepared and prepared.get("sha256") != content:
        raise SystemExit("derived CSV does not match the locked SHA-256")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(gzip.compress(text.encode(), mtime=0))
    print(json.dumps(stats))
    if args.update_lock:
        lock = {"schema_version": "1.0", "package": "metaopt", "sources": {"bigbench-results": {
            "url": f"https://github.com/google/BIG-bench/tree/{COMMIT}/bigbench/benchmark_tasks",
            "commit": COMMIT, "pattern": PATTERN, "files": len(files), "size": sum(s for _, _, s in files), "sha256": digest,
            "digest": "sha256 over sorted 'path git-blob-id' lines",
            "license": "Apache-2.0 (google/BIG-bench repository licence). Not redistributed by this repository."}},
            "prepared": {metaopt_lasr.DATA_FILE: {"sha256": content, "digest": "sha256 of the decompressed CSV",
                                                   "rows": stats["rows"], "statistics": stats, "derived_from": "bigbench-results"}}}
        LOCK.write_text(json.dumps(lock, indent=2) + "\n")
        print(f"wrote {LOCK.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
