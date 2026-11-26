#!/usr/bin/env python3
"""Fetch and verify the inputs of the jax_station evaluators (LIMEN on XLand-MiniGrid, ZAPBench).

    python scripts/prepare_jax_station.py --limen      # LIMEN files at the pinned commit (MIT), ~0.2 MB
    python scripts/prepare_jax_station.py --zapbench   # ZAPBench traces (CC-BY-4.0), 2.36 GB download
    python scripts/prepare_jax_station.py --reference  # reproduce ZAPBench's published mean baseline (check)

Paths: <cache> is the task cache (ARENA_TASK_CACHE_DIR, see suite/_lib/arena_task_paths.py);
run the script from the task library root (the directory that holds scripts/ and suite/).

LIMEN: the GitHub tarball of Lossfunk/LIMEN@cf0b44da is cached in
<cache>/jax_station-raw/ and the files the evaluator runs (the limen package,
configs, rulesets, the XLand context and the published evolved interfaces) are
extracted into <cache>/data/jax_station/limen-cf0b44da/; each file is checked
against catalog/jax_station-data-lock.json.

ZAPBench: the zarr v3 array gs://zapbench-release/volumes/20240930/traces
(float32 [7879, 71721], 16 x 141 uncompressed 512 x 512 chunks) is downloaded
over plain HTTPS from storage.googleapis.com into <cache>/jax_station-raw/zapbench-traces
(each chunk is checked against the MD5 that Google Cloud Storage reports), and
the condition-aware splits of the Station's ZAPBench task (4-step context, see
autoresearch_suite/jax_station_zapdata.py) are written to
<cache>/data/jax_station/zapbench/: training and validation arrays as
float16 (trainval/), the test array as float32 (test/), plus per-timestep
condition labels and manifest.json. The split rules are compared with the
pinned zapbench source (constants.py, data_utils.py at 7347b3b2).

--check only hashes the prepared files against the lock (no network);
--write-lock records hashes instead of checking them (maintainers only).
"""
from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import hashlib
import json
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from arena_task_paths import cache_path  # noqa: E402

from autoresearch_suite import jax_station_zapdata as zd  # noqa: E402
from autoresearch_suite.jax_station_common import data_dir  # noqa: E402
from autoresearch_suite.jax_station_limen import LIMEN_COMMIT, limen_dir  # noqa: E402

LOCK = ROOT / "catalog" / "jax_station-data-lock.json"
RAW = cache_path("jax_station-raw")
LIMEN_URL = f"https://codeload.github.com/Lossfunk/LIMEN/tar.gz/{LIMEN_COMMIT}"
LIMEN_KEEP = ("limen/", "configs/", "rulesets/", "contexts/xminigrid.md", "examples/evolved_interfaces/", "LICENSE",
              "README.md", "pyproject.toml", "requirements.txt")
ZAPBENCH_COMMIT = "7347b3b21a3f332e281b774c945f1505dd966abf"
ZAPBENCH_SOURCES = ("zapbench/constants.py", "zapbench/data_utils.py")
CHUNK_BYTES = zd.CHUNK * zd.CHUNK * 4
EXPECTED_ZARR = {"shape": [zd.NUM_TIMESTEPS, zd.NUM_NEURONS], "data_type": "float32", "zarr_format": 3,
                 "chunk_grid": {"configuration": {"chunk_shape": [zd.CHUNK, zd.CHUNK]}, "name": "regular"},
                 "codecs": [{"configuration": {"endian": "little"}, "name": "bytes"}]}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, attempts=4, timeout=120):
    """GET url; returns (bytes, headers). Retries transient errors."""
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "autoresearch-tasks-prepare/1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), response.headers
        except OSError as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"download failed: {url}: {error}") from None
            time.sleep(2 ** attempt)


def load_lock():
    return json.loads(LOCK.read_text()) if LOCK.is_file() else {"schema": "artask.data-lock/1", "package": "jax_station"}


def check(expected, actual, what, write):
    if write:
        return actual
    if expected is None:
        raise SystemExit(f"{what}: no entry in {LOCK.name}; rerun with --write-lock")
    if expected != actual:
        raise SystemExit(f"{what}: mismatch (expected {expected}, got {actual})")
    return actual


# --- LIMEN -------------------------------------------------------------------------------------------------

def prepare_limen(lock, write):
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / f"limen-{LIMEN_COMMIT[:8]}.tar.gz"
    if not archive.is_file():
        data, _ = fetch(LIMEN_URL)
        archive.write_bytes(data)
    entry = lock.setdefault("limen", {})
    entry.update({"url": LIMEN_URL, "repository": "https://github.com/Lossfunk/LIMEN", "commit": LIMEN_COMMIT,
                  "license": "MIT", "archive_note": "files are checked one by one; the GitHub tarball itself may be regenerated"})
    expected = entry.get("files", {}) if not write else {}
    target = limen_dir()
    files = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            relative = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if not relative.startswith(LIMEN_KEEP) or ".." in Path(relative).parts:
                continue
            data = tar.extractfile(member).read()
            files[relative] = {"sha256": sha256_bytes(data), "bytes": len(data)}
            if not write and expected.get(relative) != files[relative]:
                raise SystemExit(f"LIMEN file {relative}: hash mismatch or not in the lock")
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    missing = sorted(set(expected) - set(files))
    if missing:
        raise SystemExit(f"LIMEN files missing from the archive: {missing}")
    if write:
        entry["files"] = dict(sorted(files.items()))
    print(f"LIMEN {LIMEN_COMMIT[:8]}: {len(files)} files verified in {target}")


# --- ZAPBench --------------------------------------------------------------------------------------------

def check_split_rules(lock, write):
    """Compare jax_station_zapdata's split bounds with the pinned zapbench functions (exec'd from source)."""
    entry = lock.setdefault("zapbench_code", {"repository": "https://github.com/google-research/zapbench",
                                              "commit": ZAPBENCH_COMMIT, "license": "Apache-2.0"})
    namespace, hashes = {}, {}
    for name in ZAPBENCH_SOURCES:
        url = f"https://raw.githubusercontent.com/google-research/zapbench/{ZAPBENCH_COMMIT}/{name}"
        cached = RAW / "zapbench-src" / name
        if not cached.is_file():
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(fetch(url)[0])
        source = cached.read_bytes()
        hashes[name] = check(entry.get("files", {}).get(name), {"sha256": sha256_bytes(source), "bytes": len(source)},
                             f"zapbench {name}", write)
        namespace[name] = source.decode("utf-8")
    if write:
        entry["files"] = hashes
    constants = {}
    exec(compile(namespace["zapbench/constants.py"], "constants.py", "exec"), constants)  # noqa: S102 - pinned, hashed source
    tree = ast.parse(namespace["zapbench/data_utils.py"])
    wanted = {"get_condition_bounds", "adjust_condition_bounds_for_split"}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[])
    scope = {"constants": type("constants", (), constants), "Optional": object}
    exec(compile(module, "data_utils.py", "exec"), scope)  # noqa: S102 - pinned, hashed source
    for key in ("CONDITION_OFFSETS", "CONDITION_PADDING", "CONDITIONS_TRAIN", "CONDITIONS_HOLDOUT", "VAL_FRACTION",
                "TEST_FRACTION", "MAX_CONTEXT_LENGTH", "PREDICTION_WINDOW_LENGTH"):
        local = {"PREDICTION_WINDOW_LENGTH": zd.HORIZON}.get(key, getattr(zd, key, None))
        if constants[key] != local:
            raise SystemExit(f"zapbench constant {key} differs: {constants[key]} vs {local}")
    for split in ("train", "val", "test"):
        for condition, start, stop in zd.segments(split):
            name = "test_holdout" if condition in zd.CONDITIONS_HOLDOUT else split
            official = scope["adjust_condition_bounds_for_split"](name, *scope["get_condition_bounds"](condition), zd.CONTEXT)
            if tuple(official) != (start, stop):
                raise SystemExit(f"split {split} condition {condition}: {official} != {(start, stop)}")
    print("ZAPBench split rules agree with zapbench@" + ZAPBENCH_COMMIT[:8])
    return True


def chunk_path(i, j):
    return RAW / "zapbench-traces" / "c" / str(i) / str(j)


def download_chunk(key):
    i, j = key
    path = chunk_path(i, j)
    if path.is_file() and path.stat().st_size == CHUNK_BYTES:
        return key, "cached"
    data, headers = fetch(f"{zd.ZARR_URL}c/{i}/{j}")
    md5 = [h.split("=", 1)[1] for h in headers.get_all("x-goog-hash", []) if h.startswith("md5=")]
    if len(data) != CHUNK_BYTES or not md5 or base64.b64encode(hashlib.md5(data).digest()).decode() != md5[0]:  # noqa: S324
        raise RuntimeError(f"chunk c/{i}/{j}: size or MD5 mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return key, "downloaded"


def prepare_zapbench(lock, write, workers):
    check_split_rules(lock, write)
    entry = lock.setdefault("zapbench", {})
    entry.update({"url": zd.ZARR_URL, "license": "CC-BY-4.0",
                  "license_source": "https://storage.googleapis.com/zapbench-release/volumes/README.html",
                  "attribution": "ZAPBench (Lueckmann et al., 2025), Google Research and HHMI Janelia"})
    meta, _ = fetch(zd.ZARR_URL + "zarr.json")
    zarr = json.loads(meta)
    for key, value in EXPECTED_ZARR.items():
        if zarr.get(key) != value:
            raise SystemExit(f"zarr.json field {key} is {zarr.get(key)}, expected {value}")
    entry["zarr_json"] = check(entry.get("zarr_json"), {"sha256": sha256_bytes(meta), "bytes": len(meta)}, "zarr.json", write)
    rows, cols = -(-zd.NUM_TIMESTEPS // zd.CHUNK), -(-zd.NUM_NEURONS // zd.CHUNK)
    keys = [(i, j) for i in range(rows) for j in range(cols)]
    started, counts = time.time(), {"cached": 0, "downloaded": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (_, status) in enumerate(pool.map(download_chunk, keys), 1):
            counts[status] += 1
            if n % 200 == 0:
                print(f"  chunks {n}/{len(keys)} ({time.time() - started:.0f} s)", flush=True)
    print(f"ZAPBench chunks: {counts}")
    manifest_lines, array_digest = [], hashlib.sha256()
    out = zd.zap_dir()
    stage = out / ".staging"  # built here, verified, then moved into place (running evaluations keep their files)
    targets = {}
    for split, (data_name, label_name) in zd.FILES.items():
        parts = zd.segments(split)
        steps = sum(stop - start for _, start, stop in parts)
        dtype = np.float32 if split == "test" else np.float16
        (stage / data_name).parent.mkdir(parents=True, exist_ok=True)
        array = np.lib.format.open_memmap(stage / data_name, mode="w+", dtype=dtype, shape=(steps, zd.NUM_NEURONS))
        labels, offset, placed = np.zeros(steps, dtype=np.int8), 0, []
        for condition, start, stop in parts:
            labels[offset:offset + stop - start] = condition
            placed.append((start, stop, offset))
            offset += stop - start
        np.save(stage / label_name, labels)
        targets[split] = (array, placed)
    max_f16_error = 0.0
    for i in range(rows):
        block = np.empty((zd.CHUNK, cols * zd.CHUNK), dtype="<f4")
        for j in range(cols):
            raw = chunk_path(i, j).read_bytes()
            manifest_lines.append(f"c/{i}/{j} {sha256_bytes(raw)}")
            block[:, j * zd.CHUNK:(j + 1) * zd.CHUNK] = np.frombuffer(raw, dtype="<f4").reshape(zd.CHUNK, zd.CHUNK)
        first, last = i * zd.CHUNK, min((i + 1) * zd.CHUNK, zd.NUM_TIMESTEPS)
        block = block[:last - first, :zd.NUM_NEURONS]
        if not np.all(np.isfinite(block)):
            raise SystemExit(f"non-finite values in timesteps {first}..{last}")
        array_digest.update(np.ascontiguousarray(block).tobytes())
        for split, (array, placed) in targets.items():
            for start, stop, offset in placed:
                lo, hi = max(start, first), min(stop, last)
                if lo < hi:
                    values = block[lo - first:hi - first]
                    array[offset + lo - start:offset + hi - start] = values.astype(array.dtype)
                    if array.dtype == np.float16:
                        max_f16_error = max(max_f16_error, float(np.max(np.abs(values.astype(np.float16).astype(np.float32) - values))))
    for array, _ in targets.values():
        array.flush()
    del array, targets
    entry["chunks"] = {"count": len(keys), "bytes_each": CHUNK_BYTES,
                       "manifest_sha256": check(entry.get("chunks", {}).get("manifest_sha256"),
                                                sha256_bytes("\n".join(manifest_lines).encode()), "chunk manifest", write)}
    entry["array_sha256"] = check(entry.get("array_sha256"), array_digest.hexdigest(), "float32 array", write)
    files = {}
    for split, (data_name, label_name) in zd.FILES.items():
        for name in (data_name, label_name):
            files[name] = {"sha256": sha256_file(stage / name), "bytes": (stage / name).stat().st_size}
    expected_files = entry.get("files") if not write else files
    if expected_files != files:
        raise SystemExit(f"prepared ZAPBench files differ from the lock: {files}")
    entry["files"] = files
    for name in files:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage / name, out / name)
    for directory in sorted({(stage / name).parent for name in files}, reverse=True) + [stage]:
        try:
            directory.rmdir()  # only empty staging directories are removed
        except OSError:
            pass
    manifest = {"source": zd.ZARR_URL, "license": "CC-BY-4.0", "context": zd.CONTEXT, "horizon": zd.HORIZON,
                "segments": {s: zd.segments(s) for s in zd.FILES}, "counts": zd.expected_counts(),
                "dtypes": {"train": "float16", "val": "float16", "test": "float32"},
                "max_float16_rounding_error": max_f16_error, "array_sha256": entry["array_sha256"], "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    entry["max_float16_rounding_error"] = max_f16_error
    print(f"ZAPBench splits written to {out} (max float16 rounding error {max_f16_error:.3g})")


REFERENCE_URL = "https://storage.googleapis.com/zapbench-release/dataframes/20250131/combined.json"


def check_reference(lock, write):
    """Reproduce ZAPBench's published 'mean' baseline (context 4) on the prepared test split.

    combined.json holds the per-condition, per-step test MAE of every method in the ZAPBench paper; the
    mean baseline repeats the mean of the 4 context steps. Prints the largest deviation over 9 x 32 values.
    """
    entry = lock.setdefault("zapbench_reference", {"url": REFERENCE_URL, "license": "CC-BY-4.0 (bucket README)",
                                                    "use": "validation only (reference check); not used by the evaluator"})
    cached = RAW / "zapbench-results" / "combined.json"
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(fetch(REFERENCE_URL)[0])
    raw = cached.read_bytes()
    entry.update(check({k: entry.get(k) for k in ("sha256", "bytes")}, {"sha256": sha256_bytes(raw), "bytes": len(raw)},
                       "combined.json", write))
    table = json.loads(json.loads(raw))
    rows = [k for k, method in table["method"].items() if method == "mean" and table["context"][k] == 4]
    data, labels = zd.load_split("test")
    starts = zd.window_starts(labels)
    sums, counts = {}, {}
    for begin in range(0, len(starts), 16):
        part = starts[begin:begin + 16]
        windows = np.asarray(data[(part[:, None] + np.arange(zd.CONTEXT + zd.HORIZON)).reshape(-1)], dtype=np.float32)
        windows = windows.reshape(len(part), zd.CONTEXT + zd.HORIZON, -1)
        err = np.abs(windows[:, :zd.CONTEXT].mean(axis=1, keepdims=True) - windows[:, zd.CONTEXT:]).sum(axis=2, dtype=np.float64)
        for row, start in enumerate(part):
            condition = zd.CONDITION_NAMES[int(labels[start])]
            sums[condition] = sums.get(condition, 0) + err[row]
            counts[condition] = counts.get(condition, 0) + 1
    local = {c: sums[c] / (counts[c] * zd.NUM_NEURONS) for c in sums}
    worst = max(abs(local[table["condition"][k]][int(table["steps_ahead"][k]) - 1] - table["MAE"][k]) for k in rows)
    score = float(np.mean([v.mean() for v in local.values()]))
    print(f"mean baseline: {len(rows)} published values, largest deviation {worst:.2e}; condition-averaged MAE {score:.7f}")
    if len(rows) != 9 * zd.HORIZON or worst > 1e-6:
        raise SystemExit("the prepared test split does not reproduce ZAPBench's published mean baseline")


def check_prepared(lock):
    """Hash the prepared files against the lock without network access."""
    problems = []
    for relative, expected in lock.get("limen", {}).get("files", {}).items():
        path = limen_dir() / relative
        if not path.is_file() or {"sha256": sha256_file(path), "bytes": path.stat().st_size} != expected:
            problems.append(f"limen-{LIMEN_COMMIT[:8]}/{relative}")
    for relative, expected in lock.get("zapbench", {}).get("files", {}).items():
        path = zd.zap_dir() / relative
        if not path.is_file() or {"sha256": sha256_file(path), "bytes": path.stat().st_size} != expected:
            problems.append(f"zapbench/{relative}")
    if problems:
        raise SystemExit("missing or modified prepared files: " + ", ".join(problems))
    print("all prepared jax_station files match the lock")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limen", action="store_true")
    parser.add_argument("--zapbench", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="parallel chunk downloads")
    parser.add_argument("--write-lock", action="store_true")
    parser.add_argument("--check", action="store_true", help="only hash the prepared files against the lock (no network)")
    parser.add_argument("--reference", action="store_true", help="reproduce ZAPBench's published mean baseline (1.3 MB download)")
    args = parser.parse_args(argv)
    lock = load_lock()
    if args.check:
        return check_prepared(lock)
    if not (args.limen or args.zapbench or args.reference):
        parser.error("choose --limen, --zapbench and/or --reference (or --check)")
    data_dir().mkdir(parents=True, exist_ok=True)
    if args.limen:
        prepare_limen(lock, args.write_lock)
    if args.zapbench:
        prepare_zapbench(lock, args.write_lock, args.workers)
    if args.reference:
        check_reference(lock, args.write_lock)
    if args.write_lock:
        LOCK.write_text(json.dumps(lock, indent=1, sort_keys=False) + "\n")
        print(f"wrote {LOCK}")


if __name__ == "__main__":
    main()
