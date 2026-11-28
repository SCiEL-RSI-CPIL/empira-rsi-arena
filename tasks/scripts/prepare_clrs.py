#!/usr/bin/env python3
"""Build the CLRS-30 files of the EvoPrompting CLRS tasks in data/clrs of the task cache.

Downloads the official CLRS-30 v1.0.0 archive (85.6 MB, the dataset that
clrs.create_dataset reads), checks its size and SHA-256 against
catalog/clrs-data-lock.json, and converts the TFRecord files with a small
built-in reader (no TensorFlow): one compressed <algorithm>.npz per algorithm
with the train split (all hint frames) and the val/test splits (first hint frame
only). Each file is verified against the content digest recorded in the lock
(a SHA-256 over array names, dtypes, shapes and bytes, so it does not depend on
the zip compressor). The task cache is ARENA_TASK_CACHE_DIR (default
~/.cache/empira/tasks); nothing is written inside the task library. Never run
during scoring. Needs numpy only:

    $PY scripts/prepare_clrs.py                  # build what is missing, verify everything
    $PY scripts/prepare_clrs.py --check          # verify only
    $PY scripts/prepare_clrs.py --archive CLRS30_v1.0.0.tar.gz   # use a local copy of the archive
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "catalog" / "clrs-data-lock.json"
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
from autoresearch_suite import clrs_data, clrs_specs as S  # noqa: E402

SPLITS = ("train", "val", "test")


# --- minimal TFRecord / tf.train.Example reader ---------------------------------------------

def _varint(buf, pos):
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def _fields(buf):
    """Yield (field number, payload) of a protobuf message (wire types 0, 1, 2, 5)."""
    pos, end = 0, len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 2:
            size, pos = _varint(buf, pos)
            yield number, buf[pos:pos + size]
            pos += size
        elif wire == 0:
            value, pos = _varint(buf, pos)
            yield number, value
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            yield number, buf[pos:pos + size]
            pos += size
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _decode_varints(payload):
    data = np.frombuffer(bytes(payload), dtype=np.uint8)
    ends = np.flatnonzero(data < 0x80)
    if len(ends) == len(data):
        return data.astype(np.int64)
    starts = np.concatenate([[0], ends[:-1] + 1])
    sizes = ends - starts + 1
    out = np.zeros(len(ends), dtype=np.uint64)
    for k in range(int(sizes.max())):
        ok = k < sizes
        out[ok] |= (data[starts[ok] + k] & 0x7F).astype(np.uint64) << np.uint64(7 * k)
    return out.view(np.int64)


def parse_example(record):
    """tf.train.Example -> {feature name: flat numpy array} (float_list and int64_list only)."""
    features = {}
    for number, message in _fields(record):
        if number != 1:
            continue
        for entry_number, entry in _fields(message):
            if entry_number != 1:
                continue
            key = value = None
            for field, payload in _fields(entry):
                if field == 1:
                    key = bytes(payload).decode()
                elif field == 2:
                    value = payload
            for kind, values in _fields(value):
                chunks = [bytes(chunk) for _, chunk in _fields(values)]
                if kind == 2:  # FloatList (packed or not): little-endian float32
                    features[key] = np.frombuffer(b"".join(chunks), dtype="<f4")
                elif kind == 3:  # Int64List: varints
                    features[key] = _decode_varints(b"".join(chunks)) if chunks else np.zeros(0, np.int64)
                else:
                    raise ValueError(f"{key}: unsupported tf.train.Feature kind {kind}")
    return features


def tfrecords(buffer):
    """Yield the records of a TFRecord file (length, masked CRC, data, masked CRC); CRCs not checked."""
    pos, end = 0, len(buffer)
    view = memoryview(buffer)
    while pos < end:
        (size,) = struct.unpack_from("<Q", buffer, pos)
        pos += 12
        yield view[pos:pos + size]
        pos += size + 4
    if pos != end:
        raise ValueError("truncated TFRecord file")


def read_archive(path, algorithms):
    """Yield (algorithm, split, {feature: array[samples, ...]}) with shards merged in file-name order."""
    shapes, counts, shards = {}, {}, {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")
            if len(parts) != 4 or not member.isfile():
                continue
            algorithm, split = parts[1].rsplit("_", 1)
            if algorithm not in algorithms:
                continue
            key = (algorithm, split)
            content = tar.extractfile(member).read()
            if parts[3] == "features.json":
                spec = json.loads(content)["featuresDict"]["features"]
                shapes[key] = {name: tuple(int(d) for d in item["tensor"]["shape"].get("dimensions", []))
                               for name, item in spec.items()}
            elif parts[3] == "dataset_info.json":
                counts[key] = [int(n) for n in json.loads(content)["splits"][0]["shardLengths"]]
            elif ".tfrecord-" in parts[3]:
                shards.setdefault(key, {})[parts[3]] = [parse_example(r) for r in tfrecords(content)]
            if key in shapes and key in counts and len(shards.get(key, {})) == len(counts[key]):
                parts_by_name = shards.pop(key)
                samples = [s for name in sorted(parts_by_name) for s in parts_by_name[name]]
                if len(samples) != sum(counts.pop(key)):
                    raise ValueError(f"{key}: sample count differs from dataset_info.json")
                yield algorithm, split, {n: np.stack([s[n].reshape(shape) for s in samples]) for n, shape in shapes.pop(key).items()}
    if shards or counts:
        raise ValueError(f"incomplete splits in the archive: {sorted(set(shards) | set(counts))}")


# --- conversion ----------------------------------------------------------------------------

def convert_split(algorithm, split, features):
    """Archive split -> {"<split>/<key>": compact array}, with lossless-cast and spec checks."""
    spec = S.SPECS[algorithm]
    expected = {f"{stage}_{name}" for name, (stage, _, _) in spec.items()} | {"lengths"}
    if set(features) != expected:
        raise ValueError(f"{algorithm}/{split}: features {sorted(set(features) ^ expected)} differ from the spec")
    out = {}
    lengths = features["lengths"]
    if not np.array_equal(lengths, np.round(lengths)) or lengths.min() < 1:
        raise ValueError(f"{algorithm}/{split}: lengths are not positive integers")
    out[f"{split}/lengths"] = lengths.astype(np.int16)
    for key, array in features.items():
        if key == "lengths":
            continue
        stage, name = key.split("_", 1)
        typ = spec[name][2]
        dtype = clrs_data.storage_dtype(typ)
        if stage == S.HINT and split != "train":
            array = array[:, :1]
        compact = array.astype(dtype)
        if not np.array_equal(compact.astype(np.float64), array.astype(np.float32).astype(np.float64)):
            raise ValueError(f"{algorithm}/{split}/{key}: values do not fit {np.dtype(dtype).name} losslessly")
        out[f"{split}/{key}"] = compact
    if split == "train" and algorithm in S.PRED_AS_INPUT:
        pred_h, steps = features["hint_pred_h"], lengths.astype(int)
        if any(np.any(pred_h[i, 1:steps[i]] != pred_h[i, 0]) for i in range(len(steps))):
            raise ValueError(f"{algorithm}: pred_h is not constant over a trajectory")
    return out


def write_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(name, **arrays)
        os.chmod(name, 0o644)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def summary(arrays):
    samples = {s: int(arrays[f"{s}/lengths"].shape[0]) for s in SPLITS}
    nodes = {s: int(arrays[f"{s}/input_pos"].shape[1]) for s in SPLITS}
    longest = {s: int(arrays[f"{s}/lengths"].max()) for s in SPLITS}
    return {"samples": samples, "nodes": nodes, "max_hint_frames": longest}


def fetch(source, directory):
    destination = Path(directory) / Path(source["url"]).name
    request = urllib.request.Request(source["url"], headers={"User-Agent": "autoresearch-tasks-pinned-data"})
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(request, timeout=300) as response, open(destination, "wb") as stream:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
            stream.write(chunk)
    if size != source["size_bytes"] or digest.hexdigest() != source["sha256"]:
        raise ValueError(f"{source['url']}: download does not match the locked size/SHA-256")
    return destination


def check_archive(path, source):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    if Path(path).stat().st_size != source["size_bytes"] or digest.hexdigest() != source["sha256"]:
        raise ValueError(f"{path} is not the locked CLRS-30 archive")


def dump_lock(lock):
    """The lock as JSON with one line per derived file."""
    head = json.dumps({key: value for key, value in lock.items() if key != "files"}, indent=1)
    rows = ",\n".join("  " + json.dumps(entry) for entry in lock["files"])
    return head[:-2] + ',\n "files": [\n' + rows + "\n ]\n}\n"


def verify(entry):
    path = clrs_data.data_dir() / entry["path"]
    if not path.is_file():
        return "missing"
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError):
        return "unreadable"
    return "verified" if clrs_data.content_digest(arrays) == entry.get("content_sha256") else "digest_mismatch"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="verify the prepared files only")
    parser.add_argument("--archive", help="local copy of CLRS30_v1.0.0.tar.gz instead of downloading")
    parser.add_argument("--only", help="comma-separated algorithm names")
    parser.add_argument("--write-lock", action="store_true", help="maintainers: record digests of freshly converted files")
    args = parser.parse_args(argv)
    lock = json.loads(LOCK.read_text())
    source = lock["sources"]["clrs30"]
    entries = {entry["path"][:-4]: entry for entry in lock["files"]}
    wanted = [a for a in S.ALGORITHMS if not args.only or a in args.only.split(",")]
    status = {a: verify(entries.get(a, {"path": f"{a}.npz"})) for a in wanted}
    todo = [a for a in wanted if status[a] != "verified" or args.write_lock]
    if todo and not args.check:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(args.archive) if args.archive else fetch(source, tmp)
            check_archive(archive, source)
            pending = {}
            for algorithm, split, features in read_archive(archive, set(todo)):
                pending.setdefault(algorithm, {}).update(convert_split(algorithm, split, features))
                if all(f"{s}/lengths" in pending[algorithm] for s in SPLITS):
                    arrays = pending.pop(algorithm)
                    write_npz(clrs_data.data_dir() / f"{algorithm}.npz", arrays)
                    if args.write_lock:
                        entries[algorithm] = {"path": f"{algorithm}.npz", "source": "clrs30",
                                              "content_sha256": clrs_data.content_digest(arrays), **summary(arrays)}
                    print(f"converted {algorithm}: {summary(arrays)['samples']}", flush=True)
            if pending:
                raise ValueError(f"splits missing from the archive: {sorted(pending)}")
        if args.write_lock:
            lock["files"] = [entries[a] for a in S.ALGORITHMS if a in entries]
            LOCK.write_text(dump_lock(lock))
        status = {a: verify(entries.get(a, {"path": f"{a}.npz"})) for a in wanted}
    for algorithm in wanted:
        print(f"{status[algorithm]:16s} {algorithm}")
    bad = [a for a in wanted if status[a] != "verified"]
    print(f"{len(wanted) - len(bad)}/{len(wanted)} CLRS-30 files verified in {clrs_data.data_dir()}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
