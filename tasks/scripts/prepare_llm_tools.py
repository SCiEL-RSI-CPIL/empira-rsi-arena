#!/usr/bin/env python3
"""Download, verify and derive the data used by the llm_tools evaluators.

    python scripts/prepare_llm_tools.py                  # every part
    python scripts/prepare_llm_tools.py telemetry autoda # selected parts
    python scripts/prepare_llm_tools.py --verify-only    # check prepared files only

Parts: telemetry (ADRS Abilene snapshot), autoda (CIFAR-10 test images and the
ResNet-18 target), hover (HoVer claims, 2017 Wikipedia abstracts and their
BM25 index).  Every download must match the SHA-256 and size in
``catalog/llm_tools-data-lock.json``; derived files are
written to ``data/llm_tools`` in the task cache and hash-checked where
they are deterministic.  Raw downloads are cached in ``llm_tools-raw`` under the task cache.
Nothing prepared here is redistributed by the repository.
``--update-lock`` (maintainers only) records the hashes of the current sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from arena_task_paths import cache_path  # noqa: E402
from autoresearch_suite.llm_tools_common import data_dir  # noqa: E402

LOCK = ROOT / "catalog" / "llm_tools-data-lock.json"
RAW = cache_path("llm_tools-raw")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

ADRS = "https://raw.githubusercontent.com/UCB-ADRS/ADRS/259292f3ad1924c03be684996dbd83e6891c3e39/openevolve/examples/ADRS/telemetry_repair"
SOURCES = {
    "telemetry-evaluation-data": {
        "url": f"{ADRS}/ref_data/evaluation_data.csv", "part": "telemetry", "target": "telemetry/evaluation_data.csv",
        "license": "Apache-2.0 (UCB-ADRS/ADRS repository licence); the provenance licence of the underlying Abilene "
                   "measurements is not stated. Not redistributed by this repository."},
    "cifar10-binary": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", "part": "autoda",
        "license": "not stated (the CIFAR-10 page asks users to cite Krizhevsky, 'Learning Multiple Layers of Features from "
                   "Tiny Images', 2009); downloaded at setup only and never redistributed."},
    "resnet18-cifar10": {
        "url": "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/ccacbae365101b9bfa45e1f17f4acac30c2601b2/pytorch_model.bin",
        "part": "autoda", "license": "MIT (model card of edadaltocg/resnet18_cifar10). Not redistributed by this repository."},
    "hover-train": {
        "url": "https://raw.githubusercontent.com/hover-nlp/hover/39b84697f196308f398a251a7aea9b82ae0f0562/data/hover/hover_train_release_v1.1.json",
        "part": "hover", "license": "CC-BY-SA-4.0 (HoVer data, Hugging Face dataset card hover-nlp/hover); repository code MIT. "
                                    "Not redistributed by this repository."},
    "wiki-abstracts-2017": {
        "url": "https://huggingface.co/dspy/cache/resolve/ef6a5e72a98b47cef31574a400fea8fe149559a3/wiki.abstracts.2017.tar.gz",
        "part": "hover", "license": "not stated by the dspy/cache repository; the abstracts are Wikipedia text (CC BY-SA). "
                                    "Prepared locally, never committed or redistributed."},
}
AUTODA_SEED = 20240127  # fixes the held-out sample and the starting-point pools


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, headers=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    last = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "autoresearch-tasks-llm-tools", **(headers or {})})
            with urllib.request.urlopen(request, timeout=600) as response, partial.open("wb") as handle:
                for block in iter(lambda: response.read(1 << 20), b""):
                    handle.write(block)
            partial.replace(target)
            return
        except OSError as error:
            last = error
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def load_lock():
    return json.loads(LOCK.read_text()) if LOCK.is_file() else {"schema_version": "1.0", "package": "llm_tools", "sources": {}, "derived": {}}


def fetch(name, lock, update):
    """Download one locked source into the raw cache (or reuse it) and return its path."""
    spec = SOURCES[name]
    path = RAW / name / Path(spec["url"].split("?")[0]).name
    row = lock["sources"].get(name)
    if path.is_file() and row and path.stat().st_size == row["size"] and sha256_file(path) == row["sha256"]:
        return path
    if not (path.is_file() and update and not row):  # an unlocked cached copy is recorded as it is when re-locking
        print(f"downloading {name}: {spec['url']}", flush=True)
        download(spec["url"], path, spec.get("headers"))
    digest, size = sha256_file(path), path.stat().st_size
    if update:
        lock["sources"][name] = {"url": spec["url"], "sha256": digest, "size": size, "license": spec["license"]}
    elif not row:
        raise SystemExit(f"{name} is not in the lock; run with --update-lock")
    elif (digest, size) != (row["sha256"], row["size"]):
        raise SystemExit(f"{name}: download differs from the lock ({digest}, {size} bytes)")
    return path


def record(lock, update, relative):
    """Hash-check (or record) one derived file under data_dir()."""
    path = data_dir() / relative
    digest = sha256_file(path)
    if update:
        lock["derived"][relative] = {"sha256": digest, "size": path.stat().st_size}
    elif lock["derived"].get(relative, {}).get("sha256") != digest:
        raise SystemExit(f"derived file {relative} differs from the lock")


def copy_source(name, lock, update):
    raw = fetch(name, lock, update)
    target = data_dir() / SOURCES[name]["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw.read_bytes())
    record(lock, update, SOURCES[name]["target"])


def prepare_telemetry(lock, update):
    copy_source("telemetry-evaluation-data", lock, update)


def cifar_test_batch(archive):
    """CIFAR-10 test images (10000, 3, 32, 32) uint8 and labels from the binary archive (no pickle involved)."""
    import tarfile
    import numpy as np
    with tarfile.open(archive, "r:gz") as tar:
        raw = tar.extractfile(tar.getmember("cifar-10-batches-bin/test_batch.bin")).read()
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(10000, 3073)
    return rows[:, 1:].reshape(10000, 3, 32, 32).copy(), rows[:, 0].astype(np.int64)


def prepare_autoda(lock, update):
    """Target model, model accuracy, the paper's 8 evolution images and a class-balanced held-out sample with starting points."""
    import numpy as np
    from autoresearch_suite import llm_tools_autoda as autoda
    images, labels = cifar_test_batch(fetch("cifar10-binary", lock, update))
    model_target = data_dir() / autoda.MODEL_FILE
    model_target.parent.mkdir(parents=True, exist_ok=True)
    model_target.write_bytes(fetch("resnet18-cifar10", lock, update).read_bytes())
    record(lock, update, autoda.MODEL_FILE)
    model = autoda.load_model(4)
    predictions = autoda.classify(model, images.astype(np.float32) / np.float32(255.0))
    correct = predictions == labels
    print(f"ResNet-18 test accuracy on the 10,000 CIFAR-10 test images: {correct.mean():.4f}", flush=True)
    rng = np.random.default_rng(AUTODA_SEED)
    evolution = [int(i) for i in range(8) if correct[i]]
    order = [int(i) for i in rng.permutation(np.arange(8, 10000)) if correct[i]]
    per_class = {c: [i for i in order if labels[i] == c][:10] for c in range(10)}
    heldout = [per_class[c][k] for k in range(10) for c in range(10)]
    used = set(evolution) | set(heldout)
    pool = np.array([i for i in range(10000) if i not in used])
    rows, sets = [], {}
    for name, targets in (("evolution", evolution), ("heldout", heldout)):
        sets[name] = []
        for index in targets:
            draws = [int(j) for j in rng.choice(pool, size=60, replace=False) if predictions[j] != labels[index]][:10]
            gaps = [float(np.linalg.norm(images[j].astype(np.float64) - images[index].astype(np.float64)) / 255.0) for j in draws]
            start = draws[int(np.argmin(gaps))]
            sets[name].append({"index": index, "label": int(labels[index]), "start_index": start, "start_prediction": int(predictions[start]),
                               "row": len(rows), "start_row": len(rows) + 1, "start_l2": round(min(gaps), 6)})
            rows += [index, start]
    out = data_dir() / "autoda"
    np.save(out / "images.npy", images[rows])
    meta = {"source": "CIFAR-10 test batch (binary version)", "model_test_accuracy": round(float(correct.mean()), 4),
            "selection_seed": AUTODA_SEED, "sets": sets,
            "notes": "evolution = the first eight test images that the model classifies correctly; heldout = 10 correctly classified "
                     "test images per class (index >= 8, seeded order, interleaved by class); each starting point is the nearest (L2) "
                     "of 10 seeded test images that the model assigns to a different class than the target's label."}
    (out / "attack_sets.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    for name in ("autoda/images.npy", "autoda/attack_sets.json"):
        record(lock, update, name)


def gepa_hover_split(rows):
    """GEPA artifact (hover_data.py + benchmark.py): 3-document claims, Random(0) shuffle, 40/40/20 split, Random(1) trims."""
    import random
    three = [row for row in rows if len({fact[0] for fact in row["supporting_facts"]}) == 3]
    random.Random(0).shuffle(three)
    n = len(three)
    test, val, train = three[:int(0.4 * n)], three[int(0.4 * n):int(0.8 * n)], three[int(0.8 * n):]
    trim = lambda part, size: part if size >= len(part) else random.Random(1).sample(part, size)  # noqa: E731
    return {"three_document_claims": n, "test_pool": len(test), "val_pool": len(val), "train_pool": len(train)}, trim(test, 300)


def prepare_hover(lock, update):
    """GEPA's 300 HoVer test claims, the 2017 abstracts as 'title | text' documents and their bm25s index."""
    import subprocess
    import tarfile
    from autoresearch_suite import llm_tools_hover as hover
    rows = json.loads(fetch("hover-train", lock, update).read_text(encoding="utf-8"))
    sizes, test = gepa_hover_split(rows)
    out = data_dir() / "hover"
    out.mkdir(parents=True, exist_ok=True)
    claims = [{"uid": row["uid"], "claim": row["claim"], "label": row["label"], "num_hops": row["num_hops"],
               "gold_titles": list(dict.fromkeys(fact[0] for fact in row["supporting_facts"]))} for row in test]
    index_dir = out / "index"
    archive = fetch("wiki-abstracts-2017", lock, update)  # always checked against the lock, also when the index exists
    if not (index_dir / "bm25s" / "params.index.json").is_file():
        extracted = RAW / "wiki-abstracts-2017" / "wiki.abstracts.2017.jsonl"
        if not extracted.is_file():
            with tarfile.open(archive, "r:gz") as tar:
                member = next(m for m in tar.getmembers() if m.isfile() and Path(m.name).name == "wiki.abstracts.2017.jsonl")
                with tar.extractfile(member) as source, extracted.open("wb") as target:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        target.write(block)
        print("building the bm25s index (one process, PYTHONHASHSEED=0 for a byte-identical index; a few minutes and several GB of memory)", flush=True)
        env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "4"}
        subprocess.run([sys.executable, "-m", "autoresearch_suite.llm_tools_hover", str(extracted), str(index_dir)],
                       check=True, env=env, cwd=str(ROOT / "suite" / "_lib"))
    corpus = hover.Corpus(index_dir)
    titles = set()
    with corpus.path.open(encoding="utf-8") as handle:
        for line in handle:
            titles.add(hover.normalize_text(hover.doc_title(json.loads(line))))
    gold = [hover.normalize_text(t) for c in claims for t in c["gold_titles"]]
    coverage = sum(t in titles for t in gold) / len(gold)
    print(f"{len(corpus)} documents; {coverage:.4f} of the {len(gold)} gold titles of the 300 test claims occur as document titles", flush=True)
    meta = {"split": sizes, "test": claims, "documents": len(corpus), "gold_title_coverage": round(coverage, 6),
            "notes": "test = GEPA's HoVer test split (300 three-document claims of the HoVer train release)."}
    (out / "claims.json").write_text(json.dumps(meta, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in ("claims.json", "index/docs.jsonl", "index/doc_offsets.npy", "index/bm25s/params.index.json", "index/bm25s/vocab.index.json",
                 "index/bm25s/data.csc.index.npy", "index/bm25s/indices.csc.index.npy", "index/bm25s/indptr.csc.index.npy"):
        record(lock, update, "hover/" + name)


PARTS = {"telemetry": prepare_telemetry, "autoda": prepare_autoda, "hover": prepare_hover}


def verify(lock, parts):
    bad = []
    for relative, row in sorted(lock.get("derived", {}).items()):
        if relative.split("/")[0] not in parts:
            continue
        path = data_dir() / relative
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            bad.append(relative)
    print("verified" if not bad else "missing or modified: " + ", ".join(bad))
    return not bad


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("parts", nargs="*", help=f"parts to prepare (default: all of {', '.join(PARTS)})")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--update-lock", action="store_true")
    args = parser.parse_args(argv)
    parts = args.parts or list(PARTS)
    unknown = sorted(set(parts) - set(PARTS))
    if unknown:
        parser.error("unknown parts: " + ", ".join(unknown))
    lock = load_lock()
    if args.verify_only:
        return 0 if verify(lock, parts) else 1
    for part in parts:
        print(f"== {part}", flush=True)
        PARTS[part](lock, args.update_lock)
    if args.update_lock:  # merge into the current file so that runs for different parts do not overwrite each other
        current = load_lock()
        for key in ("sources", "derived"):
            current[key].update(lock[key])
        LOCK.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
