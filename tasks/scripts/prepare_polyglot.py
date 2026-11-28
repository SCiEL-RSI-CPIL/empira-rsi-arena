#!/usr/bin/env python3
"""Download, verify and prepare the Aider Polyglot exercises for the polyglot evaluator.

    python scripts/prepare_polyglot.py                 # download (cached), verify, write data/polyglot in the task cache
    python scripts/prepare_polyglot.py --verify-only   # check the prepared data against the lock
    python scripts/prepare_polyglot.py --build-image   # also build the local Docker toolchain image (network needed)
    python scripts/prepare_polyglot.py --update-lock   # maintainers: record the hashes of the current sources

Sources (catalog/polyglot-data-lock.json): the Aider-AI/polyglot-benchmark repository at a pinned commit (GitHub
archive; the extracted content is verified file by file because GitHub does not guarantee byte-stable archives) and
DGM's exercise subsets (polyglot/subsets/small.json and medium.json at a pinned commit). The prepared directory holds
one directory per exercise (``exercises/<language>/<slug>``, without ``.approaches`` and ``.articles``), the two
subset files and ``manifest.json``. Raw downloads are cached in ``polyglot-raw/pinned`` under the task cache.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from arena_task_paths import cache_path  # noqa: E402
from autoresearch_suite import polyglot_data as pd  # noqa: E402

LOCK = ROOT / "catalog" / "polyglot-data-lock.json"
CACHE = cache_path("polyglot-raw") / "pinned"
ARCHIVE_URL = f"https://codeload.github.com/Aider-AI/polyglot-benchmark/tar.gz/{pd.BENCHMARK_COMMIT}"
SUBSET_URL = "https://raw.githubusercontent.com/jennyzzt/dgm/{commit}/polyglot/subsets/{name}.json"
BENCHMARK_LICENSE = ("MIT: every exercise comes from an Exercism language track (exercism/cpp, go, java, javascript, python, rust; "
                     "each track's LICENSE is MIT) and the problem descriptions from exercism/problem-specifications (MIT). "
                     "The polyglot-benchmark repository itself has no LICENSE file; its README states that the exercises are "
                     "used in accordance with Exercism's open source licenses. Not redistributed by this repository.")
DGM_LICENSE = "Apache-2.0 (jennyzzt/dgm LICENSE). Not redistributed by this repository."
# Lock fields that name a location give it relative to the task cache.
CACHE_NOTE = "of the task cache (ARENA_TASK_CACHE_DIR, default ~/.cache/empira/tasks)"


def sha256_file(path):
    return pd.file_sha256(path)


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    last = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "autoresearch-tasks-polyglot"})
            with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as handle:
                for block in iter(lambda: response.read(1 << 20), b""):
                    handle.write(block)
            partial.replace(target)
            return target
        except OSError as error:
            last = error
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def load_lock():
    return json.loads(LOCK.read_text()) if LOCK.is_file() else None


def lock_source(lock, source_id):
    if not lock:
        return None
    return next((row for row in lock["sources"] if row["id"] == source_id), None)


def fetch(url, name):
    target = CACHE / name
    if not target.is_file():
        print(f"downloading {url}", flush=True)
        download(url, target)
    return target


def extract_benchmark(archive, workdir):
    """Extract the archive into ``workdir`` (data filter: no links out, no absolute paths); return the repository root."""
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(workdir, filter="data")
    roots = [p for p in Path(workdir).iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"unexpected archive layout: {[p.name for p in roots]}")
    return roots[0]


def exercise_dirs(repo):
    for language in pd.LANGUAGES:
        practice = repo / language / "exercises" / "practice"
        for directory in sorted(p for p in practice.iterdir() if p.is_dir()):
            yield language, directory


def derive(repo, subsets, out):
    """Write exercises/<language>/<slug>, subsets/*.json and manifest.json into ``out`` (replaced)."""
    if out.exists():
        shutil.rmtree(out)
    records = {}
    for language, directory in exercise_dirs(repo):
        slug = directory.name
        target = out / "exercises" / language / slug
        files = [rel for rel in pd.list_files(directory) if rel.split("/", 1)[0] not in pd.EXCLUDED_DIRS]
        for rel in files:
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / rel, target / rel)
        config = json.loads((directory / ".meta" / "config.json").read_text())
        records[f"{language}__{slug}"] = {"language": language, "slug": slug, "files": config.get("files", {}),
                                          "sha256": pd.tree_digest(target, files)}
    (out / "subsets").mkdir(parents=True)
    for name, path in subsets.items():
        ids = json.loads(path.read_text())
        missing = [i for i in ids if i not in records]
        if missing:
            raise RuntimeError(f"subset {name} names exercises that are not in the benchmark: {missing}")
        shutil.copyfile(path, out / "subsets" / f"{name}.json")
    manifest = {"schema": "artask.polyglot-manifest/1",
                "benchmark": {"repository": pd.BENCHMARK_REPOSITORY, "commit": pd.BENCHMARK_COMMIT},
                "dgm": {"repository": pd.DGM_REPOSITORY, "commit": pd.DGM_COMMIT},
                "subsets": {name: json.loads(path.read_text()) for name, path in subsets.items()},
                "exercises": records}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def derived_digest(out):
    return pd.tree_digest(out, pd.list_files(out))


def verify_prepared(lock):
    out = pd.data_dir()
    if not (out / "manifest.json").is_file():
        raise SystemExit(f"prepared data missing: {out}")
    digest = derived_digest(out)
    expected = lock["derived"]["sha256"]
    if digest != expected:
        raise SystemExit(f"prepared data digest {digest} != lock {expected}; rerun scripts/prepare_polyglot.py")
    manifest = json.loads((out / "manifest.json").read_text())
    print(f"verified {out}: {len(manifest['exercises'])} exercises, subsets " +
          ", ".join(f"{k} ({len(v)})" for k, v in manifest["subsets"].items()) + f", digest {digest[:16]}")


def prepare(update_lock):
    lock = load_lock()
    if lock is None and not update_lock:
        raise SystemExit(f"{LOCK} is missing; run with --update-lock to create it")
    archive = fetch(ARCHIVE_URL, f"polyglot-benchmark-{pd.BENCHMARK_COMMIT}.tar.gz")
    subsets = {name: fetch(SUBSET_URL.format(commit=pd.DGM_COMMIT, name=name), f"dgm-{pd.DGM_COMMIT[:12]}-{name}.json")
               for name in ("small", "medium")}
    rows = []
    with tempfile.TemporaryDirectory(prefix="polyglot-extract-") as tmp:
        repo = extract_benchmark(archive, tmp)
        content = pd.tree_digest(repo, pd.list_files(repo))
        rows.append({"id": "polyglot-benchmark", "url": ARCHIVE_URL, "sha256": sha256_file(archive), "size": archive.stat().st_size,
                     "content_sha256": content, "files": len(pd.list_files(repo)), "license": BENCHMARK_LICENSE,
                     "target": f"polyglot-raw/pinned/{archive.name}"})
        for name, path in subsets.items():
            rows.append({"id": f"dgm-subset-{name}", "url": SUBSET_URL.format(commit=pd.DGM_COMMIT, name=name),
                         "sha256": sha256_file(path), "size": path.stat().st_size, "license": DGM_LICENSE,
                         "target": f"data/polyglot/subsets/{name}.json"})
        if not update_lock:
            for row in rows:
                locked = lock_source(lock, row["id"])
                if locked is None:
                    raise SystemExit(f"{row['id']} is not in the lock")
                key = "content_sha256" if "content_sha256" in row else "sha256"
                if row[key] != locked[key]:
                    raise SystemExit(f"{row['id']}: {key} {row[key]} != lock {locked[key]}")
                if key == "content_sha256" and row["sha256"] != locked["sha256"]:
                    print(f"note: the GitHub archive bytes changed ({row['sha256'][:12]} vs {locked['sha256'][:12]}); its content matches the lock")
        manifest = derive(repo, subsets, pd.data_dir())
    digest = derived_digest(pd.data_dir())
    if update_lock:
        lock = {"schema": "artask.data-lock/1", "package": "polyglot", "reviewed_on": time.strftime("%Y-%m-%d"),
                "raw_cache": "polyglot-raw/pinned " + CACHE_NOTE, "data_directory": "data/polyglot " + CACHE_NOTE,
                "note": ("Raw files are fetched from pinned URLs. The benchmark archive is verified by the SHA-256 of its extracted "
                         "content (sorted '<path> <sha256>' lines), because GitHub does not guarantee byte-stable archives; the "
                         "archive's own SHA-256 is recorded for reference. Derived files are byte-stable functions of the sources "
                         "(scripts/prepare_polyglot.py). Nothing listed here is redistributed by this repository."),
                "sources": rows,
                "derived": {"sha256": digest, "exercises": len(manifest["exercises"]),
                            "by_language": {lang: sum(1 for r in manifest["exercises"].values() if r["language"] == lang) for lang in pd.LANGUAGES},
                            "subsets": {k: len(v) for k, v in manifest["subsets"].items()},
                            "size_bytes": sum(p.stat().st_size for p in pd.data_dir().rglob("*") if p.is_file())}}
        LOCK.write_text(json.dumps(lock, indent=2) + "\n")
        print(f"wrote {LOCK}")
    verify_prepared(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--update-lock", action="store_true")
    parser.add_argument("--build-image", action="store_true", help="build the Docker toolchain image after preparing the data")
    args = parser.parse_args()
    if args.verify_only:
        lock = load_lock()
        if lock is None:
            raise SystemExit(f"{LOCK} is missing")
        verify_prepared(lock)
    else:
        prepare(args.update_lock)
    if args.build_image:
        from autoresearch_suite import polyglot_image
        pd.manifest.cache_clear()
        with tempfile.TemporaryDirectory(prefix="polyglot-image-") as context:
            record = polyglot_image.build(Path(context))
        print(json.dumps(record))


if __name__ == "__main__":
    main()
