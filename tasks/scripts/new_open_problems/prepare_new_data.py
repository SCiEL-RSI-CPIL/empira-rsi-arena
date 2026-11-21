#!/usr/bin/env python3
"""Fetch the public instance files for the new tasks into the arnew data folder (ARNEW_DATA, else <ARENA_TASK_CACHE_DIR>/data/arnew/<family>/) and pin their SHA-256 in MANIFEST.json.

    python3 tasks/scripts/new_open_problems/prepare_new_data.py --family jssp               # dry run: lists what would be fetched, with sizes unknown
    python3 tasks/scripts/new_open_problems/prepare_new_data.py --family jssp --yes         # actually download

Nothing is downloaded without --yes. The URLs below follow each source's public layout; the CVRPLIB ids come from the site's instance page (read 2026-11-02). Families:
gset (71 small text files), cvrp (139 files, XL ones are large), jssp (one 819 KB file split into 85), qap (one archive),
miplib (212 .mps.gz, sizes vary, some large), epfl (40 .aig via 20 circuits). admet needs no files here (PyTDC downloads).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parents[1] / "suite" / "_lib"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LIB))
import registry_cvrp as RC  # noqa: E402
import registry_misc as RM  # noqa: E402
import registry_miplib as RP  # noqa: E402


def _data_root():
    """Prepared data lives outside the task library: ARNEW_DATA, else <ARENA_TASK_CACHE_DIR>/data/arnew."""
    import os

    from arena_task_paths import cache_path

    return Path(os.environ["ARNEW_DATA"]) if os.environ.get("ARNEW_DATA") else cache_path("data", "arnew")


DATA = _data_root()
JSSP_URL = "https://raw.githubusercontent.com/thomasWeise/jsspInstancesAndResults/master/data-raw/instance-data/instance_data.txt"
QAP_URL = "https://datashare.ed.ac.uk/bitstreams/7dc05fa0-a4dc-4530-9ce4-57b79ffffc4f/download"


def plan(family):
    """(relative file name, url) pairs for direct downloads."""
    if family == "gset":
        return [(g, f"https://web.stanford.edu/~yyye/yyye/Gset/{g}") for g in RM.GSET]
    if family == "cvrp":
        ids = json.loads((HERE / "cvrplib_ids.json").read_text())
        return [(f"{n}.vrp", f"https://galgos.inf.puc-rio.br/cvrplib/index.php/en/download/instance/{ids[n]}")
                for n, _ in RC.X_INSTANCES + RC.XL_INSTANCES]
    if family == "miplib":
        return [(f"{n}.mps.gz", f"https://miplib.zib.de/WebData/instances/{n}.mps.gz") for n in RP.OPEN]
    if family == "epfl":
        return [(f"{stem}.aig", f"https://raw.githubusercontent.com/lsils/benchmarks/master/{folder}/{stem}.aig")
                for folder, stem, _, _ in RM.EPFL.values()]
    if family == "sphere":
        # two tables of best known codes: Sloane's (1994-96, never updated) and Cohn's (spherical-codes.org, updated continuously).
        # Cohn's table has no entry for n <= 2 * dim (solved, Rankin 1955)
        return ([(f"pack.{d}.{n}.txt", f"https://neilsloane.com/packings/dim{d}/pack.{d}.{n}.txt") for d in (3, 4, 5) for n in range(4, 131)]
                + [(f"cohn.{d}.{n}.csv", f"https://spherical-codes.org/data/{d}/{n}") for d in (3, 4, 5) for n in range(2 * d + 1, 131)])
    if family == "thomson":
        return [("thomson_problem.wikitext", "https://en.wikipedia.org/w/index.php?title=Thomson_problem&action=raw")]
    if family == "dnacode":
        return [("dnacodes.html", "https://rambutan.umassmed.edu/king/dnacodes.html")]
    if family == "matbench":
        return ([(f"raw/matbench_{t}.json.gz", f"https://ml.materialsproject.org/projects/matbench_{t}.json.gz") for t in MATBENCH_NEW]
                + [("raw/validation.json", MATBENCH_VALIDATION_URL)])
    return []


MATBENCH_NEW = ("glass", "log_kvrh", "mp_gap", "mp_is_metal", "mp_e_form")
MATBENCH_VALIDATION_URL = ("https://raw.githubusercontent.com/materialsproject/matbench/936176db18ca4cd7b38cbd957c017a5bac770c6b/matbench/"
                           "matbench_v0.1_validation.json")


def derive_matbench(folder):
    """Write data/matbench/<task>.json in the old suite's format, using its own deriver (autoresearch_suite.chem_sources)."""
    from autoresearch_suite import chem_sources as cs

    cs.MATBENCH.update({"glass": ("composition", "gfa", "1 = glass former, 0 = not"), "log_kvrh": ("structure", "log10(K_VRH)", "log10(GPa)"),
                        "mp_gap": ("structure", "gap pbe", "eV"), "mp_is_metal": ("structure", "is_metal", "1 = metal, 0 = non-metal"),
                        "mp_e_form": ("structure", "e_form", "eV/atom")})
    cs.CLASSIFICATION.update({"glass", "mp_is_metal"})
    validation = (folder / "raw" / "validation.json").read_bytes()
    made = []
    for t in MATBENCH_NEW:
        payloads = {f"matbench-{t.replace('_', '-')}": (folder / "raw" / f"matbench_{t}.json.gz").read_bytes(), "matbench-validation": validation}
        (folder / f"{t}.json").write_bytes(cs.matbench_deriver(t)(payloads))
        made.append(f"{t}.json")
    return made


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; autoresearch-tasks-new/0.1)"}), timeout=120) as r:
        return r.read()


def fetch_to_file(url, path):
    """Stream a (possibly large) download to disk and return its SHA-256."""
    digest = hashlib.sha256()
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; autoresearch-tasks-new/0.1)"}), timeout=120) as r, open(tmp, "wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
    tmp.rename(path)
    return digest.hexdigest()


def split_jssp(raw, folder):
    text = raw.decode("utf-8", "replace")
    pieces = re.split(r"\n\s*\+{10,}\s*\n\s*instance (\S+)\s*\n\s*\+{10,}\s*\n", text)
    out = {}
    for name, body in zip(pieces[1::2], pieces[2::2]):
        if name not in RM.JSSP:
            continue
        lines = body.strip("\n").splitlines()
        i = next(k for k, l in enumerate(lines) if re.fullmatch(r"\s*\d+\s+\d+\s*", l))
        n = int(lines[i].split()[0])
        (folder / f"{name}.txt").write_text("\n".join(l.strip() for l in lines[i:i + 1 + n]) + "\n")
        out[name] = (lines[i - 1].strip() if i else "")
    missing = sorted(set(RM.JSSP) - set(out))
    if missing:
        print("not found in the data file:", missing)
    return out


def extract_qap(raw, folder):
    names = set(RM.QAP)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
        for m in tar.getmembers():
            stem = Path(m.name).name
            if stem.endswith(".dat") and stem[:-4] in names:
                (folder / stem).write_bytes(tar.extractfile(m).read())
    return sorted(p.name for p in folder.glob("*.dat"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["gset", "cvrp", "jssp", "qap", "miplib", "epfl", "admet", "sphere", "thomson", "dnacode", "matbench", "none"])
    ap.add_argument("--yes", action="store_true", help="actually download (otherwise only list)")
    ap.add_argument("--only", help="comma-separated file names to fetch (direct-download families)")
    a = ap.parse_args()
    if a.family in ("admet", "none"):
        print("No files to prepare: PyTDC downloads the ADMET data itself (needs PyTDC and rdkit).")
        return
    folder = DATA / a.family
    items = plan(a.family)
    if a.only:
        want = set(a.only.split(","))
        items = [(n, u) for n, u in items if n in want]
    if a.family == "jssp":
        items = [("instance_data.txt", JSSP_URL)]
    if a.family == "qap":
        items = [("qapdata archive", QAP_URL)]
    print(f"{a.family}: {len(items)} download(s) into {folder}")
    for n, u in items[:6]:
        print("  ", n, u)
    if len(items) > 6:
        print("   ...")
    if not a.yes:
        print("dry run; pass --yes to download")
        return
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    failed = []
    if a.family == "jssp":
        raw = fetch(JSSP_URL)
        split_jssp(raw, folder)
        for p in folder.glob("*.txt"):
            manifest[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    elif a.family == "qap":
        raw = fetch(QAP_URL)
        for name in extract_qap(raw, folder):
            manifest[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    else:
        import concurrent.futures as cf

        def one(item):
            name, url = item
            try:
                return name, fetch_to_file(url, folder / name), None
            except Exception as error:  # noqa: BLE001
                return name, None, str(error)

        for name, _ in items:  # files fetched earlier by hand: pin them instead of downloading again
            if (folder / name).is_file() and name not in manifest:
                manifest[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        (folder / "raw").mkdir(exist_ok=True) if a.family == "matbench" else None
        todo = [it for it in items if not (folder / it[0]).is_file()]
        with cf.ThreadPoolExecutor(6) as ex:
            for name, digest, error in ex.map(one, todo):
                if error:
                    failed.append((name, error))
                else:
                    manifest[name] = digest
                print("ok" if not error else "FAILED", name, flush=True)
    if a.family == "matbench" and not failed:
        for name in derive_matbench(folder):
            manifest[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"done: {len(manifest)} files pinned in {manifest_path}")
    for name, err in failed:
        print("FAILED", name, err)


if __name__ == "__main__":
    main()
