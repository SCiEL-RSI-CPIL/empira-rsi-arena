#!/usr/bin/env python3
"""Fetch the pinned public inputs of the agent_bench evaluators and set up their environment runtime.

Raw files listed in catalog/agent_bench-data-lock.json are downloaded from their
pinned URLs, verified by SHA-256, cached in agent_bench-raw/pinned of the task
cache (ARENA_TASK_CACHE_DIR, default ~/.cache/empira/tasks) and copied or derived
into data/agent_bench of the task cache. Nothing is redistributed by this
repository. Components:

    m3tooleval    CodeAct's M3ToolEval task package (MIT)
    agentboard    AgentBoard labels for ScienceWorld and PDDL (data GPL-2.0) and the PDDL domains/problems (code Apache-2.0)
    travel        TravelPlanner flights restricted to the cities and dates of the validation queries (CC-BY-4.0)
    minihack      Boxoban levels for MiniHack's Boxoban tasks (Apache-2.0)
    alfworld      ALFWorld's out-of-distribution evaluation games (MIT)

    python3 scripts/prepare_agent_bench.py                       # all data components
    python3 scripts/prepare_agent_bench.py m3tooleval            # one component
    python3 scripts/prepare_agent_bench.py --setup-environment   # pinned Python 3.11 runtime, agent_bench-env of the task cache
    python3 scripts/prepare_agent_bench.py --update-lock         # maintainers: record new hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from arena_task_paths import cache_path, data_root  # noqa: E402

LOCK = ROOT / "catalog" / "agent_bench-data-lock.json"
RAW = cache_path("agent_bench-raw") / "pinned"
DATA = data_root() / "agent_bench"
ENV = cache_path("agent_bench-env")
ENV_PYTHON = "python3.11"
GIT = "https://github.com"
# Pinned runtime of the environment servers (BALROG pins gym 0.23 and balrog-nle, which has macOS wheels up to CPython 3.11).
ENV_REQUIREMENTS = [
    {"args": [], "packages": ["numpy==1.26.4", "scipy==1.13.1", "pillow==10.4.0", "gym==0.23.0", "gymnasium==0.29.1", "pygame==2.6.1",
                              "matplotlib==3.9.2", "opencv-python-headless==4.10.0.84", "imageio==2.35.1", "opensimplex==0.4.5.1",
                              "ruamel.yaml==0.18.6", "typing_extensions", "pybind11==2.13.6", "scikit-image==0.24.0", "nltk==3.9.1",
                              "py4j==0.10.9.7", "scienceworld==1.2.1", "setuptools==75.8.0",
                              "textworld[pddl]==1.7.0", "alfworld==0.4.2"]},
    {"args": ["--no-deps"], "packages": ["crafter==1.8.3", "balrog-nle==0.9.0",
                                         f"minigrid @ {GIT}/BartekCupial/Minigrid/archive/cf73dd148e51276bd675a37e3bfb0bf2b42329b2.zip",
                                         f"baba @ {GIT}/nacloos/baba-is-ai/archive/40ae1b29dbc67f2ecea06cc161ed5ca51719e5e3.zip",
                                         f"minihack @ {GIT}/balrog-ai/minihack/archive/3ecb6da4eadcac7a4dbc5f8e801ccf636e34d7ec.zip",
                                         f"balrog @ {GIT}/balrog-ai/BALROG/archive/b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5.zip"]},
]


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def fetch(source, update):
    cached = RAW / source["id"]
    payload = cached.read_bytes() if cached.is_file() else None
    if payload is None or update or sha256(payload) != source.get("sha256"):
        request = urllib.request.Request(source["url"], headers={"User-Agent": "autoresearch-tasks agent_bench data (pinned)"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=900) as response:
                    payload = response.read()
                break
            except OSError as error:
                if attempt == 3:
                    raise RuntimeError(f"download failed for {source['id']}: {error}") from error
                time.sleep(5 * (attempt + 1))
    if update:
        source.update(sha256=sha256(payload), size=len(payload))
    elif sha256(payload) != source.get("sha256"):
        raise RuntimeError(f"{source['id']}: SHA-256 mismatch ({sha256(payload)}); the pinned file changed")
    RAW.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(payload)
    return payload


def derive_travel_flights(payloads):
    """Flights between the origin and the destination-state cities of each validation query, on the query's dates."""
    import csv
    import io
    from autoresearch_suite.llm_bench_travel import parse_validation_csv
    states = {}
    for line in payloads["travelplanner-city-states"].decode("utf-8").split("\n"):
        if "\t" in line:
            city, state = line.split("\t")
            states.setdefault(state, set()).add(city)
    keep = set()
    for row in parse_validation_csv(payloads["travelplanner-validation"].decode("utf-8")):
        cities = {row["org"]} | (states.get(row["dest"], set()) if row["visiting_city_number"] > 1 else {row["dest"]})
        for date in row["date"]:
            keep |= {(a, b, date) for a in cities for b in cities if a != b}
    reader = csv.reader(io.StringIO(payloads["travelplanner-flights"].decode("utf-8")))
    header = next(reader)
    columns = {name: i for i, name in enumerate(header)}
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    for record in reader:
        if (record[columns["OriginCityName"]], record[columns["DestCityName"]], record[columns["FlightDate"]]) in keep:
            writer.writerow(record)
    return out.getvalue().encode("utf-8")


ALFWORLD_TASK_TYPES = ("pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
                       "pick_cool_then_place_in_recep", "pick_two_obj_and_place")


def derive_alfworld(payloads):
    """The eval_out_of_distribution games exactly as AlfredTWEnv.collect_game_files selects them, plus a sorted manifest."""
    import io
    import zipfile
    trajectories = zipfile.ZipFile(io.BytesIO(payloads["alfworld-json"]))
    games = zipfile.ZipFile(io.BytesIO(payloads["alfworld-tw-pddl"]))
    game_names = set(games.namelist())
    out, manifest = {}, []
    prefix = "json_2.1.1/valid_unseen/"
    for name in sorted(trajectories.namelist()):
        if not (name.startswith(prefix) and name.endswith("/traj_data.json")):
            continue
        root = name[:-len("traj_data.json")]
        if "movable" in root or "Sliced" in root:
            continue
        task_type = json.loads(trajectories.read(name))["task_type"]
        game = root + "game.tw-pddl"
        if task_type not in ALFWORLD_TASK_TYPES or game not in game_names:
            continue
        payload = games.read(game)
        if not json.loads(payload).get("solvable"):
            continue
        relative = root[len(prefix):] + "game.tw-pddl"
        out["alfworld/valid_unseen/" + relative] = payload
        manifest.append({"game": relative, "task_type": task_type})
    out["alfworld/games.json"] = (json.dumps(manifest, indent=1) + "\n").encode()
    return out


def extract(payload, spec):
    """Members of a zip archive under the given prefixes, keyed by their path below ``spec['to']``."""
    import io
    import zipfile
    out = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/") or "/__pycache__/" in name:
                continue
            for prefix, target in spec.items():
                if name.startswith(prefix):
                    out[target + name[len(prefix):]] = archive.read(name)
    return out


def tree_digest(files):
    return sha256("\n".join(f"{path} {sha256(data)}" for path, data in sorted(files.items())).encode())


def build(component, lock, update):
    sources = [s for s in lock["sources"] if s["component"] == component]
    payloads = {s["id"]: fetch(s, update) for s in sources}
    produced, trees = {}, {}
    for source in sources:
        if source.get("target"):
            produced[source["target"]] = payloads[source["id"]]
        for root, spec in source.get("extract", {}).items():
            trees[root] = extract(payloads[source["id"]], spec)
    if component == "travel":
        produced["travel/flights.csv"] = derive_travel_flights(payloads)
    if component == "alfworld":
        trees["alfworld/valid_unseen/"] = {k: v for k, v in derive_alfworld(payloads).items() if k.startswith("alfworld/valid_unseen/")}
        produced["alfworld/games.json"] = derive_alfworld(payloads)["alfworld/games.json"]
    for root, files in trees.items():
        if (DATA / root).exists():
            shutil.rmtree(DATA / root)
        produced.update(files)
    for path, payload in produced.items():
        destination = DATA / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    singles = {path: payload for path, payload in produced.items() if not any(path.startswith(root) for root in trees)}
    entries = {path: {"sha256": sha256(payload), "size": len(payload)} for path, payload in singles.items()}
    for root, files in trees.items():
        entries[root] = {"sha256": tree_digest({p[len(root):]: d for p, d in files.items()}), "size": sum(len(d) for d in files.values()),
                         "files": len(files)}
    records = {f["path"]: f for f in lock["files"] if f["component"] == component}
    for path, entry in sorted(entries.items()):
        if update:
            records[path] = {"path": path, "component": component, **entry}
        elif path not in records or records[path]["sha256"] != entry["sha256"]:
            raise RuntimeError(f"{path}: derived data do not match the lock ({entry['sha256']})")
    if update:
        lock["files"] = [f for f in lock["files"] if f["component"] != component] + [records[p] for p in sorted(records)]
    print(f"{component}: {len(produced)} files in {DATA}")


def setup_environment():
    """Create the pinned runtime used by the BALROG, PDDL and ScienceWorld environment servers."""
    python = shutil.which(ENV_PYTHON)
    if not python:
        raise SystemExit(f"{ENV_PYTHON} not found on PATH; install Python 3.11 (for example `brew install python@3.11` or `uv python install 3.11`)")
    if not (ENV / "bin" / "python").exists():
        subprocess.run([python, "-m", "venv", str(ENV)], check=True)
    pip = [str(ENV / "bin" / "python"), "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    for group in ENV_REQUIREMENTS:
        subprocess.run(pip + group["args"] + group["packages"], check=True)
    print(f"environment runtime ready: {ENV / 'bin' / 'python'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("components", nargs="*", help="m3tooleval, agentboard, travel, minihack, alfworld (default: all)")
    parser.add_argument("--update-lock", action="store_true")
    parser.add_argument("--setup-environment", action="store_true")
    args = parser.parse_args(argv)
    if args.setup_environment:
        setup_environment()
        if not args.components:
            return 0
    lock = json.loads(LOCK.read_text())
    components = args.components or sorted({s["component"] for s in lock["sources"]})
    for component in components:
        build(component, lock, args.update_lock)
    if args.update_lock:
        LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
