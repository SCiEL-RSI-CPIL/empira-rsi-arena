#!/usr/bin/env python3
"""Set up the private simulation runtime of the sim_control evaluators.

The Push-T task runs gym-pusht 0.1.6 (Apache-2.0), which calls pymunk's
``Space.add_collision_handler``; pymunk 7 removed that API (replaced by
``on_collision``), so gym-pusht needs pymunk < 7. The simulator therefore runs in
its own virtual environment, <task cache>/sim_control-env, with the pinned versions
below; the suite's shared environment is not modified. No data are downloaded:
the Sodarace and Eureka-analogue evaluators only need Box2D and MuJoCo from the
suite environment.

    python3 scripts/prepare_sim_control.py --setup-environment   # create or update <task cache>/sim_control-env
    python3 scripts/prepare_sim_control.py --check               # print the installed versions
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from arena_task_paths import cache_path  # noqa: E402

ENV = cache_path("sim_control-env")
# Versions used for the recorded checks (validation/sim_control-checks.json); gym-pusht's own
# dependencies (opencv-python, pygame, scikit-image) are resolved by pip and listed by --check.
REQUIREMENTS = ["gym-pusht==0.1.6", "pymunk==6.11.1", "gymnasium==1.3.0", "numpy==2.5.3", "shapely==2.1.2"]


def python():
    return ENV / "bin" / "python"


def setup_environment(base=None):
    if not python().exists():
        subprocess.run([base or sys.executable, "-m", "venv", str(ENV)], check=True)
    subprocess.run([str(python()), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", *REQUIREMENTS], check=True)
    check()


def check():
    code = ("import json, importlib.metadata as m, sys; "
            "print(json.dumps({'python': sys.version.split()[0], **{p: m.version(p) for p in "
            "['gym-pusht', 'pymunk', 'gymnasium', 'numpy', 'shapely', 'pygame', 'opencv-python']}}))")
    result = subprocess.run([str(python()), "-I", "-c", code], check=True, capture_output=True, text=True)
    versions = json.loads(result.stdout)
    print(json.dumps(versions, indent=2))
    if not versions["pymunk"].startswith("6."):
        raise SystemExit("gym-pusht 0.1.6 needs pymunk < 7")
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup-environment", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--base-python", help="interpreter used to create the virtual environment (default: this one)")
    args = parser.parse_args(argv)
    if args.setup_environment:
        setup_environment(args.base_python)
    elif args.check:
        check()
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
