"""Ground states of Edwards-Anderson spin glasses on two- and three-dimensional tori.

The instances are the toroidal grid graphs with Gaussian couplings of Liers, Juenger,
Reinelt and Rinaldi (2004), distributed in the Biq Mac library as max-cut problems with
integer weights. Their optimal cut values were computed exactly by branch-and-cut and are
listed in Table 10 of the library documentation; the corresponding spin-glass ground-state
energy of the couplings J = -w is E* = sum(w) - 2 * maxcut.

The candidate returns a spin configuration per instance; the evaluator recomputes the cut
weight from the edge list and compares it with the published optimum.
"""
from __future__ import annotations

import numpy as np

from .bio_physics_common import as_int_array, load_derived, run_sandboxed
from .runtime import require

RAW_SOURCES = ("biqmac-ising_all.tar.gz",)
DATA2 = "spinglass/torus_2d.json.gz"
DATA3 = "spinglass/torus_3d.json.gz"
DERIVED = {DATA2: "96a620bb831065ace12b0af01626e611fb31ec4be5fdd832b3bf01dc07ef475a",
           DATA3: "d2562deb6148d031c4fc45760d61e95201b54b08022573ade00cb02171e3cc51"}

# Optimal max-cut values, Table 10 of the Biq Mac library documentation (biqmaclib.pdf),
# computed by branch-and-cut (Liers et al. 2004). name -> (side length, dimension, optimum).
OPTIMA = {
    "t2g10_5555": (10, 2, 6049461), "t2g10_6666": (10, 2, 5757868), "t2g10_7777": (10, 2, 6509837),
    "t2g15_5555": (15, 2, 15051133), "t2g15_6666": (15, 2, 15763716), "t2g15_7777": (15, 2, 15269399),
    "t2g20_5555": (20, 2, 24838942), "t2g20_6666": (20, 2, 29290570), "t2g20_7777": (20, 2, 28349398),
    "t3g5_5555": (5, 3, 10933215), "t3g5_6666": (5, 3, 11582216), "t3g5_7777": (5, 3, 11552046),
    "t3g6_5555": (6, 3, 17434469), "t3g6_6666": (6, 3, 20217380), "t3g6_7777": (6, 3, 19475011),
    "t3g7_5555": (7, 3, 28302918), "t3g7_6666": (7, 3, 33611981), "t3g7_7777": (7, 3, 29118445),
}


def derive(paths):
    """Parse the Biq Mac rudy-format instances and keep the toroidal grid graphs."""
    import tarfile

    out = {2: {}, 3: {}}
    with tarfile.open(paths[RAW_SOURCES[0]]) as archive:
        for member in archive.getmembers():
            name = member.name.rsplit("/", 1)[-1]
            if name not in OPTIMA:
                continue
            side, dimension, optimum = OPTIMA[name]
            lines = archive.extractfile(member).read().decode().split()
            nodes, edges = int(lines[0]), int(lines[1])
            numbers = [int(x) for x in lines[2:]]
            require(len(numbers) == 3 * edges, f"{name}: expected {edges} edges")
            require(nodes == side ** dimension and edges == dimension * nodes,
                    f"{name}: {nodes} nodes and {edges} edges do not form a {side}^{dimension} torus")
            rows = [[numbers[3 * k] - 1, numbers[3 * k + 1] - 1, numbers[3 * k + 2]] for k in range(edges)]
            out[dimension][name] = {"nodes": nodes, "side": side, "edges": rows, "optimum": optimum}
    for dimension, payload in out.items():
        require(len(payload) == 9, f"expected nine {dimension}D instances, found {len(payload)}")
    return {DATA2: {"source": RAW_SOURCES[0], "dimension": 2, "instances": out[2]},
            DATA3: {"source": RAW_SOURCES[0], "dimension": 3, "instances": out[3]}}


def load_instances(dimension):
    name, digest = (DATA2, DERIVED[DATA2]) if dimension == 2 else (DATA3, DERIVED[DATA3])
    payload = load_derived(name, digest)
    return [(key, payload["instances"][key]) for key in sorted(payload["instances"])]


def public_instance(name, instance):
    """What the candidate sees: the weighted edge list and the lattice shape, never the optimum."""
    return {"name": name, "nodes": instance["nodes"], "side": instance["side"],
            "dimension": 2 if instance["nodes"] == instance["side"] ** 2 else 3,
            "edges": [list(e) for e in instance["edges"]]}


def cut_weight(instance, spins):
    """Recompute the weight of the cut induced by a +-1 spin vector."""
    values = as_int_array(spins, shape=(instance["nodes"],), name="spins", low=-1, high=1)
    require(bool(np.all(np.abs(values) == 1)), "spins: every entry must be -1 or +1")
    rows = np.array(instance["edges"], dtype=np.int64)
    crossing = values[rows[:, 0]] != values[rows[:, 1]]
    return int(rows[crossing, 2].sum())


DRIVER = '''

def _bp_spinglass_driver(instances):
    _bp_seed()
    return [_bp_plain(ground_state(instance)) for instance in instances]
'''


def score(module, P):
    dimension = int(P["dimension"])
    instances = load_instances(dimension)
    public = [public_instance(name, instance) for name, instance in instances]
    answers = run_sandboxed(module, DRIVER, "_bp_spinglass_driver", [public], required=("ground_state",),
                            cpu_seconds=int(P.get("cpu_seconds", 180)), memory_mb=2048)
    require(isinstance(answers, list) and len(answers) == len(instances),
            f"expected {len(instances)} spin configurations")
    rows, ratios = [], []
    for (name, instance), spins in zip(instances, answers):
        value = cut_weight(instance, spins)
        total = sum(e[2] for e in instance["edges"])
        ratios.append(value / instance["optimum"])
        rows.append({"instance": name, "nodes": instance["nodes"], "cut": value, "optimum": instance["optimum"],
                     "relative_gap": (instance["optimum"] - value) / instance["optimum"],
                     "energy": total - 2 * value, "optimal_energy": total - 2 * instance["optimum"]})
    details = {"instances": rows, "optimal_instances": sum(r["cut"] == r["optimum"] for r in rows),
               "worst_relative_gap": max(r["relative_gap"] for r in rows),
               "cpu_seconds_limit": int(P.get("cpu_seconds", 180))}
    return float(np.mean(ratios)), details
