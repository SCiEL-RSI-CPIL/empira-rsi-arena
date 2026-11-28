"""HP lattice-protein folding on the square (2D) and simple cubic (3D) lattices.

The HP model (Lau and Dill 1989) reduces a protein to a chain of hydrophobic (H) and
polar (P) residues on a lattice; a conformation is a self-avoiding walk and its energy
is minus the number of H-H pairs that are lattice neighbours but not chain neighbours.
The candidate folds the standard benchmark sequences; the evaluator rechecks every
conformation and recomputes its energy. Reference energies are the lowest values in
the cited literature (the ten 3D 48-mers of Yue et al. 1995 have proven ground states).
"""
from __future__ import annotations

import re

import numpy as np

from .bio_physics_common import as_int_array, run_sandboxed
from .runtime import require


def expand(compact):
    """'HP2H3' -> 'HPPHHH' (the run-length notation used in the 3D benchmark tables)."""
    return "".join(symbol * (int(count) if count else 1) for symbol, count in re.findall(r"([HP])(\d*)", compact))


# (name, sequence, best-known energy). Sources: Kou, Oh and Wong (2006) Table IV-V and Zhang, Kou and
# Liu (2007) Tables II-III for 2D; Bachmann and Janke (2004) Table II (ten 48-mers of Yue et al. 1995,
# ground states proven by CHCC) and Zhang, Kou and Liu (2007) Tables II and IV / Wuest and Landau (2012)
# for the longer 3D chains.
BENCHMARK_2D = (
    ("2D20", "HPHPPHHPHPPHPHHPPHPH", -9),
    ("2D25", "PPHPPHHPPPPHHPPPPHHPPPPHH", -8),
    ("2D36", "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", -14),
    ("2D48", "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH", -23),
    ("2D50", "HHPHPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH", -21),
    ("2D60", "PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP", -36),
    ("2D64", "HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHPHHHHHHHHHHHH", -42),
    ("2D85", "HHHHPPPPHHHHHHHHHHHHPPPPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHHHHHHHHHHHHPPPHPPHHPPHHPPHPH", -53),
    ("2D100a", "PPPPPPHPHHPPPPPHHHPHHHHHPHHPPPPHHPPHHPHHHHHPHHHHHHHHHHPHHPHHHHHHHPPPPPPPPPPPHHHHHHHPPHPHHHPPPPPPHPHH", -48),
    ("2D100b", "PPPHHPPHHHHPPHHHPHHPHHPHHHHPPPPPPPPHHHHHHPPHHHHHHPPPPPPPPPHPHHPHHHHHHHHHHHPPHHHPHHPHPPHPHHHPPPPPPHHH", -50),
)

BENCHMARK_3D = tuple((f"3D48.{i + 1}", expand(compact), energy) for i, (compact, energy) in enumerate((
    ("HPH2P2H4PH3P2H2P2HPH3PHPH2P2H2P3HP8H2", -32),
    ("H4PH2PH5P2HP2H2P2HP6HP2HP3HP2H2P2H3PH", -34),
    ("PHPH2PH6P2HPHP2HPH2PHPHP3HP2H2P2H2P2HPHP2HP", -34),
    ("PHPH2P2HPH3P2H2PH2P3H5P2HPH2PHPHP4HP2HPHP", -33),
    ("P2HP3HPH4P2H4PH2PH3P2HPHPHP2HP6H2PH2PH", -32),
    ("H3P3H2PHPH2PH2PH2PHP7HPHP2HP3HP2H6PH", -32),
    ("PHP4HPH3PHPH4PH2PH2P3HPHP3H3P2H2P2H2P3H", -32),
    ("PH2PH3PH4P2H3P6HPH2P2H2PHP3H2PHPHPH2P3", -31),
    ("PHPHP4HPHPHP2HPH6P2H3PHP2HPH2P2HPH3P4H", -34),
    ("PH2P6H2P3H3PHP2HPH2P2HP2HP2H2P2H7P2H2", -33),
))) + (
    ("3D64", "PHHPHHPHHHPPHPHPPHPHPPHHHPHHPHHPPHHPHHPHHHPPHPHPPHPHPPHHHPHHPHHP", -56),
    ("3D67", "PHPHHPHHPHPPHHHPPPHPHHPHHPHPPHHHPPPHPHHPHHPHPPHHHPPPHPHHPHHPHPPHHHP", -56),
    ("3D88", "PHPHHPHHPHPPHHPPHPPHPPHPPHPPHPPHHPPHHHPPHHHPPHHHPPHHHPPHPHHPHHPHPPHPPHPPHHPPHPPHPPHHPPHP", -72),
    ("3D103", "PPHHPPPPPHHPPHHPHPPHPPPPPPPHPPPHHPHHPPPPPPHPPHPHPPHPPPPPHHHPPPPHHPHHPPPPPHHPPPPHHHHPHPPPPPPPPHHHHHPPHPP", -58),
    ("3D124", "PPPHHHPHPPPPHPPPPPHHPPPPHHPPHHPPPPHPPPPHPPHPPHHPPPHHPHPHHHPPPPHHHPPPPPPHHPPHPPHPHPPHPPPPPPPHPPHHHPPPPHPPPHHHHHPPPPHHPHPHPHPH", -75),
    ("3D136", "HPPPPPHPPPPHPHHPHHPPPPHPHHHPPPPHPHPHHHHPPPPPPPPPPPHPPHPPPHPHHPPPHHPPHPPHPHPHPPPPPPPPHPPPHHHHHHPPPHHPPHHHPPPHHPHHHHHPPPPPPPPPHPPPPHPHPPPP", -83),
)


def benchmark(dimension):
    require(dimension in (2, 3), "dimension must be 2 or 3")
    return BENCHMARK_2D if dimension == 2 else BENCHMARK_3D


# Lattice coordinates may lie anywhere: a walk is the same conformation after any translation. The bound only
# keeps every coordinate and difference an exact integer in float64 and int64.
COORDINATE_LIMIT = 2 ** 40


def energy(sequence, conformation, dimension):
    """Check a conformation (n lattice points) and return its HP energy (a non-positive integer)."""
    n = len(sequence)
    coords = as_int_array(conformation, shape=(n, dimension), name="conformation", low=-COORDINATE_LIMIT,
                          high=COORDINATE_LIMIT)
    steps = np.abs(np.diff(coords, axis=0)).sum(axis=1)
    require(bool(np.all(steps == 1)), "consecutive residues must occupy neighbouring lattice sites")
    keys = [tuple(p) for p in coords.tolist()]
    require(len(set(keys)) == n, "the chain intersects itself (two residues on one site)")
    where = {key: i for i, key in enumerate(keys)}
    contacts = 0
    units = np.eye(dimension, dtype=np.int64).tolist()
    for i, key in enumerate(keys):
        if sequence[i] != "H":
            continue
        for unit in units:  # each unordered neighbour pair counted once (positive directions only)
            j = where.get(tuple(a + b for a, b in zip(key, unit)))
            if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                contacts += 1
    return -contacts


DRIVER = '''

def _bp_hp_driver(sequences, dimension):
    _bp_seed()
    return [_bp_plain(fold(sequence, dimension)) for sequence in sequences]
'''


def evaluate_conformations(dimension, conformations):
    """Score a list of conformations (one per benchmark sequence) against the best-known energies."""
    instances = benchmark(dimension)
    require(isinstance(conformations, list) and len(conformations) == len(instances),
            f"expected {len(instances)} conformations")
    rows, ratios = [], []
    for (name, sequence, best), conformation in zip(instances, conformations):
        value = energy(sequence, conformation, dimension)
        ratios.append(value / best)
        rows.append({"name": name, "length": len(sequence), "energy": value, "best_known": best})
    total = sum(r["energy"] for r in rows)
    details = {"instances": rows, "total_energy": total, "best_known_total": sum(b for _, _, b in instances),
               "matched_best_known": sum(r["energy"] <= r["best_known"] for r in rows),
               "below_best_known": [r["name"] for r in rows if r["energy"] < r["best_known"]]}
    return float(np.mean(ratios)), details


def score(module, P):
    dimension = int(P["dimension"])
    sequences = [s for _, s, _ in benchmark(dimension)]
    conformations = run_sandboxed(module, DRIVER, "_bp_hp_driver", [sequences, dimension], required=("fold",),
                                  cpu_seconds=int(P.get("cpu_seconds", 120)), memory_mb=2048)
    value, details = evaluate_conformations(dimension, conformations)
    details["cpu_seconds_limit"] = int(P.get("cpu_seconds", 120))
    return value, details
