"""RNA secondary-structure prediction against Rfam consensus structures.

Reference structures are the covariance-model consensus structures of the Rfam families
that have experimentally determined three-dimensional structures (Rfam 3D seed alignments,
CC0). For each selected sequence the consensus pairing is projected onto the sequence,
keeping the canonical pairs (Watson-Crick and GU) whose two columns are both present. The
candidate predicts a dot-bracket structure per sequence; the evaluator compares base-pair
sets and scores the mean F1, the standard measure in the structure-prediction literature
(Mathews et al. 2004; Lorenz et al. 2011).
"""
from __future__ import annotations

import gzip
import re

import numpy as np

from .bio_physics_common import load_derived, run_sandboxed
from .runtime import require

RAW_SOURCES = ("rfam-Rfam.3d.seed.gz",)
DATA = "rna/rfam_consensus_structures.json.gz"
DERIVED = {DATA: "90a79f3f0f71e07e5a2947ec8e5cf4a6f414f72b37810eda8419a90d8c595e7b"}

# Structured non-coding RNA families only (riboswitches, ribozymes, sRNAs, snRNAs, miRNAs and
# similar); families describing viral elements or CRISPR loci are excluded by name.
EXCLUDED = ("hiv", "hbv", "corona", "flavi", "denv", "tymv", "virus", "viral", "retro", "ires",
            "frameshift", "crispr", "tracr", "xrrna", "s2m", "phage", "gp_knot", "k10_tls")
MIN_LENGTH, MAX_LENGTH, MIN_PAIRS, PER_FAMILY = 40, 150, 8, 2
CANONICAL = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}


def consensus_pairs(structure):
    """WUSS consensus string -> nested pairs from the four bracket families."""
    openers = {"<": [], "(": [], "[": [], "{": []}
    closers = {">": "<", ")": "(", "]": "[", "}": "{"}
    pairs = []
    for index, symbol in enumerate(structure):
        if symbol in openers:
            openers[symbol].append(index)
        elif symbol in closers and openers[closers[symbol]]:
            pairs.append((openers[closers[symbol]].pop(), index))
    return sorted(pairs)


def dot_bracket_pairs(structure):
    """Dot-bracket string -> list of (i, j); raises for unbalanced or unknown characters."""
    stack, pairs = [], []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            require(stack, "structure: unbalanced closing bracket")
            pairs.append((stack.pop(), index))
        else:
            require(symbol == ".", f"structure: unexpected character {symbol!r} (use '.', '(' and ')')")
    require(not stack, "structure: unbalanced opening bracket")
    return sorted(pairs)


def project(sequence_row, pairs):
    """Map consensus pairs onto one aligned row; returns (ungapped sequence, pair list)."""
    position, mapping, letters = -1, {}, []
    for column, symbol in enumerate(sequence_row):
        if symbol in "-.~_":
            continue
        position += 1
        mapping[column] = position
        letters.append("U" if symbol in "Tt" else symbol.upper())
    sequence = "".join(letters)
    kept = []
    for i, j in pairs:
        if i in mapping and j in mapping:
            a, b = sequence[mapping[i]], sequence[mapping[j]]
            if (a, b) in CANONICAL and mapping[j] - mapping[i] >= 4:
                kept.append((mapping[i], mapping[j]))
    return sequence, sorted(kept)


def derive(paths):
    with gzip.open(paths[RAW_SOURCES[0]], "rt", encoding="latin-1") as handle:
        text = handle.read()
    instances, seen = [], set()
    for block in text.split("//\n"):
        if not block.strip():
            continue
        identifier = re.search(r"#=GF ID\s+(\S+)", block)
        description = re.search(r"#=GF DE\s+(.+)", block)
        if identifier is None:
            continue
        name = identifier.group(1)
        label = (name + " " + (description.group(1) if description else "")).lower()
        if any(word in label for word in EXCLUDED):
            continue
        structure = "".join(re.search(r"#=GC SS_cons\s+(\S+)", line).group(1)
                            for line in block.splitlines() if line.startswith("#=GC SS_cons"))
        rows = re.findall(r"^(?!#|//)(\S+)\s+(\S+)$", block, re.M)
        pairs = consensus_pairs(structure)
        taken = 0
        for accession, row in rows:
            if taken >= PER_FAMILY or len(row) != len(structure):
                continue
            sequence, projected = project(row, pairs)
            if not (MIN_LENGTH <= len(sequence) <= MAX_LENGTH) or len(projected) < MIN_PAIRS:
                continue
            if sequence in seen or set(sequence) - set("ACGU"):
                continue
            seen.add(sequence)
            taken += 1
            instances.append({"family": name, "accession": accession, "sequence": sequence,
                              "pairs": [list(p) for p in projected],
                              "description": (description.group(1).strip() if description else "")})
    require(len(instances) >= 30, f"only {len(instances)} usable Rfam instances found")
    instances.sort(key=lambda d: (d["family"], d["accession"]))
    return {DATA: {"source": RAW_SOURCES[0], "selection": {"min_length": MIN_LENGTH, "max_length": MAX_LENGTH,
                                                           "min_pairs": MIN_PAIRS, "per_family": PER_FAMILY,
                                                           "excluded_keywords": list(EXCLUDED)},
                   "instances": instances}}


def load_instances():
    return load_derived(DATA, DERIVED[DATA])["instances"]


def f1(reference, predicted):
    """(F1, sensitivity, precision) of a predicted base-pair set against the reference."""
    reference, predicted = {tuple(p) for p in reference}, {tuple(p) for p in predicted}
    if not reference:
        return 0.0, 0.0, 0.0
    hits = len(reference & predicted)
    sensitivity = hits / len(reference)
    precision = hits / len(predicted) if predicted else 0.0
    score = 0.0 if hits == 0 else 2 * sensitivity * precision / (sensitivity + precision)
    return score, sensitivity, precision


DRIVER = '''

def _bp_rna_driver(sequences):
    _bp_seed()
    return [str(predict_structure(sequence)) for sequence in sequences]
'''


def score(module, P):
    instances = load_instances()
    sequences = [item["sequence"] for item in instances]
    answers = run_sandboxed(module, DRIVER, "_bp_rna_driver", [sequences], required=("predict_structure",),
                            cpu_seconds=int(P.get("cpu_seconds", 180)), memory_mb=2048)
    require(isinstance(answers, list) and len(answers) == len(instances), f"expected {len(instances)} structures")
    rows = []
    for item, structure in zip(instances, answers):
        require(isinstance(structure, str) and len(structure) == len(item["sequence"]),
                f"structure for {item['family']}/{item['accession']}: expected a string of length {len(item['sequence'])}")
        value, sensitivity, precision = f1(item["pairs"], dot_bracket_pairs(structure))
        rows.append({"family": item["family"], "accession": item["accession"], "length": len(item["sequence"]),
                     "reference_pairs": len(item["pairs"]), "f1": value, "sensitivity": sensitivity,
                     "precision": precision})
    values = [r["f1"] for r in rows]
    return float(np.mean(values)), {
        "instances": rows, "mean_f1": float(np.mean(values)),
        "mean_sensitivity": float(np.mean([r["sensitivity"] for r in rows])),
        "mean_precision": float(np.mean([r["precision"] for r in rows])),
        "families": len({r["family"] for r in rows}), "worst_family": min(rows, key=lambda r: r["f1"])["family"]}
