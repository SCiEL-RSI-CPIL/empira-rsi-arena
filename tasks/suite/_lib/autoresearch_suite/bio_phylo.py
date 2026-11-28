"""Phylogenetic topology reconstruction from simulated sequences with known trees.

Twenty instances are generated inside the evaluator: a random unrooted topology with
exponential branch lengths, and DNA sequences evolved along it under the Jukes-Cantor
model. Everything is driven by uniform random numbers from a seeded PCG64 bit generator,
so the instances are identical on every run. The candidate returns a Newick tree per
instance and the evaluator scores the normalized Robinson-Foulds distance (Robinson and
Foulds 1981) to the generating topology, the standard accuracy measure of simulation
studies of tree reconstruction (Huelsenbeck 1995).
"""
from __future__ import annotations

import re

import numpy as np

from .bio_physics_common import run_sandboxed
from .runtime import CandidateError, require

BASES = "ACGT"
# (taxa, sites, mean internal branch length, mean pendant branch length, seed)
INSTANCES = (
    (8, 500, 0.05, 0.15, 11), (8, 500, 0.02, 0.30, 12), (8, 1000, 0.08, 0.10, 13),
    (12, 500, 0.05, 0.15, 21), (12, 1000, 0.03, 0.20, 22), (12, 250, 0.05, 0.15, 23),
    (16, 500, 0.05, 0.15, 31), (16, 1000, 0.04, 0.12, 32), (16, 250, 0.06, 0.25, 33),
    (24, 1000, 0.05, 0.15, 41), (24, 500, 0.04, 0.18, 42), (24, 2000, 0.02, 0.10, 43),
    (8, 250, 0.03, 0.25, 51), (12, 2000, 0.02, 0.08, 52), (16, 2000, 0.03, 0.09, 53),
    (24, 250, 0.08, 0.20, 54), (10, 750, 0.05, 0.15, 61), (14, 750, 0.04, 0.16, 62),
    (20, 750, 0.05, 0.14, 63), (32, 1000, 0.04, 0.12, 64),
)


def _generator(seed):
    return np.random.Generator(np.random.PCG64(seed))


def random_tree(taxa, internal_mean, pendant_mean, rng):
    """Random unrooted topology: repeatedly join two random subtrees; returns (children, lengths, splits)."""
    children, lengths = {}, {}
    active = list(range(taxa))
    for leaf in active:
        lengths[leaf] = -pendant_mean * np.log(1.0 - rng.random())
    next_node = taxa
    while len(active) > 3:
        i = int(rng.random() * len(active))
        a = active.pop(i)
        j = int(rng.random() * len(active))
        b = active.pop(j)
        children[next_node] = (a, b)
        lengths[next_node] = -internal_mean * np.log(1.0 - rng.random())
        active.append(next_node)
        next_node += 1
    root = next_node
    children[root] = tuple(active)
    lengths[root] = 0.0
    return children, lengths, root


def leaf_sets(children, root, taxa):
    """Leaf set below every node (post-order)."""
    sets = {}

    def walk(node):
        if node < taxa:
            sets[node] = frozenset([node])
        else:
            below = set()
            for child in children[node]:
                walk(child)
                below |= sets[child]
            sets[node] = frozenset(below)

    walk(root)
    return sets


def splits(children, root, taxa):
    """Non-trivial unrooted splits, each canonicalized as the side without taxon 0."""
    sets = leaf_sets(children, root, taxa)
    universe = frozenset(range(taxa))
    out = set()
    for node, below in sets.items():
        if node == root or len(below) < 2 or len(below) > taxa - 2:
            continue
        side = below if 0 not in below else universe - below
        if 2 <= len(side) <= taxa - 2:
            out.add(side)
    return out


def simulate(taxa, sites, internal_mean, pendant_mean, seed):
    """Jukes-Cantor sequences along a random tree; returns (sequences, true splits)."""
    rng = _generator(seed)
    children, lengths, root = random_tree(taxa, internal_mean, pendant_mean, rng)
    states = {root: (rng.random(sites) * 4).astype(np.int64)}
    order = [root]
    while order:
        node = order.pop()
        for child in children.get(node, ()):
            change_probability = 0.75 * (1.0 - np.exp(-4.0 / 3.0 * lengths[child]))
            draws = rng.random(sites)
            shift = 1 + (rng.random(sites) * 3).astype(np.int64)
            inherited = states[node].copy()
            mutate = draws < change_probability
            inherited[mutate] = (inherited[mutate] + shift[mutate]) % 4
            states[child] = inherited
            if child in children:
                order.append(child)
    sequences = {f"t{leaf}": "".join(BASES[i] for i in states[leaf]) for leaf in range(taxa)}
    return sequences, splits(children, root, taxa)


def parse_newick(text, taxa):
    """Parse a Newick tree over the labels t0..t(taxa-1) into its non-trivial unrooted splits."""
    require(isinstance(text, str) and text.strip(), "tree: expected a non-empty Newick string")
    tokens = re.findall(r"[(),;]|[^(),;]+", text.strip())
    stack, current, seen, groups = [], [], set(), set()
    for token in tokens:
        if token == "(":
            stack.append(current)
            current = []
        elif token == ")":
            require(stack, "tree: unbalanced parentheses")
            require(len(current) >= 2, "tree: internal node with fewer than two children")
            group = frozenset().union(*current)
            groups.add(group)
            current = stack.pop()
            current.append(group)
        elif token in (",", ";"):
            continue
        else:
            label = token.split(":")[0].strip()
            if not label:
                continue  # a branch length attached to the group that was just closed
            require(re.fullmatch(r"t\d+", label) is not None, f"tree: unknown taxon label {label!r}")
            index = int(label[1:])
            require(0 <= index < taxa, f"tree: taxon index out of range in {label!r}")
            require(index not in seen, f"tree: taxon {label} appears more than once")
            seen.add(index)
            current.append(frozenset([index]))
    require(not stack, "tree: unbalanced parentheses")
    require(seen == set(range(taxa)), f"tree: expected all {taxa} taxa exactly once, got {len(seen)}")
    universe = frozenset(range(taxa))
    out = set()
    for group in groups:
        if not 2 <= len(group) <= taxa - 2:
            continue
        side = group if 0 not in group else universe - group
        if 2 <= len(side) <= taxa - 2:
            out.add(side)
    return out


def robinson_foulds(true_splits, predicted_splits):
    """Normalized RF distance in [0, 1]: symmetric difference over the total number of splits."""
    total = len(true_splits) + len(predicted_splits)
    if total == 0:
        return 0.0
    return len(true_splits ^ predicted_splits) / total


DRIVER = '''

def _bp_phylo_driver(problems):
    _bp_seed()
    return [str(reconstruct(problem["sequences"])) for problem in problems]
'''


def score(module, P):
    instances = [simulate(*spec) for spec in INSTANCES]
    problems = [{"name": f"sim{index}", "sequences": sequences} for index, (sequences, _) in enumerate(instances)]
    answers = run_sandboxed(module, DRIVER, "_bp_phylo_driver", [problems], required=("reconstruct",),
                            cpu_seconds=int(P.get("cpu_seconds", 180)), memory_mb=2048)
    require(isinstance(answers, list) and len(answers) == len(instances), f"expected {len(instances)} trees")
    rows = []
    for spec, (sequences, true_splits), answer in zip(INSTANCES, instances, answers):
        taxa = spec[0]
        predicted = parse_newick(answer, taxa)
        distance = robinson_foulds(true_splits, predicted)
        rows.append({"taxa": taxa, "sites": spec[1], "true_splits": len(true_splits),
                     "predicted_splits": len(predicted), "normalized_rf": distance,
                     "exact": distance == 0.0})
    values = [r["normalized_rf"] for r in rows]
    return float(np.mean(values)), {"instances": rows, "mean_normalized_rf": float(np.mean(values)),
                                    "exact_recoveries": sum(r["exact"] for r in rows),
                                    "worst_normalized_rf": float(max(values))}


def check_tree(text, taxa):
    """Public helper for the tests: parse and return the splits, raising CandidateError on bad input."""
    try:
        return parse_newick(text, taxa)
    except CandidateError:
        raise
