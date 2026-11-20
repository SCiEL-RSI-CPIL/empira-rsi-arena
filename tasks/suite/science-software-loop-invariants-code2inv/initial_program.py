"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Seed for loop-invariant inference: Houdini over simple templates (Flanagan and Leino, 2001).

Candidate atoms compare each variable with the integer constants of the C program (and 0, 1) and each pair of
variables. The conjunction of all atoms is checked; a counterexample state from initiation (or the successor
state from consecution) removes every atom it falsifies, and the loop repeats until the conjunction is
inductive or the query budget runs out. Houdini cannot add atoms, so it fails when safety needs a fact
outside the template.
"""
import itertools
import re


def _atoms(problem):
    names = problem["variables"]
    constants = sorted({0, 1} | {int(c) for c in re.findall(r"(?<![\w.])-?\d+(?![\w.])", problem["c_source"])})
    atoms = []
    for v in names:
        for c in constants:
            literal = str(c) if c >= 0 else f"(- {-c})"
            atoms += [(f"(>= {v} {literal})", lambda s, v=v, c=c: s[v] >= c), (f"(<= {v} {literal})", lambda s, v=v, c=c: s[v] <= c)]
    for a, b in itertools.permutations(names, 2):
        atoms.append((f"(<= {a} {b})", lambda s, a=a, b=b: s[a] <= s[b]))
    for a, b in itertools.combinations(names, 2):
        atoms.append((f"(= {a} {b})", lambda s, a=a, b=b: s[a] == s[b]))
    return atoms


def _conjunction(atoms):
    return "(and " + " ".join(text for text, _ in atoms) + ")" if atoms else "true"


def synthesize(problem, oracle):
    atoms = _atoms(problem)
    while True:
        reply = oracle(_conjunction(atoms))
        if reply.get("valid") or reply.get("queries_left", 0) <= 0:
            return _conjunction(atoms)
        if reply["failed"] == "initiation" and reply.get("state"):
            witness = reply["state"]
        elif reply["failed"] == "consecution" and reply.get("next"):
            witness = reply["next"]
        else:
            return _conjunction(atoms)  # safety fails (or no witness): Houdini cannot strengthen
        kept = [atom for atom in atoms if atom[1](witness)]
        if len(kept) == len(atoms):
            return _conjunction(atoms)
        atoms = kept
# EVOLVE-BLOCK-END
