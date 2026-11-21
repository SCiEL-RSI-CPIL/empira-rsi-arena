"""Constrained covering arrays for combinatorial interaction testing (CIT).

A model has k parameters with ``values[p]`` levels and a list of forbidden
value combinations (each a list of ``[parameter, level]`` pairs that may not
all appear in one test configuration). A t-way constrained covering array is a
list of configurations (rows) that avoid every forbidden combination and
together contain every *feasible* t-way combination of levels, where a
combination is feasible when some valid configuration contains it. The task
objective is the number of rows (fewer is better).

Instances come from the CIT benchmark archive of Jia, Cohen, Harman and Petke
(ICSE 2015; Zenodo record 581657, CC BY 4.0) in CASA format. Their constraint
files number levels from 1; ``parse_casa`` converts them to 0-based indices.
Feasibility of t-way combinations is decided once, at preparation time, by the
exact backtracking search below and stored as minimal infeasible combinations;
the tests cross-check every stored decision with Z3.
"""
from __future__ import annotations

import io
import itertools
import json
import zipfile

import numpy as np

from . import software_data, software_sandbox
from .runtime import CandidateError

MAX_ROWS = 5000
ARCHIVE = "cit-icse15-benchmarks"
ROOT = "ICSE15_HHSA_Benchmarks/"
# name -> (model member, constraint member). SPIN-S at strength 3 uses the archive's original 13-clause
# constraint list; the archive's expanded 3-way file encodes the same valid configurations (checked with Z3).
INSTANCES = {
    "apache-2way": ("calvagna_stvr_2011/apache_2way.model", "calvagna_stvr_2011/apache_2way_yue.cons"),
    "gcc-2way": ("calvagna_stvr_2011/gcc_2way.model", "calvagna_stvr_2011/gcc_2way_yue.cons"),
    "spinv-2way": ("calvagna_stvr_2011/spinv_2way.model", "calvagna_stvr_2011/spinv_2way_yue.cons"),
    "spins-3way": ("calvagna_stvr_2011/spins_3way.model", "calvagna_stvr_2011/spins_2way_yue.cons"),
    "healthcare4-2way": ("segall_issta_11/Healthcare4.model", "segall_issta_11/Healthcare4_yue.cons"),
}
SOURCES = (ARCHIVE,)


def derive(raw):
    """Prepared instance files from the pinned archive bytes: {relative path: bytes}."""
    files = {}
    with zipfile.ZipFile(io.BytesIO(raw[ARCHIVE])) as archive:
        for name, (model, constraints) in INSTANCES.items():
            instance = parse_casa(archive.read(ROOT + model).decode(), archive.read(ROOT + constraints).decode())
            instance["minimal_infeasible"] = Feasibility(instance["values"], instance["forbidden"]).minimal_infeasible(instance["strength"])
            instance.update(name=name, source={"archive": ARCHIVE, "model": ROOT + model, "constraints": ROOT + constraints,
                                               "level_numbering": "1-based in the archive, 0-based here"})
            files[f"cit/{name}.json"] = json.dumps(instance, sort_keys=True, separators=(",", ":")).encode()
    return files


def parse_casa(model_text, constraint_text, *, one_based=True):
    """Return {strength, values, forbidden} from CASA .model and constraint texts."""
    tokens = model_text.split()
    strength, k = int(tokens[0]), int(tokens[1])
    values = [int(x) for x in tokens[2:2 + k]]
    if len(values) != k or len(tokens) != 2 + k or min(values) < 1:
        raise ValueError("malformed CASA model")
    offsets = np.concatenate([[0], np.cumsum(values)]).tolist()
    owner = [(p, j) for p, v in enumerate(values) for j in range(v)]
    tokens = constraint_text.split() if constraint_text.strip() else ["0"]
    count, pos, forbidden = int(tokens[0]), 1, []
    for _ in range(count):
        size = int(tokens[pos]); pos += 1
        combination = []
        for _ in range(size):
            sign, index = tokens[pos], int(tokens[pos + 1]) - (1 if one_based else 0); pos += 2
            if sign != "-" or not 0 <= index < offsets[-1]:
                raise ValueError(f"unsupported literal {sign} {index}")
            combination.append(list(owner[index]))
        params = [p for p, _ in combination]
        if len(set(params)) == len(params):  # a clause naming two levels of one parameter is always satisfied
            forbidden.append(sorted(combination))
    if pos != len(tokens):
        raise ValueError("trailing tokens in CASA constraints")
    return {"strength": strength, "values": values, "forbidden": forbidden}


class Feasibility:
    """Exact satisfiability of partial configurations under forbidden combinations."""

    def __init__(self, values, forbidden):
        self.values = list(values)
        self.forbidden = [tuple(map(tuple, f)) for f in forbidden]
        parent = list(range(len(values)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for combination in self.forbidden:
            for (p, _), (q, _) in zip(combination, combination[1:]):
                parent[find(p)] = find(q)
        constrained = {p for f in self.forbidden for p, _ in f}
        self.component = {p: find(p) for p in constrained}
        self.by_component = {}
        for f in self.forbidden:
            self.by_component.setdefault(self.component[f[0][0]], []).append(f)
        self._cache = {}

    def satisfiable(self, assignment):
        """True when the partial assignment {param: level} extends to a valid configuration."""
        groups = {}
        for p, v in assignment.items():
            if p in self.component:
                groups.setdefault(self.component[p], {})[p] = v
        return all(self._component_sat(c, part) for c, part in groups.items())

    def _component_sat(self, component, part):
        key = (component, tuple(sorted(part.items())))
        if key not in self._cache:
            sets = self.by_component[component]
            params = sorted({p for f in sets for p, _ in f})
            domains = {p: ({part[p]} if p in part else set(range(self.values[p]))) for p in params}
            self._cache[key] = _search(domains, sets)
        return self._cache[key]

    def minimal_infeasible(self, strength):
        """All infeasible combinations of size <= strength whose proper subsets are feasible."""
        found, bad = [], set()
        constrained = sorted(self.component)
        for size in range(1, strength + 1):
            for params in itertools.combinations(constrained, size):
                if size > 1 and len({self.component[p] for p in params}) > 1:
                    continue  # independent components: infeasible only through a single-component subset
                for levels in itertools.product(*(range(self.values[p]) for p in params)):
                    combo = tuple(zip(params, levels))
                    if any(sub in bad for r in range(1, size) for sub in itertools.combinations(combo, r)):
                        continue
                    if not self.satisfiable(dict(combo)):
                        bad.add(combo)
                        found.append([list(x) for x in combo])
        return found


def _search(domains, sets):
    """Backtracking with forward checking; domains is modified only in copies."""
    domains = _propagate({p: set(d) for p, d in domains.items()}, sets)
    if domains is None:
        return False
    open_params = [p for p, d in domains.items() if len(d) > 1]
    if not open_params:
        return True
    p = min(open_params, key=lambda q: (len(domains[q]), q))
    for v in sorted(domains[p]):
        trial = dict(domains)
        trial[p] = {v}
        if _search(trial, sets):
            return True
    return False


def _propagate(domains, sets):
    changed = True
    while changed:
        changed = False
        for combination in sets:
            open_literal, alive = None, True
            for p, v in combination:
                d = domains[p]
                if v not in d:
                    alive = False
                    break
                if len(d) > 1:
                    if open_literal is not None:
                        alive = False  # two undecided literals: nothing to infer yet
                        break
                    open_literal = (p, v)
            if not alive:
                continue
            if open_literal is None:
                return None  # every literal of a forbidden combination is fixed
            p, v = open_literal
            domains[p] = domains[p] - {v}
            if not domains[p]:
                return None
            changed = True
    return domains


def required_masks(values, strength, minimal):
    """{params tuple: boolean array of required (feasible) level combinations}."""
    by_params = {}
    for combo in minimal:
        by_params.setdefault(tuple(p for p, _ in combo), []).append(tuple(v for _, v in combo))
    masks = {}
    for params in itertools.combinations(range(len(values)), strength):
        mask = np.ones([values[p] for p in params], dtype=bool)
        for size in range(1, strength + 1):
            for idx in itertools.combinations(range(strength), size):
                for levels in by_params.get(tuple(params[i] for i in idx), ()):
                    index = [slice(None)] * strength
                    for i, level in zip(idx, levels):
                        index[i] = level
                    mask[tuple(index)] = False
        masks[params] = mask
    return masks


def check_array(instance, rows, masks=None):
    """Validate a candidate array; return (rows, details) or raise CandidateError."""
    values, strength = instance["values"], instance["strength"]
    k = len(values)
    if not isinstance(rows, list) or not rows:
        raise CandidateError("generate() must return a non-empty list of rows")
    if len(rows) > MAX_ROWS:
        raise CandidateError(f"{len(rows)} rows exceed the limit of {MAX_ROWS}")
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != k:
            raise CandidateError(f"row {i} must be a list of {k} integers")
        if not all(type(x) is int for x in row):
            raise CandidateError(f"row {i} contains a non-integer level")
        if not all(0 <= x < v for x, v in zip(row, values)):
            raise CandidateError(f"row {i} has a level outside its parameter's range")
    array = np.asarray(rows, dtype=np.int64)
    for combination in instance["forbidden"]:
        hit = np.ones(len(array), dtype=bool)
        for p, v in combination:
            hit &= array[:, p] == v
        if hit.any():
            raise CandidateError(f"row {int(np.flatnonzero(hit)[0])} contains the forbidden combination {combination}")
    masks = masks if masks is not None else required_masks(values, strength, instance["minimal_infeasible"])
    missing, required = 0, 0
    example = None
    for params, mask in masks.items():
        covered = np.zeros(mask.shape, dtype=bool)
        covered[tuple(array[:, p] for p in params)] = True
        gap = mask & ~covered
        required += int(mask.sum())
        if gap.any():
            missing += int(gap.sum())
            if example is None:
                example = [[p, int(v)] for p, v in zip(params, np.argwhere(gap)[0])]
    if missing:
        raise CandidateError(f"{missing} of {required} feasible {strength}-way combinations are not covered, e.g. {example}")
    return len(rows), {"rows": len(rows), "required_combinations": required, "parameters": k, "strength": strength}


_MASKS = {}


def load_instance(name):
    instance = software_data.load_json(f"cit/{name}.json")
    if name not in _MASKS:
        _MASKS[name] = required_masks(instance["values"], instance["strength"], instance["minimal_infeasible"])
    return instance, _MASKS[name]


def public_model(instance):
    """What the candidate sees: strength, levels per parameter and the forbidden combinations."""
    return {"name": instance["name"], "strength": instance["strength"], "values": list(instance["values"]),
            "forbidden": [list(map(list, f)) for f in instance["forbidden"]]}


def cit_score(module, name, *, cpu_seconds=60, reference=None, lower_bound=None):
    """Run ``generate(model)`` in the sandbox and certify the returned covering array."""
    program = software_sandbox.require_entry(module, "generate")
    instance, masks = load_instance(name)
    rows = software_sandbox.run_candidate(program, "generate", [public_model(instance)], cpu_seconds=cpu_seconds,
                                          memory_mb=2048)
    size, details = check_array(instance, rows, masks)
    details.update(instance=name, lower_bound=lower_bound, published_reference=reference,
                   gap_to_reference=None if reference is None else size - reference)
    return float(size), details
