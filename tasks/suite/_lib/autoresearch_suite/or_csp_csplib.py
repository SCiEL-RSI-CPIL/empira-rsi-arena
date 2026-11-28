"""Instance generators and independent answer checkers for eight CSPLib problems.

The PDR paper (arXiv:2607.28268) draws instances from AutoIG pools that are not
public; these generators are local. Every generator is deterministic for its
arguments (SplitMix64 stream, see ``or_csp_cpsat.Rng``). Generators that plant
a hidden solution never expose it. Checkers implement the CSPLib definitions
(github.com/csplib/csplib Essence models, 0-based indices) and raise
``CandidateError`` with the first violated constraint. The checkers run in the
evaluator's process; the models (candidate and baseline) are built only in
sandboxed child processes (``software_sandbox``, see ``or_csp_modelrun``).
"""
from __future__ import annotations

import itertools

from .or_csp_cpsat import Rng
from .runtime import CandidateError

CAR_OPTIONS = [(1, 2), (2, 3), (1, 3), (2, 5), (1, 5)]  # (maxcars, blksize) of the classic CSPLib instances


def _int_list(value, length, name, low=None, high=None):
    if not isinstance(value, list) or len(value) != length:
        raise CandidateError(f"{name} must be a list of length {length}")
    for x in value:
        if isinstance(x, bool) or not isinstance(x, int):
            raise CandidateError(f"{name} must contain integers")
        if (low is not None and x < low) or (high is not None and x > high):
            raise CandidateError(f"{name} value {x} outside [{low}, {high}]")
    return value


def _matrix(value, rows, cols, name, low, high):
    if not isinstance(value, list) or len(value) != rows:
        raise CandidateError(f"{name} must have {rows} rows")
    return [_int_list(row, cols, f"{name} row", low, high) for row in value]


def _answer(answer, keys):
    if not isinstance(answer, dict) or set(answer) != set(keys):
        raise CandidateError(f"answer must be a dict with exactly the keys {sorted(keys)}")
    return [answer[k] for k in keys]


# --- car sequencing (CSPLib 1) ----------------------------------------------

def generate_car_sequencing(seed, n_cars, n_pool, bias):
    """Plant a random sequence that respects every option window, then derive classes and demands."""
    rng = Rng(seed)
    options = [CAR_OPTIONS[o] for o in range(len(CAR_OPTIONS))]
    subsets = [tuple(rng.randint(0, 1) for _ in options) for _ in range(n_pool)] + [tuple(0 for _ in options)]
    subsets = sorted(set(subsets))
    sequence = []
    for _ in range(n_cars):
        feasible = []
        for s in subsets:
            ok = all(not s[o] or sum(prev[o] for prev in sequence[max(0, len(sequence) - block + 1):]) + 1 <= limit
                     for o, (limit, block) in enumerate(options))
            if ok:
                feasible.append(s)
        weights = [1.0 + bias * sum(s) for s in feasible]
        pick, acc = rng.random() * sum(weights), 0.0
        for s, w in zip(feasible, weights):
            acc += w
            if pick <= acc:
                sequence.append(s)
                break
        else:
            sequence.append(feasible[-1])
    used = sorted(set(sequence))
    rng.shuffle(used)
    return {"n_cars": n_cars, "n_classes": len(used), "n_options": len(options),
            "quantity": [sequence.count(s) for s in used], "maxcars": [o[0] for o in options],
            "blksize": [o[1] for o in options], "usage": [list(s) for s in used]}


def check_car_sequencing(data, answer):
    (sequence,) = _answer(answer, ["sequence"])
    _int_list(sequence, data["n_cars"], "sequence", 0, data["n_classes"] - 1)
    for c in range(data["n_classes"]):
        if sequence.count(c) != data["quantity"][c]:
            raise CandidateError(f"class {c} appears {sequence.count(c)} times, demand {data['quantity'][c]}")
    for o in range(data["n_options"]):
        block, limit = data["blksize"][o], data["maxcars"][o]
        for start in range(data["n_cars"] - block + 1):
            if sum(data["usage"][sequence[i]][o] for i in range(start, start + block)) > limit:
                raise CandidateError(f"option {o} exceeds {limit}/{block} in window starting at {start}")


# --- vessel loading (CSPLib 8) ------------------------------------------------

def generate_vessel_loading(seed, deck_width, deck_length, n_containers, n_classes, fill, sep_max):
    """Guillotine-split the deck, shrink pieces into containers, derive separations the plant satisfies."""
    rng = Rng(seed)
    pieces = [(0, 0, deck_width, deck_length)]
    while len(pieces) < n_containers:
        pieces.sort(key=lambda p: p[2] * p[3])
        x, y, w, h = pieces.pop()
        if w >= h and w >= 2:
            cut = rng.randint(1, w - 1)
            pieces += [(x, y, cut, h), (x + cut, y, w - cut, h)]
        elif h >= 2:
            cut = rng.randint(1, h - 1)
            pieces += [(x, y, w, cut), (x, y + cut, w, h - cut)]
        else:
            pieces.append((x, y, w, h))
            break
    boxes = []
    for x, y, w, h in pieces:
        cw = max(1, min(w, round(w * (fill + (1 - fill) * rng.random()))))
        ch = max(1, min(h, round(h * (fill + (1 - fill) * rng.random()))))
        ox, oy = rng.randint(0, w - cw), rng.randint(0, h - ch)
        boxes.append((x + ox, y + oy, cw, ch))
    rng.shuffle(boxes)
    klass = [rng.randint(0, n_classes - 1) for _ in boxes]
    achieved = [[None] * n_classes for _ in range(n_classes)]
    for i, j in itertools.combinations(range(len(boxes)), 2):
        (x1, y1, w1, h1), (x2, y2, w2, h2) = boxes[i], boxes[j]
        gap = max(max(x1, x2) - min(x1 + w1, x2 + w2), max(y1, y2) - min(y1 + h1, y2 + h2))
        a, b = klass[i], klass[j]
        for u, v in ((a, b), (b, a)):
            achieved[u][v] = gap if achieved[u][v] is None else min(achieved[u][v], gap)
    separation = [[0] * n_classes for _ in range(n_classes)]
    for a in range(n_classes):
        for b in range(a, n_classes):
            bound = min(sep_max, achieved[a][b] if achieved[a][b] is not None else sep_max)
            separation[a][b] = separation[b][a] = rng.randint(0, max(0, bound))
    widths, lengths = [], []
    for _, _, w, h in boxes:  # the listed orientation is random; rotation is part of the problem
        if rng.randint(0, 1):
            w, h = h, w
        widths.append(w)
        lengths.append(h)
    return {"deck_width": deck_width, "deck_length": deck_length, "n_containers": len(boxes), "n_classes": n_classes,
            "width": widths, "length": lengths, "container_class": klass, "separation": separation}


def check_vessel_loading(data, answer):
    west, east, north, south = _answer(answer, ["west", "east", "north", "south"])
    n = data["n_containers"]
    _int_list(west, n, "west", 0, data["deck_width"])
    _int_list(east, n, "east", 0, data["deck_width"])
    _int_list(north, n, "north", 0, data["deck_length"])
    _int_list(south, n, "south", 0, data["deck_length"])
    for c in range(n):
        dims = (east[c] - west[c], south[c] - north[c])
        if dims not in ((data["width"][c], data["length"][c]), (data["length"][c], data["width"][c])):
            raise CandidateError(f"container {c} has extent {dims}, expected {data['width'][c]}x{data['length'][c]} in some orientation")
    for c1, c2 in itertools.combinations(range(n), 2):
        if west[c1] == west[c2] and north[c1] == north[c2]:
            raise CandidateError(f"containers {c1} and {c2} share a corner position")
        gap = max(max(west[c1], west[c2]) - min(east[c1], east[c2]), max(north[c1], north[c2]) - min(south[c1], south[c2]))
        need = max(0, data["separation"][data["container_class"][c1]][data["container_class"][c2]])
        if gap < need:
            raise CandidateError(f"containers {c1} and {c2} are {gap} apart, need {need} (0 means no overlap)")


# --- social golfers (CSPLib 10) -------------------------------------------------

def generate_social_golfers(weeks, groups, group_size):
    return {"weeks": weeks, "groups": groups, "group_size": group_size}


def check_social_golfers(data, answer):
    (schedule,) = _answer(answer, ["schedule"])
    weeks, groups, size = data["weeks"], data["groups"], data["group_size"]
    golfers = groups * size
    _matrix(schedule, weeks, golfers, "schedule", 0, groups - 1)
    for w, row in enumerate(schedule):
        for g in range(groups):
            if row.count(g) != size:
                raise CandidateError(f"week {w} group {g} has {row.count(g)} golfers, expected {size}")
    for p, q in itertools.combinations(range(golfers), 2):
        meetings = sum(row[p] == row[q] for row in schedule)
        if meetings > 1:
            raise CandidateError(f"golfers {p} and {q} play together {meetings} times")


# --- balanced incomplete block design (CSPLib 28) ------------------------------------

def generate_bibd(v, b, r, k, lam):
    if r * v != b * k or lam * (v - 1) != r * (k - 1):
        raise ValueError("inadmissible BIBD parameters")
    return {"v": v, "b": b, "r": r, "k": k, "lam": lam}


def check_bibd(data, answer):
    (matrix,) = _answer(answer, ["matrix"])
    v, b = data["v"], data["b"]
    _matrix(matrix, v, b, "matrix", 0, 1)
    for o, row in enumerate(matrix):
        if sum(row) != data["r"]:
            raise CandidateError(f"object {o} is in {sum(row)} blocks, expected {data['r']}")
    for j in range(b):
        if sum(matrix[o][j] for o in range(v)) != data["k"]:
            raise CandidateError(f"block {j} has the wrong size")
    for o1, o2 in itertools.combinations(range(v), 2):
        if sum(x & y for x, y in zip(matrix[o1], matrix[o2])) != data["lam"]:
            raise CandidateError(f"objects {o1} and {o2} share the wrong number of blocks")


# --- balanced academic curriculum (CSPLib 30, decision version) ------------------------

def generate_bacp(seed, n_courses, n_periods, density, load_slack, count_slack):
    """Plant a balanced curriculum, then add prerequisites and per-period bounds that it satisfies."""
    rng = Rng(seed)
    load = [rng.randint(1, 5) for _ in range(n_courses)]
    order = list(range(n_courses))
    rng.shuffle(order)
    order.sort(key=lambda c: -load[c])
    period, loads, counts = [0] * n_courses, [0] * n_periods, [0] * n_periods
    room = -(-n_courses // n_periods)
    for c in order:  # longest-load-first balancing keeps the planted period loads within one course load
        p = min((p for p in range(n_periods) if counts[p] < room), key=lambda p: (loads[p], counts[p], p))
        period[c], loads[p], counts[p] = p, loads[p] + load[c], counts[p] + 1
    prerequisites = [[a, b] for a in range(n_courses) for b in range(n_courses)
                     if period[a] < period[b] and rng.random() < density]
    loads = [sum(load[c] for c in range(n_courses) if period[c] == p) for p in range(n_periods)]
    counts = [period.count(p) for p in range(n_periods)]
    return {"n_courses": n_courses, "n_periods": n_periods, "course_load": load, "prerequisites": prerequisites,
            "load_per_period_lb": max(1, min(loads) - load_slack), "load_per_period_ub": max(loads) + load_slack,
            "courses_per_period_lb": max(1, min(counts) - count_slack), "courses_per_period_ub": max(counts) + count_slack}


def check_bacp(data, answer):
    (period,) = _answer(answer, ["period"])
    n, periods = data["n_courses"], data["n_periods"]
    _int_list(period, n, "period", 0, periods - 1)
    for before, after in data["prerequisites"]:
        if not period[before] < period[after]:
            raise CandidateError(f"prerequisite {before} -> {after} violated")
    for p in range(periods):
        load = sum(data["course_load"][c] for c in range(n) if period[c] == p)
        count = period.count(p)
        if not data["load_per_period_lb"] <= load <= data["load_per_period_ub"]:
            raise CandidateError(f"period {p} load {load} outside bounds")
        if not data["courses_per_period_lb"] <= count <= data["courses_per_period_ub"]:
            raise CandidateError(f"period {p} has {count} courses, outside bounds")


# --- fixed-length error correcting codes (CSPLib 36) --------------------------------

def generate_error_correcting_codes(alphabet_size, length, num_codewords, min_dist, distance):
    q = alphabet_size
    if distance == "lee":
        dist = [[min(abs(a - b), q - abs(a - b)) for b in range(q)] for a in range(q)]
    elif distance == "hamming":
        dist = [[int(a != b) for b in range(q)] for a in range(q)]
    else:
        raise ValueError(distance)
    return {"alphabet_size": q, "length": length, "num_codewords": num_codewords, "min_dist": min_dist, "dist": dist}


def check_error_correcting_codes(data, answer):
    (code,) = _answer(answer, ["codewords"])
    q, length, m = data["alphabet_size"], data["length"], data["num_codewords"]
    _matrix(code, m, length, "codewords", 0, q - 1)
    for i, j in itertools.combinations(range(m), 2):
        if code[i] == code[j]:
            raise CandidateError(f"codewords {i} and {j} are equal")
        d = sum(data["dist"][a][b] for a, b in zip(code[i], code[j]))
        if d < data["min_dist"]:
            raise CandidateError(f"codewords {i} and {j} are at distance {d} < {data['min_dist']}")


# --- covering arrays (CSPLib 45, decision version) -------------------------------------

def generate_covering_array(t, k, g, b):
    return {"t": t, "k": k, "g": g, "b": b}


def check_covering_array(data, answer):
    (matrix,) = _answer(answer, ["matrix"])
    t, k, g, b = data["t"], data["k"], data["g"], data["b"]
    _matrix(matrix, k, b, "matrix", 0, g - 1)
    for rows in itertools.combinations(range(k), t):
        seen = {tuple(matrix[r][col] for r in rows) for col in range(b)}
        if len(seen) != g ** t:
            raise CandidateError(f"rows {rows} miss {g ** t - len(seen)} of the {g ** t} value tuples")


# --- tail assignment (CSPLib 115) ---------------------------------------------------

def generate_tail_assignment(seed, n_planes, min_legs, max_legs, n_airports, restrict):
    """Plant one route per aircraft in a dated timetable; arcs are the legal connections."""
    rng = Rng(seed)
    bases = {0}
    legs, owner = [], []
    starts = []
    for p in range(n_planes):
        airport, clock = rng.randint(0, n_airports - 1), rng.randint(0, 120)
        starts.append((airport, clock))
        for _ in range(rng.randint(min_legs, max_legs)):
            dest = rng.choice([a for a in range(n_airports) if a != airport])
            depart = clock + 30 + rng.randint(0, 90)
            arrive = depart + 45 + rng.randint(0, 150)
            legs.append((airport, dest, depart, arrive))
            owner.append(p)
            airport, clock = dest, arrive
    order = rng.shuffle(list(range(len(legs))))
    legs = [legs[i] for i in order]
    owner = [owner[i] for i in order]
    n = len(legs)
    network = [[f, g] for f in range(n) for g in range(n)
               if legs[f][1] == legs[g][0] and legs[g][2] >= legs[f][3] + 30]
    start = [[g for g in range(n) if legs[g][0] == starts[p][0] and legs[g][2] >= starts[p][1] + 30] for p in range(n_planes)]
    can_fly = [[f for f in range(n) if owner[f] == p or rng.random() >= restrict] for p in range(n_planes)]
    maint = [f for f in range(n) if legs[f][1] in bases]
    longest = 0
    for p in range(n_planes):  # the planted routes fix the loosest maintenance frequency they satisfy
        route = sorted((f for f in range(n) if owner[f] == p), key=lambda f: legs[f][2])
        run = 0
        for f in route:
            run = 0 if f in maint else run + 1
            longest = max(longest, run)
    return {"n_flights": n, "n_planes": n_planes, "maint_freq": max(2, longest), "network": network,
            "can_fly": can_fly, "start": start, "maint": maint}


def check_tail_assignment(data, answer):
    (routes,) = _answer(answer, ["routes"])
    n, planes = data["n_flights"], data["n_planes"]
    if not isinstance(routes, list) or len(routes) != planes:
        raise CandidateError(f"routes must list one route per aircraft ({planes})")
    arcs = {tuple(a) for a in data["network"]}
    maint, freq = set(data["maint"]), data["maint_freq"]
    covered = []
    for p, route in enumerate(routes):
        _int_list(route, len(route) if isinstance(route, list) else -1, f"route {p}", 0, n - 1)
        if not route:
            raise CandidateError(f"aircraft {p} has an empty route (each route starts after its carry-in)")
        if route[0] not in data["start"][p]:
            raise CandidateError(f"aircraft {p} cannot start with flight {route[0]}")
        allowed = set(data["can_fly"][p])
        for f in route:
            if f not in allowed:
                raise CandidateError(f"aircraft {p} may not fly flight {f}")
        for a, b in zip(route, route[1:]):
            if (a, b) not in arcs:
                raise CandidateError(f"aircraft {p}: flight {b} cannot follow flight {a}")
        for i in range(len(route) - freq):
            if not any(f in maint for f in route[i:i + freq + 1]):
                raise CandidateError(f"aircraft {p}: no maintenance visit in flights {i}..{i + freq}")
        covered += route
    if sorted(covered) != list(range(n)):
        raise CandidateError("every flight must be covered exactly once")


PROBLEMS = {
    "car_sequencing": (generate_car_sequencing, check_car_sequencing),
    "vessel_loading": (generate_vessel_loading, check_vessel_loading),
    "social_golfers": (generate_social_golfers, check_social_golfers),
    "bibd": (generate_bibd, check_bibd),
    "bacp": (generate_bacp, check_bacp),
    "error_correcting_codes": (generate_error_correcting_codes, check_error_correcting_codes),
    "covering_array": (generate_covering_array, check_covering_array),
    "tail_assignment": (generate_tail_assignment, check_tail_assignment),
}


def score(module, params):
    """Family entry point: PAR2 runtime speedup of the candidate model over the baseline."""
    from .or_csp_baselines import BASELINES
    from .or_csp_cpsat import reformulation_score
    generate, check = PROBLEMS[params["problem"]]
    return reformulation_score(module, params["problem"], params, generate=generate, check=check,
                               baseline_source=BASELINES[params["problem"]])
