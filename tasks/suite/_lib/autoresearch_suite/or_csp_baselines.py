"""Baseline CPMpy models for the PDR / CSPLib reformulation tasks.

Each baseline is a direct, unoptimised translation of the CSPLib Essence model
of the problem (github.com/csplib/csplib, Problems/probNNN/models/*.essence),
written for this package. The same source text is the task's initial program
(the candidate to be reformulated) and the reference whose runtime defines the
speedup. All indices are 0-based. ``build_model(data)`` returns
``(cpmpy.Model, extractor)``; the extractor returns the JSON answer checked by
``or_csp_csplib``.
"""

CAR_SEQUENCING = '''import cpmpy as cp


def build_model(data):
    n, n_classes, n_options = data["n_cars"], data["n_classes"], data["n_options"]
    usage = data["usage"]  # usage[c][o] = 1 when class c requires option o
    car = cp.intvar(0, n_classes - 1, shape=n, name="car")
    model = cp.Model()
    for c in range(n_classes):
        model += cp.Count(car, c) == data["quantity"][c]
    for o in range(n_options):
        needs = cp.cpm_array([usage[c][o] for c in range(n_classes)])
        block, limit = data["blksize"][o], data["maxcars"][o]
        for start in range(n - block + 1):
            model += cp.sum(needs[car[i]] for i in range(start, start + block)) <= limit

    def extractor():
        return {"sequence": car.value().tolist()}
    return model, extractor
'''

VESSEL_LOADING = '''import cpmpy as cp


def build_model(data):
    n, width, length = data["n_containers"], data["width"], data["length"]
    klass, separation = data["container_class"], data["separation"]
    west = cp.intvar(0, data["deck_width"], shape=n, name="west")
    east = cp.intvar(0, data["deck_width"], shape=n, name="east")
    north = cp.intvar(0, data["deck_length"], shape=n, name="north")
    south = cp.intvar(0, data["deck_length"], shape=n, name="south")
    model = cp.Model()
    for c in range(n):
        model += (((east[c] - west[c] == width[c]) & (south[c] - north[c] == length[c])) |
                  ((east[c] - west[c] == length[c]) & (south[c] - north[c] == width[c])))
    for c1 in range(n):
        for c2 in range(n):
            if c1 == c2:
                continue
            model += (west[c1] != west[c2]) | (north[c1] != north[c2])
            gap_x = cp.max([west[c1], west[c2]]) - cp.min([east[c1], east[c2]])
            gap_y = cp.max([north[c1], north[c2]]) - cp.min([south[c1], south[c2]])
            model += (gap_x >= 0) | (gap_y >= 0)
            sep = separation[klass[c1]][klass[c2]]
            model += (gap_x >= sep) | (gap_y >= sep)

    def extractor():
        return {"west": west.value().tolist(), "east": east.value().tolist(),
                "north": north.value().tolist(), "south": south.value().tolist()}
    return model, extractor
'''

SOCIAL_GOLFERS = '''import cpmpy as cp


def build_model(data):
    weeks, groups, size = data["weeks"], data["groups"], data["group_size"]
    golfers = groups * size
    group = cp.intvar(0, groups - 1, shape=(weeks, golfers), name="group")  # group of each golfer per week
    model = cp.Model()
    for w in range(weeks):
        for g in range(groups):
            model += cp.Count(group[w], g) == size
    for p in range(golfers):
        for q in range(p + 1, golfers):
            model += cp.sum(group[w, p] == group[w, q] for w in range(weeks)) <= 1

    def extractor():
        return {"schedule": group.value().tolist()}
    return model, extractor
'''

BIBD = '''import cpmpy as cp


def build_model(data):
    v, b, r, k, lam = data["v"], data["b"], data["r"], data["k"], data["lam"]
    incidence = cp.boolvar(shape=(v, b), name="incidence")
    model = cp.Model()
    for o in range(v):
        model += cp.sum(incidence[o, :]) == r
    for block in range(b):
        model += cp.sum(incidence[:, block]) == k
    for o1 in range(v):
        for o2 in range(o1 + 1, v):
            model += cp.sum(incidence[o1, j] & incidence[o2, j] for j in range(b)) == lam

    def extractor():
        return {"matrix": [[int(x) for x in row] for row in incidence.value().tolist()]}
    return model, extractor
'''

BACP = '''import cpmpy as cp


def build_model(data):
    n, periods, load = data["n_courses"], data["n_periods"], data["course_load"]
    curr = cp.intvar(0, periods - 1, shape=n, name="curr")  # period of each course
    model = cp.Model()
    for before, after in data["prerequisites"]:
        model += curr[before] < curr[after]
    for p in range(periods):
        period_load = cp.sum(load[c] * (curr[c] == p) for c in range(n))
        model += period_load <= data["load_per_period_ub"]
        model += period_load >= data["load_per_period_lb"]
        count = cp.sum(curr[c] == p for c in range(n))
        model += count <= data["courses_per_period_ub"]
        model += count >= data["courses_per_period_lb"]

    def extractor():
        return {"period": curr.value().tolist()}
    return model, extractor
'''

ERROR_CORRECTING_CODES = '''import cpmpy as cp


def build_model(data):
    q, length, m = data["alphabet_size"], data["length"], data["num_codewords"]
    dist = cp.cpm_array([d for row in data["dist"] for d in row])  # dist[a][b] flattened to a*q+b
    code = cp.intvar(0, q - 1, shape=(m, length), name="code")
    model = cp.Model()
    for i in range(m):
        for j in range(i + 1, m):
            model += cp.sum(dist[code[i, l] * q + code[j, l]] for l in range(length)) >= data["min_dist"]
            model += cp.any(code[i, l] != code[j, l] for l in range(length))  # a set: codewords are distinct

    def extractor():
        return {"codewords": code.value().tolist()}
    return model, extractor
'''

COVERING_ARRAY = '''import itertools

import cpmpy as cp


def build_model(data):
    t, k, g, b = data["t"], data["k"], data["g"], data["b"]
    matrix = cp.intvar(0, g - 1, shape=(k, b), name="matrix")
    model = cp.Model()
    for rows in itertools.combinations(range(k), t):
        for values in itertools.product(range(g), repeat=t):
            model += cp.any(cp.all(matrix[rows[i], col] == values[i] for i in range(t)) for col in range(b))

    def extractor():
        return {"matrix": matrix.value().tolist()}
    return model, extractor
'''

TAIL_ASSIGNMENT = '''import cpmpy as cp


def build_model(data):
    flights, planes, freq = data["n_flights"], data["n_planes"], data["maint_freq"]
    none = flights  # value of an unused route position
    route = cp.intvar(0, flights, shape=(planes, flights), name="route")
    follows = [tuple(arc) for arc in data["network"]] + [(f, none) for f in range(flights)] + [(none, none)]
    model = cp.Model()
    for p in range(planes):
        model += cp.InDomain(route[p, 0], data["start"][p])  # first flight can follow the carry-in
        model += cp.InDomain(route[p, 0], data["can_fly"][p])
        for i in range(1, flights):
            model += cp.Table([route[p, i - 1], route[p, i]], follows)
            model += (route[p, i] == none) | cp.InDomain(route[p, i], data["can_fly"][p])
        for i in range(flights - freq):
            window = [route[p, j] for j in range(i, i + freq + 1)]
            model += (window[-1] == none) | cp.any(cp.InDomain(x, data["maint"]) for x in window)
    for f in range(flights):
        model += cp.Count(route.flatten(), f) == 1

    def extractor():
        return {"routes": [[int(x) for x in row if x != none] for row in route.value().tolist()]}
    return model, extractor
'''

BASELINES = {"car_sequencing": CAR_SEQUENCING, "vessel_loading": VESSEL_LOADING, "social_golfers": SOCIAL_GOLFERS,
             "bibd": BIBD, "bacp": BACP, "error_correcting_codes": ERROR_CORRECTING_CODES,
             "covering_array": COVERING_ARRAY, "tail_assignment": TAIL_ASSIGNMENT}
