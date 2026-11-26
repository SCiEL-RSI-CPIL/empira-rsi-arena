"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import cpmpy as cp


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
# EVOLVE-BLOCK-END
