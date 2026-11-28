"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import cpmpy as cp


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
# EVOLVE-BLOCK-END
