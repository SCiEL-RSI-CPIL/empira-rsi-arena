"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import cpmpy as cp


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
# EVOLVE-BLOCK-END
