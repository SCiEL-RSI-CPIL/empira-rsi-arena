"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import cpmpy as cp


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
# EVOLVE-BLOCK-END
