"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import cpmpy as cp


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
# EVOLVE-BLOCK-END
