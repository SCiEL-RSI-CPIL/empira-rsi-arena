"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

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
# EVOLVE-BLOCK-END
