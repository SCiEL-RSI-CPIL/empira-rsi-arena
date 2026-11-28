"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import cpmpy as cp


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
# EVOLVE-BLOCK-END
