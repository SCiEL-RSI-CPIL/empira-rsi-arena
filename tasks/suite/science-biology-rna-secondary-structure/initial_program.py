"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Nussinov-style maximum-weight base pairing with a minimum hairpin length.'''

WEIGHT = {("G", "C"): 3.0, ("C", "G"): 3.0, ("A", "U"): 2.0, ("U", "A"): 2.0, ("G", "U"): 1.0, ("U", "G"): 1.0}
MIN_LOOP = 4


def predict_structure(sequence):
    n = len(sequence)
    best = [[0.0] * n for _ in range(n)]
    split = [[-1] * n for _ in range(n)]
    for span in range(MIN_LOOP + 1, n):
        for i in range(0, n - span):
            j = i + span
            value, choice = best[i][j - 1], -1               # j unpaired
            weight = WEIGHT.get((sequence[i], sequence[j]), 0.0)
            if weight > 0:
                inner = best[i + 1][j - 1] if j - 1 >= i + 1 else 0.0
                if inner + weight > value:
                    value, choice = inner + weight, i
            for k in range(i + 1, j - MIN_LOOP):
                weight = WEIGHT.get((sequence[k], sequence[j]), 0.0)
                if weight <= 0:
                    continue
                inner = best[k + 1][j - 1] if j - 1 >= k + 1 else 0.0
                total = best[i][k - 1] + inner + weight
                if total > value:
                    value, choice = total, k
            best[i][j], split[i][j] = value, choice
    structure = ["."] * n
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if i >= j:
            continue
        k = split[i][j]
        if k < 0:
            stack.append((i, j - 1))
            continue
        structure[k], structure[j] = "(", ")"
        stack.append((k + 1, j - 1))
        if k > i:
            stack.append((i, k - 1))
    return "".join(structure)
# EVOLVE-BLOCK-END
