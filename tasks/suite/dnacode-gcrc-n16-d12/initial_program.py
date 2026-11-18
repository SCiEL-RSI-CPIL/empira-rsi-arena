"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 12
RC = True
# EVOLVE-BLOCK-START
def construct():
    # Random words with exactly N//2 letters from {C, G}, accepted greedily while the distance conditions hold (15 s budget).
    import time

    import numpy as np

    rng = np.random.default_rng(0)
    comp = np.array([1, 0, 3, 2], dtype=np.int8)  # letters coded A=0, T=1, C=2, G=3
    w = N // 2
    code = np.empty((0, N), dtype=np.int8)
    start = time.time()
    while time.time() - start < 15:
        batch = rng.integers(0, 2, size=(3000, N)).astype(np.int8)
        gc = np.zeros((3000, N), dtype=bool)
        for row in range(3000):
            gc[row, rng.choice(N, w, replace=False)] = True
        batch = np.where(gc, batch + 2, batch).astype(np.int8)
        for word in batch:
            if RC and (word != comp[word[::-1]]).sum() < D:
                continue
            if len(code):
                if ((code != word).sum(1) < D).any():
                    continue
                if RC and ((comp[code[:, ::-1]] != word).sum(1) < D).any():
                    continue
            code = np.vstack([code, word])
    return ["".join("ATCG"[c] for c in row) for row in code]
# EVOLVE-BLOCK-END
