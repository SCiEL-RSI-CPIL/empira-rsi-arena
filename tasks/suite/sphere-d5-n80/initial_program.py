"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIM = 5
N = 80
# EVOLVE-BLOCK-START
def construct():
    # Random start, then repeatedly push every point away from its nearest neighbours (a crude repulsion).
    import numpy as np

    rng = np.random.default_rng(0)
    p = rng.normal(size=(N, DIM))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    step = 0.05
    for it in range(400):
        diff = p[:, None, :] - p[None, :, :]
        dist2 = (diff ** 2).sum(-1) + np.eye(N)
        force = (diff / dist2[..., None] ** 2).sum(1)
        p = p + step * force / (np.linalg.norm(force, axis=1, keepdims=True) + 1e-12)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        step *= 0.99
    return p
# EVOLVE-BLOCK-END
