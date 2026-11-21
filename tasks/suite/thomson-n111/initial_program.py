"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 111
# EVOLVE-BLOCK-START
def construct():
    # Random start and projected gradient descent on the Coulomb energy.
    import numpy as np

    rng = np.random.default_rng(0)
    p = rng.normal(size=(N, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    step = 0.01 * N ** -0.5
    for it in range(1500):
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1)) + np.eye(N)
        grad = (diff / dist[..., None] ** 3).sum(1)
        grad -= (grad * p).sum(1, keepdims=True) * p
        p = p + step * grad
        p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p
# EVOLVE-BLOCK-END
