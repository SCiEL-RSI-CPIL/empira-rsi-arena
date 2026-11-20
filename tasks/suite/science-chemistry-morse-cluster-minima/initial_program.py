"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax


def construct(instance):
    """Best of four random compact starting clusters, each relaxed locally (a weak but valid seed)."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}[potential]
    rng = np.random.default_rng(size)
    radius = spacing * size ** (1.0 / 3.0) * 0.75
    best = None
    for _ in range(4):
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential, parameter)
        if best is None or energy < best[0]:
            best = (energy, positions)
    return best[1]
# EVOLVE-BLOCK-END
