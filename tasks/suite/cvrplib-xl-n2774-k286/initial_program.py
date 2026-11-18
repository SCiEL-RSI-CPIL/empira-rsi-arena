"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def solve(inst):
    # Sweep heuristic: sort customers by angle around the depot and cut the sequence into routes by capacity.
    xy, dem, cap, n = inst["coords"], inst["demand"], inst["capacity"], inst["n"]
    angle = np.arctan2(xy[1:, 1] - xy[0, 1], xy[1:, 0] - xy[0, 0])
    routes, cur, load = [], [], 0
    for c in (np.argsort(angle) + 1).tolist():
        if cur and load + dem[c] > cap:
            routes.append(cur)
            cur, load = [], 0
        cur.append(c)
        load += int(dem[c])
    if cur:
        routes.append(cur)
    return routes
# EVOLVE-BLOCK-END
