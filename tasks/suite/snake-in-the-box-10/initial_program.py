"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 10
# EVOLVE-BLOCK-START
def construct():
    # Greedy snake: always step to the lowest-bit neighbour that keeps the path induced.
    count = {1 << b: 1 for b in range(N)}
    path, used = [0], {0}
    while True:
        v = path[-1]
        nxt = None
        for b in range(N):
            u = v ^ (1 << b)
            if u not in used and count.get(u, 0) == 1:
                nxt = u
                break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)
        for b in range(N):
            w = nxt ^ (1 << b)
            count[w] = count.get(w, 0) + 1
# EVOLVE-BLOCK-END
