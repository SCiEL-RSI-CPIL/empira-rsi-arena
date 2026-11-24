"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The tree generator is copied from the verifier: trees[k - 1] lists the edges of T_k on vertices 0..k-1.
    # Entry k-1 of the answer gives the host vertex of each vertex 0..k-1 of T_k, or None to leave T_k out.
    from random import Random
    def instance_trees(n, seed):
        rng = Random(seed)
        trees = []
        for k in range(1, n + 1):
            if k <= 2:
                trees.append([] if k == 1 else [(0, 1)])
                continue
            code = [rng.randrange(k) for _ in range(k - 2)]
            degree = [1] * k
            for x in code:
                degree[x] += 1
            edges = []
            for x in code:
                leaf = min(v for v in range(k) if degree[v] == 1)
                edges.append((leaf, x))
                degree[leaf] -= 1
                degree[x] -= 1
            u, w = [v for v in range(k) if degree[v] == 1]
            edges.append((u, w))
            trees.append(edges)
        return trees
    trees = instance_trees(20, 611)
    # Only the spanning tree T20, embedded by the identity map.
    return [None] * 19 + [list(range(len(trees[19]) + 1))]
# EVOLVE-BLOCK-END
