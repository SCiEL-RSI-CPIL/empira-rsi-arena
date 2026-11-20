"""Exact check for (k, g)-graphs: simple k-regular graphs of girth at least g.

The cage problem asks for the fewest vertices n(k, g) of such a graph. The
evaluator validates the edge list (simple, exactly k-regular) and certifies
that no cycle is shorter than g. Girth is certified with non-backtracking walks
grown from every vertex at once (NumPy, processed in chunks of sources):

* two distinct non-backtracking walks of length d from one source that end at
  the same vertex close a cycle of length at most 2d;
* a walk of length d whose next step lands on the end of another length-d walk
  from the same source closes an odd closed walk, hence an odd cycle, of
  length at most 2d + 1.

If the shortest cycle has length L, every vertex on it detects a violation of
length exactly L at depth floor(L / 2), so no cycle shorter than g is missed.
``girth_reference`` is a plain breadth-first implementation used in tests.
"""
from __future__ import annotations

import collections

import numpy as np

from .frontier_combinatorics_designs import integer, sequence
from .runtime import require


def moore_bound(k, g):
    """Least possible order of a k-regular graph of girth g."""
    if g % 2:
        return 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
    return 2 * sum((k - 1) ** i for i in range(g // 2))


def regular_girth_graph(n, edges, k, g, max_vertices):
    """Return n for a simple k-regular graph on vertices 0..n-1 of girth >= g."""
    n, k, g = integer(n, "n"), integer(k, "k"), integer(g, "g")
    limit = integer(max_vertices, "max_vertices")
    require(3 <= k <= 16 and 3 <= g <= 40, "require 3 <= k <= 16 and 3 <= g <= 40")
    floor = moore_bound(k, g)
    require(floor <= n <= limit, f"n must lie between the Moore bound {floor} and {limit}")
    require(n * k % 2 == 0, "n * k must be even for a k-regular graph")
    edges = sequence(edges, "edges")
    require(len(edges) == n * k // 2, f"a {k}-regular graph on {n} vertices has exactly {n * k // 2} edges")
    neighbours = np.full((n, k), -1, dtype=np.int64)
    degree = np.zeros(n, dtype=np.int64)
    seen = set()
    for index, edge in enumerate(edges):
        edge = sequence(edge, f"edge {index}")
        require(len(edge) == 2, "each edge must have exactly two endpoints")
        a, b = (integer(x, "vertex") for x in edge)
        require(0 <= a < n and 0 <= b < n, f"vertices must be integers in 0..{n - 1}")
        require(a != b, "loops are not permitted")
        key = (min(a, b), max(a, b))
        require(key not in seen, f"edge {key} is repeated")
        seen.add(key)
        require(degree[a] < k and degree[b] < k, f"a vertex has degree above {k}")
        neighbours[a, degree[a]] = b
        neighbours[b, degree[b]] = a
        degree[a] += 1
        degree[b] += 1
    require(bool((degree == k).all()), f"every vertex must have degree exactly {k}")
    shortest = shortest_cycle_below(neighbours, g)
    require(shortest is None, f"the graph has a cycle of length {shortest}, shorter than g = {g}")
    return n, {"vertices": n, "degree": k, "girth_at_least": g, "moore_bound": floor,
               "verification": "nonbacktracking_walk_certificate"}


def shortest_cycle_below(neighbours, g, chunk_entries=4_000_000):
    """Length of a shortest cycle if one is shorter than g, else None (k-regular input)."""
    n, k = neighbours.shape
    depth_limit = (g - 1) // 2
    per_source = k * max(1, k - 1) ** max(0, depth_limit)
    step = max(1, chunk_entries // per_source)
    best = None
    for start in range(0, n, step):
        found = _chunk(neighbours, np.arange(start, min(n, start + step), dtype=np.int64), g, depth_limit)
        if found is not None and (best is None or found < best):
            best = found
    return best


def _chunk(neighbours, sources, g, depth_limit):
    """Shortest violation (< g) for walks that start in ``sources``, else None."""
    n, k = neighbours.shape
    src = np.repeat(sources, k)
    cur = neighbours[sources].ravel()
    prev = src.copy()
    for d in range(1, depth_limit + 1):
        if (cur == src).any():  # a closed non-backtracking walk of length d
            return d
        order = np.sort(src * n + cur)
        if (order[1:] == order[:-1]).any():  # two distinct length-d walks meet
            return 2 * d
        if 2 * d + 1 >= g:  # no odd cycle shorter than g can close at this depth
            return None
        # Extend every walk by one non-backtracking step.
        nxt = neighbours[cur]  # (walks, k)
        keep = (nxt != prev[:, None]).ravel()
        rows = np.repeat(np.arange(len(cur)), k)[keep]
        new_cur = nxt.ravel()[keep]
        new_src, new_prev = src[rows], cur[rows]
        query = new_src * n + new_cur
        position = np.minimum(np.searchsorted(order, query), len(order) - 1)
        if (order[position] == query).any():  # a length-(d+1) walk ends where a length-d walk ends
            return 2 * d + 1
        src, cur, prev = new_src, new_cur, new_prev
    return None


def girth_reference(n, edges, limit=64):
    """Plain BFS girth (or ``limit`` when no cycle is shorter), for tests."""
    adjacency = collections.defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    best = limit
    for s in range(n):
        dist, parent, order = {s: 0}, {s: -1}, [s]
        for u in order:
            if 2 * dist[u] + 1 >= best:
                break
            for w in adjacency[u]:
                if w not in dist:
                    dist[w], parent[w] = dist[u] + 1, u
                    order.append(w)
                elif w != parent[u]:
                    best = min(best, dist[u] + dist[w] + 1)
    return best
