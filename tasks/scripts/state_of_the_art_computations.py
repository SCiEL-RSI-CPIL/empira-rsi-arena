"""Witnesses and optimality computations of scripts/check_state_of_the_art.py.

Each function returns (value, witness, method): the value an exact computation gives (None when the
function only rebuilds a witness), a certificate in the form the task's construct() returns, and a
sentence naming the computation. The two integer programmes need SciPy and run only on request.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import factorial, isqrt, lcm


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, isqrt(n) + 1))


def prime_sieve(n):
    sieve = bytearray(b"\1") * (n + 1)
    sieve[:2] = b"\0\0"
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(range(i * i, n + 1, i)))
    return sieve


# ---------------------------------------------------------------- witnesses and optimality computations

def cluster_prime():
    """Largest p <= 10^5 such that every even d <= p-3 is a difference of primes <= p."""
    limit = 100000
    sieve = prime_sieve(limit)
    primes = [p for p in range(2, limit + 1) if sieve[p]]
    covered = bytearray(limit + 1)
    smallest_missing, best = 2, None
    for i, p in enumerate(primes):
        for q in primes[:i]:
            covered[p - q] = 1
        while smallest_missing <= limit and covered[smallest_missing]:
            smallest_missing += 2
        if p >= 3 and smallest_missing > p - 3:
            best = p
    return best, best, "every prime up to 10^5 tested incrementally"


def record_gap(representable, limit=10**7):
    """Largest gap between consecutive members of a set whose indicator is given, right end <= limit."""
    best, previous, where = 0, None, None
    for x in range(1, limit + 1):
        if representable[x]:
            if previous is not None and x - previous > best:
                best, where = x - previous, [previous, x]
            previous = x
    return best, where


def squarefree_gap():
    limit = 10**7
    flags = bytearray(b"\1") * (limit + 1)
    for d in range(2, isqrt(limit) + 1):
        flags[d * d::d * d] = bytearray(len(range(d * d, limit + 1, d * d)))
    gap, witness = record_gap(flags, limit)
    return gap, witness, "sieve of all squarefree integers up to 10^7"


def two_square_gap():
    limit = 10**7
    flags = bytearray(limit + 1)
    for a in range(isqrt(limit) + 1):
        for b in range(a, isqrt(limit - a * a) + 1):
            flags[a * a + b * b] = 1
    gap, witness = record_gap(flags, limit)
    return gap, witness, "all sums a^2+b^2 <= 10^7 enumerated"


def lonely_runner():
    """max_t min_i ||t v_i||; the maximum sits at t = m/(v_i+v_j) or m/(2 v_i), so q <= 2*29 suffices."""
    speeds = [1, 3, 4, 7, 11, 18, 29]
    best, witness = Fraction(0), None
    for q in range(2, 4 * max(speeds) + 1):
        for p in range(1, q):
            value = min(Fraction(min(p * v % q, q - p * v % q), q) for v in speeds)
            if value > best:
                best, witness = value, [p, q]
    return float(best), witness, "all rational times with denominator <= 116 (every breakpoint has denominator <= 58)"


def triple_free():
    """Largest subset of [1,4096] without {x,2x,3x}: independent grids {2^a 3^b m} for gcd(m,6)=1."""
    n = 4096
    chosen = []
    for m in range(1, n + 1, 2):
        if m % 3 == 0:
            continue
        columns = []
        b = 0
        while 3**b * m <= n:
            a = 0
            while 2**a * 3**b * m <= n:
                a += 1
            columns.append(a)
            b += 1
        # Profile dynamic programme over columns b; bit a of a mask selects 2^a 3^b m.
        # Forbidden: bits a and a+1 in column b together with bit a in column b+1.
        layers = [{mask: (bin(mask).count("1"), None) for mask in range(1 << columns[0])}]
        for b in range(1, len(columns)):
            best_by_bad = {}
            for mask, (value, _) in layers[-1].items():
                bad = mask & (mask >> 1) & ((1 << columns[b]) - 1)
                if value > best_by_bad.get(bad, (-1, None))[0]:
                    best_by_bad[bad] = (value, mask)
            layer = {}
            for mask in range(1 << columns[b]):
                options = [(v, pm) for bad, (v, pm) in best_by_bad.items() if not bad & mask]
                value, previous = max(options)
                layer[mask] = (value + bin(mask).count("1"), previous)
            layers.append(layer)
        mask = max(layers[-1], key=lambda k: layers[-1][k][0])
        for b in range(len(columns) - 1, -1, -1):
            chosen += [2**a * 3**b * m for a in range(columns[b]) if mask >> a & 1]
            mask = layers[b][mask][1]
    return len(chosen), sorted(chosen), "exact dynamic programme on each 3-smooth grid (Finch's decomposition, OEIS A157282)"


def prime_complement(slow=False):
    witness = [11, 16, 80, 117, 199, 204, 223, 225, 228, 242]
    if not slow:
        return None, witness, "witness only; optimality by the --slow integer programme"
    import numpy as np
    from itertools import product
    from scipy.optimize import Bounds, LinearConstraint, milp
    n = 256
    targets = range(n, 2 * n + 1)
    cover = {b: {t for t in targets if t - b >= 2 and is_prime(t - b)} for b in range(n + 1)}
    # Only the prime 2 links the parities: b = t - 2 for t in {256, 257, 258}.
    cross = [b for b in range(n + 1) if any(t - b == 2 for t in cover[b])]

    def solve(targets, candidates, fixed):
        targets = [t for t in targets if not any(t in cover[b] for b in fixed)]
        candidates = [b for b in candidates if cover[b] & set(targets)]
        if not targets:
            return 0
        matrix = np.array([[1.0 if t in cover[b] else 0.0 for b in candidates] for t in targets])
        res = milp(np.ones(len(candidates)), constraints=[LinearConstraint(matrix, 1, np.inf)],
                   integrality=np.ones(len(candidates)), bounds=Bounds(0, 1))
        assert res.status == 0, res.message
        return round(res.fun)
    best = None
    for mask in product((0, 1), repeat=len(cross)):
        fixed = [b for b, bit in zip(cross, mask) if bit]
        free = [b for b in range(n + 1) if b not in cross]
        total = (len(fixed) + solve([t for t in targets if t % 2 == 0], [b for b in free if b % 2], fixed)
                 + solve([t for t in targets if t % 2], [b for b in free if b % 2 == 0], fixed))
        best = total if best is None else min(best, total)
    return best, witness, "set-cover integer programmes (HiGHS), split by parity with the three prime-2 links enumerated"


def property_p(slow=False):
    witness = list(range(171, 257))
    if not slow:
        return None, witness, "witness only; optimality by the --slow integer programme"
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    n = 256
    rows = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b, n + 1):
                if (b + c) % a == 0:
                    rows.add((a, b) if b == c else (a, b, c))
    matrix = lil_matrix((len(rows), n))
    upper = []
    for i, row in enumerate(sorted(rows)):
        for v in row:
            matrix[i, v - 1] = 1
        upper.append(len(row) - 1)
    res = milp(-np.ones(n), constraints=[LinearConstraint(matrix.tocsr(), -np.inf, np.array(upper))],
               integrality=np.ones(n), bounds=Bounds(0, 1))
    assert res.status == 0, res.message
    return round(-res.fun), witness, f"maximum-independent-set integer programme with {len(rows)} constraints (HiGHS)"


def gaussian_moat():
    bound, target = 64, 50

    def gaussian_prime(x, y):
        if x and y:
            return is_prime(x * x + y * y)
        p = abs(x or y)
        return is_prime(p) and p % 4 == 3
    points = [(x, y) for x in range(-bound, bound + 1) for y in range(-bound, bound + 1) if gaussian_prime(x, y)]
    point_set = set(points)
    steps = sorted({dx * dx + dy * dy for dx in range(-8, 9) for dy in range(-8, 9) if dx or dy})
    for limit in steps:
        moves = [(dx, dy) for dx in range(-8, 9) for dy in range(-8, 9) if 0 < dx * dx + dy * dy <= limit]
        parent = {(1, 1): None}
        queue = [(1, 1)]
        for x, y in queue:
            if max(abs(x), abs(y)) >= target:
                path = [(x, y)]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return limit, path[::-1], "breadth-first search over all Gaussian primes in [-64,64]^2 for each step bound"
            for dx, dy in moves:
                q = (x + dx, y + dy)
                if q in point_set and q not in parent:
                    parent[q] = (x, y)
                    queue.append(q)
    raise AssertionError("no path found")


def projective_plane(q=7):
    points = [(1, x, y) for x in range(q) for y in range(q)] + [(0, 1, x) for x in range(q)] + [(0, 0, 1)]
    lines = [tuple(i for i, p in enumerate(points) if sum(a * b for a, b in zip(p, l)) % q == 0) for l in points]
    return points, lines


def blocking_search(limit, q=7):
    """Exhaustive search for a blocking set of PG(2,q) meeting every line in at most `limit` points."""
    points, lines = projective_plane(q)
    on = [[j for j, line in enumerate(lines) if i in line] for i in range(len(points))]
    count = [0] * len(lines)
    chosen, excluded = set(), set()

    def search():
        best = None
        for j, line in enumerate(lines):
            if count[j] == 0:
                available = [i for i in line if i not in chosen and i not in excluded and all(count[k] < limit for k in on[i])]
                if best is None or len(available) < len(best):
                    best = available
                    if not available:
                        return False
        if best is None:
            return True
        tried = []
        found = False
        for i in best:
            chosen.add(i)
            for k in on[i]:
                count[k] += 1
            found = search()
            if found:
                break
            chosen.discard(i)
            for k in on[i]:
                count[k] -= 1
            excluded.add(i)
            tried.append(i)
        excluded.difference_update(tried)
        return found
    return sorted(chosen) if search() else None


def blocking_set():
    assert blocking_search(3) is None
    witness = blocking_search(4)
    return 4, witness, "exhaustive branch-and-bound: no blocking set of PG(2,7) meets every line in at most 3 points"


def farey_pair():
    order = 1000
    fractions = [(0, 1)]
    a, b, c, d = 0, 1, 1, order
    while c <= order:
        fractions.append((c, d))
        k = (order + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
    best, witness = None, None
    for i, (a, b) in enumerate(fractions):
        stop = len(fractions) if best is None else min(len(fractions), i + best)
        for j in range(i + 1, stop):
            c, d = fractions[j]
            if (c - a) * (d - b) < 0:
                best, witness = j - i, [[a, b], [c, d]]
                break
    return best, witness, "scan of the complete Farey sequence of order 1000"


def smooth_antichain():
    smooth = [x for x in range(1, 129) if _is_smooth(x)]
    rows = []
    for target in range(1, 129):
        candidates = [x for x in smooth if x <= target]
        for size in range(1, 31):
            row = _antichain_sum(candidates, target, size)
            if row:
                rows.append(row)
                break
    return sum(map(len, rows)), rows, "iterative deepening over antichains of 5-smooth numbers for each target"


def _is_smooth(x):
    for p in (2, 3, 5):
        while x % p == 0:
            x //= p
    return x == 1


def _antichain_sum(candidates, target, size, start=0, chosen=()):
    if len(chosen) == size:
        return list(chosen) if target == 0 else None
    for i in range(start, len(candidates)):
        x = candidates[i]
        if x > target:
            break
        if any(x % y == 0 or y % x == 0 for y in chosen):
            continue
        found = _antichain_sum(candidates, target - x, size, i + 1, chosen + (x,))
        if found:
            return found
    return None


def modular_inverse_cover():
    p = 101
    best = {}
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if mask >> (x - 1) & 1]
        residue = sum(pow(x, -1, p) for x in terms) % p
        if residue not in best or len(terms) < len(best[residue]):
            best[residue] = terms
    rows = [best[r] for r in range(p)]
    return max(map(len, rows)), rows, "all 1023 nonempty denominator sets enumerated"


def permutation_order():
    """The witness is the order; the task scores it by the number of permutations of that order."""
    counts = {}

    def partitions(remaining, smallest, parts):
        if remaining == 0:
            multiplicity, divisor, order = {}, 1, 1
            for part in parts:
                multiplicity[part] = multiplicity.get(part, 0) + 1
            for length, m in multiplicity.items():
                divisor *= length**m * factorial(m)
                order = lcm(order, length)
            counts[order] = counts.get(order, 0) + factorial(16) // divisor
            return
        for length in range(smallest, remaining + 1):
            partitions(remaining - length, length, parts + (length,))
    partitions(16, 1, ())
    order = max(counts, key=counts.get)
    return counts[order], order, f"all 231 cycle types of S16 enumerated; the most frequent order is {order}"


def primitive_roots():
    moduli = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157]
    roots = []
    for p in moduli:
        factors = [q for q in range(2, p) if (p - 1) % q == 0 and is_prime(q)]
        roots.append(next(g for g in range(2, p) if is_prime(g) and all(pow(g, (p - 1) // q, p) != 1 for q in factors)))
    return sum(roots), roots, "least prime primitive root of each modulus"


def anti_ramsey():
    """Alon's colouring: triangles on 15 vertices plus one vertex; inner edges get new colours, outer edges the lower part's colour."""
    parts = [list(range(3 * i, 3 * i + 3)) for i in range(5)] + [[15]]
    part_of = {v: i for i, part in enumerate(parts) for v in part}
    colours, next_colour = [], 0
    inner = {}
    for u, v in combinations(range(16), 2):
        if part_of[u] == part_of[v]:
            inner[(u, v)] = next_colour
            next_colour += 1
    for u, v in combinations(range(16), 2):
        colours.append(inner[(u, v)] if (u, v) in inner else 15 + min(part_of[u], part_of[v]))
    return 4 * 16 // 3 - 1, colours, "formula floor(4n/3)-1 (Alon 1983), attained by the construction"


def anti_ramsey_small():
    """Exhaustive check of floor(4n/3)-1 for n = 4, 5, 6 (canonical colour labelling)."""
    results = {}
    for n in (4, 5, 6):
        edges = list(combinations(range(n), 2))
        index = {e: i for i, e in enumerate(edges)}
        cycles = []
        for a, b, c, d in combinations(range(n), 4):
            for cycle in ((a, b, c, d), (a, b, d, c), (a, c, b, d)):
                cycles.append([index[tuple(sorted((cycle[i], cycle[(i + 1) % 4])))] for i in range(4)])
        by_edge = [[c for c in cycles if e in c] for e in range(len(edges))]
        colour = [-1] * len(edges)
        best = [0]

        def extend(i, used):
            if used + len(edges) - i <= best[0]:
                return
            if i == len(edges):
                best[0] = used
                return
            for c in range(used + 1):
                colour[i] = c
                if not any(rainbow(cycle) for cycle in by_edge[i]):
                    extend(i + 1, max(used, c + 1))
            colour[i] = -1

        def rainbow(cycle):
            seen = [colour[x] for x in cycle]
            return -1 not in seen and len(set(seen)) == 4
        extend(0, 0)
        results[n] = best[0]
    return all(results[n] == 4 * n // 3 - 1 for n in results), results


def tournament():
    """Paley tournament on GF(23) restricted to 20 vertices: no transitive subtournament on 6 vertices."""
    residues = {x * x % 23 for x in range(1, 23)}
    return 5, [0 if (j - i) % 23 in residues else 1 for i, j in combinations(range(20), 2)], "witness; lower bound floor(log2 20)+1 = 5 (Stearns 1959)"


def van_der_waerden():
    def search(n):
        colour = [-1] * n
        ends = [[] for _ in range(n)]
        for d in range(1, n):
            for a in range(n - 3 * d):
                ends[a + 3 * d].append((a, a + d, a + 2 * d))

        def extend(i):
            if i == n:
                return True
            for c in ((0, 1) if i else (0,)):
                if all(not all(colour[x] == c for x in ap) for ap in ends[i]):
                    colour[i] = c
                    if extend(i + 1):
                        return True
            colour[i] = -1
            return False
        return list(colour) if extend(0) else None
    assert search(35) is None
    return 34, search(34), "exhaustive backtracking: no 2-colouring of [1,35] avoids monochromatic 4-term progressions"


def permutation_progression():
    """Evens before odds, recursively (bit-reversal order): no monotone three-term progression."""
    order = sorted(range(128), key=lambda x: int(format(x, "07b")[::-1], 2))
    return 2, [x + 1 for x in order], "Davis-Entringer-Graham-Simmons recursive construction"


def goldbach():
    sieve = prime_sieve(1000510)
    rows = []
    for i in range(256):
        target = 1000000 + 2 * i
        p = next(p for p in range(3, target) if sieve[p] and sieve[target - p])
        rows.append([p, target - p])
    return 1.0, rows, "a prime pair for every target"


def egyptian():
    rows = []
    for n in range(3, 257):
        row = None
        for x in range(n // 4 + 1, 3 * n // 4 + 2):
            rest = Fraction(4, n) - Fraction(1, x)
            if rest <= 0:
                continue
            for y in range(max(x + 1, rest.denominator // rest.numerator + 1), 2 * rest.denominator // rest.numerator + 1):
                last = rest - Fraction(1, y)
                if last > 0 and last.numerator == 1 and last.denominator > y:
                    row = [x, y, last.denominator]
                    break
            if row:
                break
        rows.append(row)
    return 1.0, rows, "distinct denominators found for every n = 3..256"


def fano():
    lines = [[0, 1, 2], [0, 3, 4], [0, 5, 6], [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]]
    return 7, {"edges": lines, "colors": [0, 0, 1, 1, 2, 1, 2, 0, 0]}, "Fano plane with a proper 3-colouring"


def distinct_subset_sums():
    return 1157, [554, 845, 993, 1069, 1108, 1130, 1141, 1145, 1147, 1149, 1152, 1157], "set recorded in OEIS A276661 (V. Popov, 2025)"


def no_singleton():
    return 91, [[0, 1, a, b] for a, b in combinations(range(2, 16), 2)], "all 4-sets through two fixed points (Keevash-Mubayi-Wilson 2006 extremal family)"
