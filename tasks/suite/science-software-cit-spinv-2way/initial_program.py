"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Seed for the constrained covering-array tasks: a constraint-aware greedy (one row at a time).

Each row starts from the smallest uncovered combination, then fixes the remaining parameters in index
order, choosing the level that covers the most uncovered combinations with the levels fixed so far
while a forward-checking search confirms that the partial row still extends to a valid configuration
(in the spirit of AETG with a satisfiability check, Cohen, Dwyer and Shi 2008).
"""
import itertools


def _satisfiable(domains, forbidden):
    """Forward-checking search: can every parameter keep one level without completing a forbidden combination?"""
    domains = [set(d) for d in domains]
    changed = True
    while changed:
        changed = False
        for combination in forbidden:
            open_literal = None
            alive = True
            for p, v in combination:
                if v not in domains[p]:
                    alive = False
                    break
                if len(domains[p]) > 1:
                    if open_literal is not None:
                        alive = False
                        break
                    open_literal = (p, v)
            if not alive:
                continue
            if open_literal is None:
                return False
            p, v = open_literal
            domains[p].discard(v)
            if not domains[p]:
                return False
            changed = True
    undecided = [p for p, d in enumerate(domains) if len(d) > 1]
    if not undecided:
        return True
    p = min(undecided, key=lambda q: len(domains[q]))
    for v in sorted(domains[p]):
        trial = list(domains)
        trial[p] = {v}
        if _satisfiable(trial, forbidden):
            return True
    return False


def generate(model):
    # Constraint-aware greedy (one row at a time, in the spirit of AETG with a satisfiability check).
    t, values, forbidden = model["strength"], model["values"], model["forbidden"]
    k = len(values)
    constrained = sorted({p for combination in forbidden for p, _ in combination})
    local = {p: i for i, p in enumerate(constrained)}
    local_forbidden = [[(local[p], v) for p, v in c] for c in forbidden]

    def feasible(row):
        domains = [{row[p]} if row[p] is not None else set(range(values[p])) for p in constrained]
        return _satisfiable(domains, local_forbidden)

    uncovered = set()
    for params in itertools.combinations(range(k), t):
        for levels in itertools.product(*(range(values[p]) for p in params)):
            uncovered.add(tuple(zip(params, levels)))
    rows = []
    while uncovered:
        target = min(uncovered)
        row = [None] * k
        for p, v in target:
            row[p] = v
        if not feasible(row):
            uncovered.discard(target)  # no valid configuration contains it
            continue
        for p in range(k):
            if row[p] is not None:
                continue
            fixed = [q for q in range(k) if row[q] is not None]

            def gain(v):
                total = 0
                for others in itertools.combinations(fixed, t - 1):
                    combo = tuple(sorted([(q, row[q]) for q in others] + [(p, v)]))
                    total += combo in uncovered
                return total
            for v in sorted(range(values[p]), key=lambda v: (-gain(v), v)):
                row[p] = v
                if p not in local or feasible(row):
                    break
            else:
                raise RuntimeError("no valid level")
        rows.append(row)
        for params in itertools.combinations(range(k), t):
            uncovered.discard(tuple((p, row[p]) for p in params))
    return rows
# EVOLVE-BLOCK-END
