"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Seed for programming by example: bottom-up enumeration with observational equivalence (as in EUSolver-style search).

Terms of the grammar are built by size; a new term is kept only if its outputs on the examples differ from every
kept term of the same nonterminal. The first start-symbol term that reproduces all example outputs is returned;
otherwise the first parameter is returned. No ranking or generalisation heuristics are used.
"""
import itertools

MAX_SIZE, MAX_TERMS, BUDGET = 7, 3000, 40000  # term size, kept terms per nonterminal, evaluated combinations per task


def _decode(p):
    if isinstance(p, dict):
        return ("str", p["str"])
    if isinstance(p, list):
        return [_decode(x) for x in p]
    return p


def _text(term):
    if isinstance(term, tuple):
        return '"' + term[1].replace('"', '""') + '"'
    if isinstance(term, list):
        return "(" + " ".join(_text(t) for t in term) + ")"
    return str(term)


def _apply(op, values):
    try:
        if op == "str.++":
            out = "".join(values)
            return out if len(out) <= 200 else None
        if op == "str.len":
            return len(values[0])
        if op == "str.at":
            s, i = values
            return s[i] if 0 <= i < len(s) else ""
        if op == "str.substr":
            s, i, n = values
            return s[i:i + n] if 0 <= i < len(s) and n > 0 else ""
        if op == "str.indexof":
            s, t, i = values
            return s.find(t, i) if 0 <= i <= len(s) else -1
        if op == "str.replace":
            s, t, u = values
            return u + s if t == "" else s.replace(t, u, 1)
        if op in ("str.to.int", "str.to_int"):
            s = values[0]
            return int(s) if s and s.isascii() and s.isdigit() else -1
        if op in ("int.to.str", "str.from_int"):
            return str(values[0]) if values[0] >= 0 else ""
        if op == "str.prefixof":
            return values[1].startswith(values[0])
        if op == "str.suffixof":
            return values[1].endswith(values[0])
        if op == "str.contains":
            return values[1] in values[0]
        if op == "+":
            return sum(values)
        if op == "-":
            return values[0] - values[1]
        if op == "=":
            return values[0] == values[1]
        if op == "ite":
            return values[1] if values[0] else values[2]
    except (TypeError, ValueError):
        return None
    return None


def synthesize(task):
    names = [p for p, _ in task["params"]]
    rows = [inputs for inputs, _ in task["examples"]]
    target = tuple(output for _, output in task["examples"])
    grammar = {nt: [_decode(p) for p in productions] for nt, _, productions in task["grammar"]}
    bank = {nt: {} for nt in grammar}  # nt -> {size: [(term, values)]}
    seen = {nt: set() for nt in grammar}

    def add(nt, size, term, values):
        if values is None or values in seen[nt] or len(seen[nt]) >= MAX_TERMS:
            return False
        seen[nt].add(values)
        bank[nt].setdefault(size, []).append((term, values))
        return nt == task["start"] and values == target

    def terminal(p):
        if isinstance(p, tuple):
            return tuple(p[1] for _ in rows)
        if type(p) is int:
            return tuple(p for _ in rows)
        if p in ("true", "false"):
            return tuple(p == "true" for _ in rows)
        if p in names:
            return tuple(row[names.index(p)] for row in rows)
        return None

    budget = BUDGET
    for size in range(1, MAX_SIZE + 1):
        for nt, productions in grammar.items():
            for p in productions:
                if isinstance(p, str) and p in grammar:
                    continue  # aliases (Start -> ntString) are copied below, after this size is complete
                if not isinstance(p, list):
                    if size == 1 and add(nt, 1, p, terminal(p)):
                        return _text(p)
                    continue
                op, children = p[0], p[1:]
                if not all(isinstance(c, str) and c in grammar for c in children):
                    continue
                for split in itertools.product(range(1, size), repeat=len(children)):
                    if sum(split) != size - 1:
                        continue
                    pools = [bank[c].get(k, []) for c, k in zip(children, split)]
                    for combo in itertools.product(*pools):
                        budget -= 1
                        if budget < 0:
                            return names[0]
                        values = tuple(_apply(op, [v[i] for _, v in combo]) for i in range(len(rows)))
                        if any(v is None for v in values):
                            continue
                        if add(nt, size, [op] + [t for t, _ in combo], values):
                            return _text([op] + [t for t, _ in combo])
        for nt, productions in grammar.items():
            for p in productions:
                if isinstance(p, str) and p in grammar:
                    for term, values in list(bank[p].get(size, [])):
                        if add(nt, size, term, values):
                            return _text(term)
    return names[0]
# EVOLVE-BLOCK-END
