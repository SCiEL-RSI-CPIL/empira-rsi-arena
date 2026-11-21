"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def assign_students(problem):
    """DA-STB seed: student-proposing deferred acceptance with the given single tie-breaking lottery."""
    pref, prio, cap = problem["preferences"], problem["priorities"], problem["capacities"]
    S, L = pref.shape
    position = np.empty(S, dtype=int)
    position[problem["lottery"]] = np.arange(S)
    held = [[] for _ in range(problem["schools"])]
    next_choice, free = [0] * S, list(range(S))
    while free:
        s = free.pop()
        if next_choice[s] >= L:
            continue
        c = int(pref[s, next_choice[s]])
        next_choice[s] += 1
        held[c].append(s)
        if len(held[c]) > cap[c]:
            worst = min(held[c], key=lambda t: (prio[t, c], -position[t]))
            held[c].remove(worst)
            free.append(worst)
    match = np.full(S, -1)
    for c, admitted in enumerate(held):
        match[admitted] = c
    return match
# EVOLVE-BLOCK-END
