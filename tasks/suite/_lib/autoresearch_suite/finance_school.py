"""School choice with coarse priorities: efficient weakly stable matchings (Erdil and Ergin 2008).

Students rank up to L schools; schools have capacities and coarse priority
classes (walk-zone tiers and siblings) with many ties. The usual mechanism
breaks ties with one lottery and runs student-proposing deferred acceptance
(DA-STB), which can be inefficient. A candidate proposes a matching; it must be
weakly stable for the coarse priorities (no student prefers a school that has a
free seat or admits a student of strictly lower priority). The score is the
share of the possible reduction in total preference rank, from DA-STB to the
proven rank-minimal weakly stable matching, that the candidate achieves.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib

import numpy as np

from .runtime import CandidateError

# (students, schools, list length, seed, walk-zone tiers)
SIZES = [(150, 6, 5, 11, 1), (200, 8, 6, 12, 1), (200, 8, 6, 13, 3), (250, 10, 6, 14, 1), (300, 10, 6, 15, 2), (300, 12, 6, 16, 1)]
INSTANCE_SHA256 = {
    0: "5a9a2625628af3f243b1f4ab1aac71d93380160f4423c271b87937bf56121643",
    1: "2ce3399745951c110412d197ab0b3b7c8470dc43a963b4e38a363543abe28066",
    2: "5165026549b153c04f885f3669155834a5fa9d30078d165670fd218b8bbbe5f4",
    3: "1f8c33614d1c58218805d1d0ee532d38aba3a52e36f328a6898eeb052ebf0b5c",
    4: "4925d9777e3d555cff2a8a8d88ddf2c51a6bbf7a0f0405233245ad3ef614ccba",
    5: "83ca22d3c3243d08a9ce66762d87ad223ae26d7aa896a8a20de6f9befa0bec22",
}
DA_RANK = {0: 324, 1: 403, 2: 414, 3: 474, 4: 768, 5: 533}  # total rank of DA with the instance's single lottery
OPT_RANK = {0: 252, 1: 324, 2: 337, 3: 391, 4: 609, 5: 406}  # proven minimum total rank over weakly stable matchings (HiGHS, gap 0)


def generate(students, schools, length, seed, tiers):
    rng = np.random.RandomState(seed)  # legacy generator: stream frozen across NumPy versions
    homes, sites = rng.random_sample((students, 2)), rng.random_sample((schools, 2))
    quality = rng.normal(0.0, 1.0, schools)
    distance = np.sqrt(((homes[:, None, :] - sites[None, :, :]) ** 2).sum(axis=-1))
    utility = quality[None, :] - 3.0 * distance + rng.gumbel(0.0, 1.0, (students, schools))
    preferences = np.argsort(-utility, axis=1, kind="stable")[:, :length]
    sibling = rng.random_sample((students, schools)) < 0.05
    zone = np.zeros((students, schools), dtype=np.int64)
    for tier in range(tiers):  # nested walk zones: closer students get higher priority
        zone += distance < 0.25 * (tier + 1) / tiers
    priorities = 1 + zone + (tiers + 1) * sibling
    capacities = np.full(schools, students // schools)
    capacities[: students - int(capacities.sum())] += 1
    lottery = rng.permutation(students)
    return {"students": students, "schools": schools, "list_length": length, "preferences": preferences.astype(np.int64),
            "priorities": priorities.astype(np.int64), "capacities": capacities.astype(np.int64), "lottery": lottery.astype(np.int64)}


def digest(problem):
    parts = [problem[k].tobytes() for k in ("preferences", "priorities", "capacities", "lottery")]
    return hashlib.sha256(b"".join(parts)).hexdigest()


@lru_cache(maxsize=None)
def instance(index):
    problem = generate(*SIZES[index])
    if digest(problem) != INSTANCE_SHA256[index]:
        raise RuntimeError(f"school-choice instance {index} changed; the NumPy random stream differs")
    for value in problem.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return problem


def deferred_acceptance(problem, lottery):
    """Student-proposing DA with priorities refined by one lottery (lottery[0] wins all ties)."""
    S, L = problem["preferences"].shape
    position = np.empty(S, dtype=np.int64)
    position[lottery] = np.arange(S)
    held = [[] for _ in range(problem["schools"])]
    next_choice, free = np.zeros(S, dtype=np.int64), list(range(S))
    while free:
        s = free.pop()
        if next_choice[s] >= L:
            continue
        c = int(problem["preferences"][s, next_choice[s]])
        next_choice[s] += 1
        held[c].append(s)
        if len(held[c]) > problem["capacities"][c]:
            worst = min(held[c], key=lambda t: (problem["priorities"][t, c], -position[t]))
            held[c].remove(worst)
            free.append(worst)
    match = np.full(S, -1, dtype=np.int64)
    for c, admitted in enumerate(held):
        match[admitted] = c
    return match


def ranks(problem, match):
    """Rank (1 = first choice) of each student's school; unassigned students count list_length + 1."""
    L = problem["list_length"]
    out = np.full(problem["students"], L + 1, dtype=np.int64)
    for s, c in enumerate(match):
        if c >= 0:
            out[s] = 1 + int(np.nonzero(problem["preferences"][s] == c)[0][0])
    return out


def validate(problem, raw):
    S, C = problem["students"], problem["schools"]
    try:
        match = np.asarray(raw)
    except Exception as error:  # noqa: BLE001 - any conversion failure is a contract violation
        raise CandidateError(f"assignment is not array-like: {error}") from None
    if match.shape != (S,) or not np.issubdtype(match.dtype, np.integer):
        raise CandidateError(f"assignment must be {S} integer school indices (-1 = unassigned)")
    match = match.astype(np.int64)
    if ((match < -1) | (match >= C)).any():
        raise CandidateError("school index out of range")
    for s, c in enumerate(match):
        if c >= 0 and c not in problem["preferences"][s]:
            raise CandidateError(f"student {s} is assigned to school {c}, which is not on her list")
    if (np.bincount(match[match >= 0], minlength=C) > problem["capacities"]).any():
        raise CandidateError("a school exceeds its capacity")
    return match


def blocking_pair(problem, match):
    """First (student, school) pair violating weak stability, or None."""
    C = problem["schools"]
    load = np.bincount(match[match >= 0], minlength=C)
    weakest = np.full(C, np.iinfo(np.int64).max)
    for s, c in enumerate(match):
        if c >= 0:
            weakest[c] = min(weakest[c], problem["priorities"][s, c])
    for s in range(problem["students"]):
        for c in problem["preferences"][s]:
            if c == match[s]:
                break
            if load[c] < problem["capacities"][c] or problem["priorities"][s, c] > weakest[c]:
                return s, int(c)
    return None


def school_score(assign):
    """Share of the DA-STB-to-optimum total-rank reduction achieved (maximize; 0 = DA-STB, 1 = optimal)."""
    if not callable(assign):
        raise CandidateError("assign_students must be callable")
    rows, gained, possible = [], 0, 0
    for index in range(len(SIZES)):
        problem = instance(index)
        given = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in problem.items()}
        match = validate(problem, assign(given))
        block = blocking_pair(problem, match)
        if block is not None:
            raise CandidateError(f"instance {index}: student {block[0]} and school {block[1]} block the matching")
        total = int(ranks(problem, match).sum())
        gained += DA_RANK[index] - total
        possible += DA_RANK[index] - OPT_RANK[index]
        rows.append({"students": problem["students"], "total_rank": total, "da_stb_rank": DA_RANK[index],
                     "optimal_rank": OPT_RANK[index], "unassigned": int((match < 0).sum())})
    return gained / possible, {"instances": rows}


def optimal_matching(problem, time_limit=600):
    """Rank-minimal weakly stable matching by integer programming (used offline and in tests)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    pref, prio, cap = problem["preferences"], problem["priorities"], problem["capacities"]
    S, L = pref.shape
    index = {(s, int(pref[s, k])): s * L + k for s in range(S) for k in range(L)}
    applicants = [[s for s in range(S) if c in pref[s]] for c in range(problem["schools"])]
    rows, cols, vals, lower, upper = [], [], [], [], []

    def add(entries, lo, hi):
        for col, val in entries:
            rows.append(len(lower))
            cols.append(col)
            vals.append(val)
        lower.append(lo)
        upper.append(hi)
    for s in range(S):
        add([(s * L + k, 1.0) for k in range(L)], -np.inf, 1.0)
    for c, group in enumerate(applicants):
        add([(index[(s, c)], 1.0) for s in group], -np.inf, float(cap[c]))
    for s in range(S):  # weak stability: s at c or better, or c full of students with priority >= s's
        for k in range(L):
            c = int(pref[s, k])
            entries = [(s * L + j, float(cap[c])) for j in range(k + 1)]
            entries += [(index[(t, c)], 1.0) for t in applicants[c] if t != s and prio[t, c] >= prio[s, c]]
            add(entries, float(cap[c]), np.inf)
    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lower), S * L)).tocsr()
    cost = np.tile(np.arange(1, L + 1, dtype=float) - (L + 1), S)
    result = milp(cost, constraints=LinearConstraint(matrix, lower, upper), integrality=np.ones(S * L), bounds=Bounds(0, 1),
                  options={"mip_rel_gap": 0, "time_limit": time_limit})
    if result.status != 0:
        raise RuntimeError(f"school-choice reference solve failed: {result.message}")
    chosen = result.x.reshape(S, L) > 0.5
    match = np.where(chosen.any(axis=1), pref[np.arange(S), chosen.argmax(axis=1)], -1)
    return match, int(ranks(problem, match).sum())
