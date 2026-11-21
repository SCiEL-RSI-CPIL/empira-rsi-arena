# Quadratic assignment, QAPLIB tai35a

**Task id:** `qaplib-tai35a`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `qap` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Quadratic assignment problem, QAPLIB instance tai35a: given two 35 x 35 integer matrices a and b, find a permutation p of 0..34 that minimises the cost, the sum over all i and j of a[i, j] * b[p[i], p[j]]. In the usual reading one matrix holds the flows between facilities and the other the distances between locations, and p assigns the one to the other; the cost formula is what is scored.

## Candidate interface

solve(a, b) -> permutation p of 0..n-1 (a list of n integers, each exactly once). a and b are the first and second matrix of the QAPLIB file (int64 arrays of shape (n, n), passed as copies). Score: the exact integer cost, the sum over all i and j of a[i, j] * b[p[i], p[j]], minimized.

## The instance

Taillard uniformly generated random instance with 35 facilities: both matrices are symmetric with integer entries drawn uniformly from 0 to 99.

## Why it matters

**Problem:** Find a minimum-cost Koopmans-Beckmann quadratic assignment of 35 facilities to 35 locations for the QAPLIB instance tai35a, which is not solved to optimality.

The quadratic assignment problem models facility and hospital layout, keyboard design, backboard wiring and the generation of grey patterns. It is NP-hard and hard in practice: QAPLIB instances above about 30 facilities are mostly unsolved, and the best known solutions of the larger ones have stood for many years, so a cheaper permutation is a new best known solution of a standard open instance.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Assignment Problems and the Location of Economic Activities](https://doi.org/10.2307/1907742): Formulate the assignment of indivisible economic activities to locations, introducing the quadratic assignment problem. The task's objective is the Koopmans-Beckmann quadratic assignment cost of the QAPLIB instance tai35a.
- [Robust taboo search for the quadratic assignment problem](https://doi.org/10.1016/S0167-8191(05)80147-4): Introduce the robust tabu search heuristic for the quadratic assignment problem and report results on test instances. QAPLIB credits the uniformly generated Taillard 'a' instances to this paper and lists robust tabu search (Ro-TS) from it as a heuristic; tai35a is one of them.

## Record

Best known 2422002, best lower bound 2216627 (QAPLIB).

**Record validation:** reproduced. published permutation (direct, 1-based) evaluated by our verifier = 2422002; equals the value in the .sln file and in the registry; the maintained QAPLIB table (copy of 2026-05-09) lists best known 2422002 and lower bound 2216627.

## Data and requirements

tai35a.dat from the QAPLIB deposit (Edinburgh DataShare, doi 10.7488/ds/3428, CC BY 4.0)

Requires: `py:numpy`, `file:data/qap/tai35a.dat`. Run `python3 scripts/prepare_new_data.py --family qap` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Same evaluator contract as the existing QAPLIB tasks tai256c, tho150 and wil100 in autoresearch-tasks.
- The record construction is public, so a candidate can embed the QAPLIB solution file; compare only genuinely searched results.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
