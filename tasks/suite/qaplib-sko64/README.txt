# Quadratic assignment, QAPLIB sko64

**Task id:** `qaplib-sko64`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `qap` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Quadratic assignment problem, QAPLIB instance sko64: given two 64 x 64 integer matrices a and b, find a permutation p of 0..63 that minimises the cost, the sum over all i and j of a[i, j] * b[p[i], p[j]]. In the usual reading one matrix holds the flows between facilities and the other the distances between locations, and p assigns the one to the other; the cost formula is what is scored.

## Candidate interface

solve(a, b) -> permutation p of 0..n-1 (a list of n integers, each exactly once). a and b are the first and second matrix of the QAPLIB file (int64 arrays of shape (n, n), passed as copies). Score: the exact integer cost, the sum over all i and j of a[i, j] * b[p[i], p[j]], minimized.

## The instance

Skorin-Kapov instance with 64 facilities on a 8 x 8 rectangular grid: the distance matrix holds Manhattan distances between grid cells and the flow entries are pseudorandom integers.

## Why it matters

**Problem:** Find a minimum-cost Koopmans-Beckmann quadratic assignment of 64 facilities to 64 locations for the QAPLIB instance sko64, which is not solved to optimality.

The quadratic assignment problem models facility and hospital layout, keyboard design, backboard wiring and the generation of grey patterns. It is NP-hard and hard in practice: QAPLIB instances above about 30 facilities are mostly unsolved, and the best known solutions of the larger ones have stood for many years, so a cheaper permutation is a new best known solution of a standard open instance.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Assignment Problems and the Location of Economic Activities](https://doi.org/10.2307/1907742): Formulate the assignment of indivisible economic activities to locations, introducing the quadratic assignment problem. The task's objective is the Koopmans-Beckmann quadratic assignment cost of the QAPLIB instance sko64.
- [Tabu Search Applied to the Quadratic Assignment Problem](https://doi.org/10.1287/ijoc.2.1.33): Apply tabu search to the quadratic assignment problem. QAPLIB credits the Skorin-Kapov instance set (rectangular-grid distances, pseudorandom flows) to this paper, and lists strict tabu search from it as a heuristic; sko64 belongs to that set.
- [Robust taboo search for the quadratic assignment problem](https://doi.org/10.1016/S0167-8191(05)80147-4): Introduce the robust tabu search heuristic for the quadratic assignment problem and report results on test instances. QAPLIB credits the best known solution of sko64 to robust tabu search (Ro-TS), which this paper introduces; the task asks to improve it.

## Record

Best known 48498, best lower bound 47017 (QAPLIB).

**Record validation:** reproduced. published permutation (direct, 1-based) evaluated by our verifier = 48498; equals the value in the .sln file and in the registry; the maintained QAPLIB table (copy of 2026-05-09) lists best known 48498 and lower bound 47017.

## Data and requirements

sko64.dat from the QAPLIB deposit (Edinburgh DataShare, doi 10.7488/ds/3428, CC BY 4.0)

Requires: `py:numpy`, `file:data/arnew/qap/sko64.dat`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family qap` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Same evaluator contract as the existing QAPLIB tasks tai256c, tho150 and wil100 in autoresearch-tasks.
- The record construction is public, so a candidate can embed the QAPLIB solution file; compare only genuinely searched results.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
