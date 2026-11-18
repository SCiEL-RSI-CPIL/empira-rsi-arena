# Quadratic assignment, QAPLIB sko100a

**Task id:** `qaplib-sko100a`\
**Status:** ready\
**Direction:** minimize\
**Family:** `qap` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Quadratic assignment problem, QAPLIB instance sko100a: given two 100 x 100 integer matrices a and b, find a permutation p of 0..99 that minimises the cost, the sum over all i and j of a[i, j] * b[p[i], p[j]]. In the usual reading one matrix holds the flows between facilities and the other the distances between locations, and p assigns the one to the other; the cost formula is what is scored.

## Candidate interface

solve(a, b) -> permutation p of 0..n-1 (a list of n integers, each exactly once). a and b are the first and second matrix of the QAPLIB file (int64 arrays of shape (n, n), passed as copies). Score: the exact integer cost, the sum over all i and j of a[i, j] * b[p[i], p[j]], minimized.

## The instance

Skorin-Kapov instance with 100 facilities on a 10 x 10 rectangular grid: the distance matrix holds Manhattan distances between grid cells and the flow entries are pseudorandom integers. It is one of six pseudorandom flow matrices on the 10 x 10 grid (a to f).

## Why it matters

**Problem:** Find a minimum-cost Koopmans-Beckmann quadratic assignment of 100 facilities to 100 locations for the QAPLIB instance sko100a, which is not solved to optimality.

The quadratic assignment problem models facility and hospital layout, keyboard design, backboard wiring and the generation of grey patterns. It is NP-hard and hard in practice: QAPLIB instances above about 30 facilities are mostly unsolved, and the best known solutions of the larger ones have stood for many years, so a cheaper permutation is a new best known solution of a standard open instance.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Assignment Problems and the Location of Economic Activities](https://doi.org/10.2307/1907742): Formulate the assignment of indivisible economic activities to locations, introducing the quadratic assignment problem. The task's objective is the Koopmans-Beckmann quadratic assignment cost of the QAPLIB instance sko100a.
- [Tabu Search Applied to the Quadratic Assignment Problem](https://doi.org/10.1287/ijoc.2.1.33): Apply tabu search to the quadratic assignment problem. QAPLIB credits the Skorin-Kapov instance set (rectangular-grid distances, pseudorandom flows) to this paper, and lists strict tabu search from it as a heuristic; sko100a belongs to that set.
- [Genetic hybrids for the quadratic assignment problem](https://doi.org/10.1090/dimacs/016/08): Design genetic algorithms hybridised with local search that find better solutions of QAPLIB instances. QAPLIB credits the best known solution of sko100a to this genetic hybrid (GEN); the task asks to improve it.

## Record

Best known 152002, best lower bound 143846 (QAPLIB).

**Record validation:** reproduced. published permutation (direct, 1-based) evaluated by our verifier = 152002; equals the value in the .sln file and in the registry; the maintained QAPLIB table (copy of 2026-05-09) lists best known 152002 and lower bound 143846.

## Data and requirements

sko100a.dat from the QAPLIB deposit (Edinburgh DataShare, doi 10.7488/ds/3428, CC BY 4.0)

Requires: `py:numpy`, `file:data/qap/sko100a.dat`. Run `python3 scripts/prepare_new_data.py --family qap` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Same evaluator contract as the existing QAPLIB tasks tai256c, tho150 and wil100 in autoresearch-tasks.
- The record construction is public, so a candidate can embed the QAPLIB solution file; compare only genuinely searched results.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
