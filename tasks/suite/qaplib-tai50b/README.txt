# Quadratic assignment, QAPLIB tai50b

**Task id:** `qaplib-tai50b`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `qap` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Quadratic assignment problem, QAPLIB instance tai50b: given two 50 x 50 integer matrices a and b, find a permutation p of 0..49 that minimises the cost, the sum over all i and j of a[i, j] * b[p[i], p[j]]. In the usual reading one matrix holds the flows between facilities and the other the distances between locations, and p assigns the one to the other; the cost formula is what is scored.

## Candidate interface

solve(a, b) -> permutation p of 0..n-1 (a list of n integers, each exactly once). a and b are the first and second matrix of the QAPLIB file (int64 arrays of shape (n, n), passed as copies). Score: the exact integer cost, the sum over all i and j of a[i, j] * b[p[i], p[j]], minimized.

## The instance

Taillard random asymmetric instance with 50 facilities, generated to resemble real-life instances: a symmetric distance-type matrix of Euclidean flavour and a sparse, non-uniform, asymmetric flow matrix (QAPLIB: 'asymmetric and randomly generated').

## Why it matters

**Problem:** Find a minimum-cost Koopmans-Beckmann quadratic assignment of 50 facilities to 50 locations for the QAPLIB instance tai50b, which is not solved to optimality.

The quadratic assignment problem models facility and hospital layout, keyboard design, backboard wiring and the generation of grey patterns. It is NP-hard and hard in practice: QAPLIB instances above about 30 facilities are mostly unsolved, and the best known solutions of the larger ones have stood for many years, so a cheaper permutation is a new best known solution of a standard open instance.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Assignment Problems and the Location of Economic Activities](https://doi.org/10.2307/1907742): Formulate the assignment of indivisible economic activities to locations, introducing the quadratic assignment problem. The task's objective is the Koopmans-Beckmann quadratic assignment cost of the QAPLIB instance tai50b.
- [Robust taboo search for the quadratic assignment problem](https://doi.org/10.1016/S0167-8191(05)80147-4): Introduce the robust tabu search heuristic for the quadratic assignment problem and report results on test instances. QAPLIB credits the best known solution of tai50b to robust tabu search (Ro-TS), which this paper introduces; tai50b is an asymmetric random Taillard 'b' instance.
- [Comparison of iterative searches for the quadratic assignment problem](https://doi.org/10.1016/0966-8349(95)00008-6): Compare iterative search methods for the quadratic assignment problem on uniform random, structured and real-life instances, and describe a generator of more realistic random instances. QAPLIB cites a 1994 Taillard report for the Taillard 'b' instances; this paper, which describes the generator of random instances with Euclidean-type distances and non-uniform flows, appears to be its published version.

## Record

Best known 458821517, best lower bound 431090700 (QAPLIB).

**Record validation:** reproduced. published permutation (direct, 1-based) evaluated by our verifier = 458821517; equals the value in the .sln file and in the registry; the maintained QAPLIB table (copy of 2026-05-09) lists best known 458821517 and lower bound 431090700.

## Data and requirements

tai50b.dat from the QAPLIB deposit (Edinburgh DataShare, doi 10.7488/ds/3428, CC BY 4.0)

Requires: `py:numpy`, `file:data/arnew/qap/tai50b.dat`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family qap` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Same evaluator contract as the existing QAPLIB tasks tai256c, tho150 and wil100 in autoresearch-tasks.
- The record construction is public, so a candidate can embed the QAPLIB solution file; compare only genuinely searched results.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
