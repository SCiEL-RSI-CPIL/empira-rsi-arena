# Quadratic assignment, QAPLIB tai100a

**Task id:** `qaplib-tai100a`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `qap` (suite/_lib/arnew)\
**Areas:** optimization

## Task

Quadratic assignment problem, QAPLIB instance tai100a: given two 100 x 100 integer matrices a and b, find a permutation p of 0..99 that minimises the cost, the sum over all i and j of a[i, j] * b[p[i], p[j]]. In the usual reading one matrix holds the flows between facilities and the other the distances between locations, and p assigns the one to the other; the cost formula is what is scored.

## Candidate interface

solve(a, b) -> permutation p of 0..n-1 (a list of n integers, each exactly once). a and b are the first and second matrix of the QAPLIB file (int64 arrays of shape (n, n), passed as copies). Score: the exact integer cost, the sum over all i and j of a[i, j] * b[p[i], p[j]], minimized.

## The instance

Taillard uniformly generated random instance with 100 facilities: both matrices are symmetric with integer entries drawn uniformly from 0 to 99.

## Why it matters

**Problem:** Find a minimum-cost Koopmans-Beckmann quadratic assignment of 100 facilities to 100 locations for the QAPLIB instance tai100a, which is not solved to optimality.

The quadratic assignment problem models facility and hospital layout, keyboard design, backboard wiring and the generation of grey patterns. It is NP-hard and hard in practice: QAPLIB instances above about 30 facilities are mostly unsolved, and the best known solutions of the larger ones have stood for many years, so a cheaper permutation is a new best known solution of a standard open instance.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Assignment Problems and the Location of Economic Activities](https://doi.org/10.2307/1907742): Formulate the assignment of indivisible economic activities to locations, introducing the quadratic assignment problem. The task's objective is the Koopmans-Beckmann quadratic assignment cost of the QAPLIB instance tai100a.
- [Robust taboo search for the quadratic assignment problem](https://doi.org/10.1016/S0167-8191(05)80147-4): Introduce the robust tabu search heuristic for the quadratic assignment problem and report results on test instances. QAPLIB credits the uniformly generated Taillard 'a' instances, including tai100a, to this paper; the best known solution of tai100a is credited to a later method (see the other paper).

## Record

Best known 21040996, from Sergienko, Shylo, Roshchyn, Shylo, Boyarchuk and Moroz 2024 (Cybernetics and Computer Technologies 2024(3):25-33, doi:10.34229/2707-451X.24.3.3), which prints the permutation; earlier values were 21,044,752 (V. Shylo 2017, QAPLIB) and 21,043,560 (Misevicius 2019, Memetic Computing 11:331-332); this task's verifier replayed the permutation printed there. QAPLIB still lists the value of 2017, whose permutation the verifier also replayed. Best lower bound 17853840 (QAPLIB).

**Record validation:** reproduced. the permutation printed in Sergienko, Shylo, Roshchyn, Shylo, Boyarchuk and Moroz 2024 (Cybernetics and Computer Technologies 2024(3):25-33, doi:10.34229/2707-451X.24.3.3), which prints the permutation; earlier values were 21,044,752 (V. Shylo 2017, QAPLIB) and 21,043,560 (Misevicius 2019, Memetic Computing 11:331-332), evaluated by our verifier = 21040996; the numbers of the permutation were found in this order in the pinned PDF text (sources/sergienko_2024_tai100a.pdf) together with the value 21040996; the maintained QAPLIB table (copy of 2026-05-09) still lists 21044752, which is 3756 more, and lower bound 17853840; QAPLIB's own solution file evaluates to 21044752. Note: No newer tai100a value was found: the 2026 paper that an earlier version of this review cited as holding a better one (Sergienko, Shylo, Roshchyn and Boyarchuk, Cybernetics and Systems Analysis 62(4):578-587, doi 10.1007/s10559-026-00893-0) is titled 'Applying Algorithm Unions to Specific Classes of Boolean Programming Problems' (Crossref), so it is not about this instance.

## Data and requirements

tai100a.dat from the QAPLIB deposit (Edinburgh DataShare, doi 10.7488/ds/3428, CC BY 4.0)

Requires: `py:numpy`, `file:data/qap/tai100a.dat`. Run `python3 scripts/prepare_new_data.py --family qap` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Same evaluator contract as the existing QAPLIB tasks tai256c, tho150 and wil100 in autoresearch-tasks.
- The record construction is public, so a candidate can embed the QAPLIB solution file; compare only genuinely searched results.
- No newer tai100a value was found: the 2026 paper that an earlier version of this review cited as holding a better one (Sergienko, Shylo, Roshchyn and Boyarchuk, Cybernetics and Systems Analysis 62(4):578-587, doi 10.1007/s10559-026-00893-0) is titled 'Applying Algorithm Unions to Specific Classes of Boolean Programming Problems' (Crossref), so it is not about this instance.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
