# Max-cut, G-set G54

**Task id:** `gset-maxcut-g54`\
**Status:** requires_setup\
**Direction:** maximize\
**Family:** `maxcut` (suite/_lib/arnew)\
**Areas:** optimization, physics, theoretical computer science, systems and hardware

## Task

Partition the 1000 vertices of the G-set graph G54 into two sides so that the total weight of the edges whose endpoints are on different sides is as large as possible (max-cut). The graph is undirected and has 5916 edges, each given once; every edge has weight 1.

## Candidate interface

solve(n, edges) -> sequence of n integers, each 0 or 1: the side of every vertex. edges is an int64 array with one row (i, j, w) per edge, vertices numbered from 0. Score: total weight of the edges whose endpoints are on different sides, maximized.

## The instance

G-set graph G54: a sparse planar-type graph of the generator's 'union of planar graphs' class (edge count below twice the planar maximum 3n-6; a few high-degree hubs; 1000 vertices, 5916 edges, all weights 1); the weight of a cut is the sum of the weights of edges whose endpoints lie on different sides.

## Why it matters

**Problem:** Find a partition of the vertices of the G-set graph G54 into two sides that maximises the total weight of the edges between the sides (max-cut).

Max-cut asks for the heaviest cut of a weighted graph and is NP-hard. It is equivalent to finding a ground state of an Ising spin glass without external field, and it models via minimisation in circuit layout; its semidefinite relaxation gave one of the best-known approximation algorithms. The G-set graphs (800 to 20,000 vertices) have been a standard test bed for heuristics and Ising machines for about 25 years, and for the large sparse graphs a better cut is a better solution of an open or unproven instance.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [An Application of Combinatorial Optimization to Statistical Physics and Circuit Layout Design](https://doi.org/10.1287/opre.36.3.493): Reduce spin-glass ground states and via minimisation in circuit layout to max-cut and solve the resulting instances with a cutting-plane algorithm. States why max-cut matters (spin-glass ground states, layout); the task scores the max-cut weight of the G-set graph G54, with a heuristic instead of a cutting-plane method.
- [Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming](https://doi.org/10.1145/227683.227684): Give a randomised approximation algorithm for max-cut based on a semidefinite programming relaxation. Introduces the semidefinite relaxation that bounds and approximates the quantity the task maximises; the task has no guarantee requirement.
- [Rank-Two Relaxation Heuristics for MAX-CUT and Other Binary Quadratic Programs](https://doi.org/10.1137/S1052623400382467): Develop fast rank-two relaxation heuristics for large max-cut instances and compare them with semidefinite programming codes. A heuristic for large max-cut graphs of the kind the task poses; the task asks for a better cut on one fixed graph. The paper's use of the G-set files was not read.
- [Breakout Local Search for the Max-Cutproblem](https://doi.org/10.1016/j.engappai.2012.09.001): Design a breakout local search heuristic for max-cut and report improved cuts on benchmark graphs. A heuristic whose main objective is the cut weight the task scores on G-set graphs. The title is as Crossref spells it; its tables were not read (publisher page blocked).

## Record

The best-known cut was not read for this instance; check the G-set literature before claiming a record.

**Record validation:** no record claimed. no best-known cut stored for this instance.

## Data and requirements

G-set file G54 from https://web.stanford.edu/~yyye/yyye/Gset/G54

Requires: `py:numpy`, `file:data/arnew/gset/G54`. Run `python3 scripts/new_open_problems/prepare_new_data.py --family gset` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Specialised solvers (breakout local search, simulated bifurcation, GES-PR) are strong on this set.
- The data file digest is not pinned until scripts/new_open_problems/prepare_new_data.py has fetched it.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
