# Small 5-regular graph of girth 9 (cage problem n(5,9))

**Source collection:** [Open combinatorial design, coding and assignment problems with published records: source-based research tasks](../../papers/science-frontier_combinatorics.md)\
**Task id:** `science-frontier-combinatorics-cage-5-9`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Construct a 5-regular graph with no cycle shorter than 9 on as few vertices as possible. The smallest known such graph has 1116 vertices (Exoo, Goedgebeur, Jooken, Stubbe and Van den Eede, 2026) and the best lower bound is 428.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct(k, g) -> (n, edges): the number of vertices n and a list of edges (i, j) on vertices 0..n-1; the evaluator passes this task's degree k and girth g. The graph must be simple (no loops or repeated edges), every vertex must have degree exactly k, and no cycle may be shorter than g (certified exactly with non-backtracking walks from every vertex). n must lie between the Moore bound and max_vertices. Score: n, minimized; it is an upper bound on the cage order n(k, g).

## State of the art

The smallest known 5-regular graph of girth 9 has 1116 vertices (Exoo, Goedgebeur, Jooken, Stubbe and Van den Eede, 2026), found by the authors' computer search over graph lifts with voltage group SmallGroup(279,1) = C31 semidirect C9; it improved 1152 (G. Exoo, 2025, a personal communication cited in the paper). The best lower bound is 428, so n(5,9) is open.

Reference value: best known value `1116` (reviewed 2026-10-12).

- [New small regular graphs of given girth: the cage problem and beyond](https://doi.org/10.37236/14834), G. Exoo, J. Goedgebeur, J. Jooken, L. Stubbe, T. Van den Eede (2026)
- [Dynamic cage survey](https://doi.org/10.37236/37), G. Exoo, R. Jajcay (2013)

Parameters: `{"k": 5, "g": 9, "max_vertices": 50000, "published": 1116}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Determine the cage order n(5,9), the least number of vertices of a 5-regular graph of girth 9.

Why it matters: The cage problem (Tutte 1947; Erdos and Sachs 1963) is a fundamental extremal problem of graph theory: it asks how small a regular graph with no short cycles can be. Graphs of high girth underlie LDPC codes and network constructions, the Moore bound is attained only in rare cases, and exact cage orders are known only for a few small parameters, so smaller graphs improve long-standing records.

- [New small regular graphs of given girth: the cage problem and beyond](https://doi.org/10.37236/14834): Main research objective: Find smaller k-regular graphs of given girth with lifts, tabu search, hill climbing and excision, improving upper bounds on cage orders. Task connection: The task asks for a 5-regular graph of girth 9 on fewer than 1116 vertices, the record of this paper (lower bound 428).
- [Dynamic cage survey](https://doi.org/10.37236/37): Main research objective: Survey the cage problem: known cages, record graphs, bounds and constructions. Task connection: Records the bounds and the record graphs of this cage case before the 2026 improvement.

Both papers have the cage problem as their main objective, and the evaluator certifies exactly k-regularity and girth of the returned graph and scores its order.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Independent exact verifier: regularity and simplicity are checked edge by edge and girth is certified with non-backtracking walks grown from all vertices. The published record construction for every task was scored with this evaluator and reproduces the published value exactly (validation/human-results/frontier_combinatorics.json). The record graphs of Exoo, Goedgebeur, Jooken, Stubbe and Van den Eede (2026) were read from their GitHub repository (GPL-3.0; kept in the cache, not redistributed). Unit tests compare the girth certificate with plain breadth-first search on random graphs and check the Petersen, Heawood and Tutte-Coxeter cages.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- paper: https://doi.org/10.37236/14834 (Exoo, Goedgebeur, Jooken, Stubbe and Van den Eede, Electron. J. Combin. 33(3) (2026) P3.48 (arXiv:2511.07247), Table 2: n(5,9) >= 428, previous upper bound 1152, new upper bound 1116; Figure 4 lists the base graph and the group SmallGroup(279,1) = C31 semidirect C9 of the lift.)
- repository: https://github.com/AGT-Kulak/smallRegGirthGraphs/tree/c2a1a4e334cd9f4dd31247f1ebfb6e18c004e4bc/genLifts/data/graphs/cage (Record graph n1116_k5_g9.g6 in graph6 format (GPL-3.0).)
- paper: https://doi.org/10.37236/37 (Exoo and Jajcay, Dynamic cage survey (Electron. J. Combin. DS16): definitions, Moore bound and record tables.)

## Limitations

- The evaluator accepts at most 50,000 vertices; the explicit Lazebnik-Ustimenko-Woldar seed has 6250 vertices.
- Connectivity is not required; a disconnected graph only has more vertices than its components.
- The record construction is public, so a candidate can reproduce the record by embedding it; the score measures the returned object, not the search that produced it, and only a strictly smaller score than the best known value is a new result.
- Candidate Python runs inside the evaluator process as trusted research code; the evaluator recomputes every property of the returned object exactly and trusts nothing the candidate reports.
