# Covering design C(19,8,5)

**Source collection:** [Open combinatorial design, coding and assignment problems with published records: source-based research tasks](../../papers/science-frontier_combinatorics.md)\
**Task id:** `science-frontier-combinatorics-covering-design-19-8-5`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Choose as few 8-element blocks of 19 points as possible so that every 5-element subset lies in some block. The best known covering has 335 blocks (La Jolla Covering Repository) and the repository's lower bound is 264, so the covering number C(19,8,5) is open.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

construct(v, k, t) -> list of blocks; the evaluator passes this task's parameters v, k and t. Each block is a list of k distinct integers in 0..v-1 (a k-subset of the v points) and blocks must be distinct; at most C(v, t) blocks. Valid when every t-subset of {0, ..., v-1} lies in at least one block (checked exhaustively). Score: the number of blocks, minimized; it is an upper bound on the covering number C(v, k, t).

## State of the art

The best known (19,8,5) covering has 335 blocks, submitted by Marco Meoli on 2015-10-31 using his own search programs ('private tools'); earlier records were 419 (the repository's first entry, from the 1995 tables of Gordon, Kuperberg and Patashnik), then a sequence of improvements by R. Belic, D. Pree, F. Rodriguez Martin, J. de Heer, S. Muir and dynamic-programming constructions down to 346 (J. de Heer and S. Muir, 2008). The repository's lower bound is 264, so C(19,8,5) is open.

Reference value: best known value `335` (reviewed 2026-10-12).

- [La Jolla Covering Repository (Zenodo dataset, version 1.2)](https://doi.org/10.5281/zenodo.10779736), D. M. Gordon (2026)
- [New constructions for covering designs](https://doi.org/10.1002/jcd.3180030404), D. M. Gordon, G. Kuperberg, O. Patashnik (1995)

Parameters: `{"v": 19, "k": 8, "t": 5, "published": 335}`

## Labels

Scientific contribution: `open_problem` (tier 1); artifact `construction`, domain `mathematics`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `mathematics`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Determine the covering number C(19,8,5): the least number of 8-element blocks of a 19-point set such that every 5-element subset lies in a block.

Why it matters: Covering designs are central objects of combinatorial design theory (Turan-type covering problems, lottery and wheel systems, group testing and software testing). Exact covering numbers are known only for small parameters, and constructions for the open cases have been tabulated and improved by a long-running research effort, so a smaller covering is a new upper bound on a stated extremal function.

- [New constructions for covering designs](https://doi.org/10.1002/jcd.3180030404): Main research objective: Give new general constructions of covering designs and tabulate improved upper bounds on covering numbers C(v,k,t) for small parameters. Task connection: The task asks for a (19,8,5) covering with fewer than 335 blocks, the best known value in the repository that grew out of this paper's tables.
- [Asymptotically optimal covering designs](https://doi.org/10.1006/jcta.1996.0077): Main research objective: Determine the asymptotic behaviour of covering numbers C(v,k,t) with explicit and probabilistic constructions. Task connection: Studies the same covering numbers; the task targets one small open case.

Both papers have covering numbers as their main research objective, and the evaluator scores exactly a (v,k,t) covering by its number of blocks.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Independent exact verifier: every t-subset of every block is ranked in the combinatorial number system and all C(v, t) t-subsets must be marked. The published record construction for every task was scored with this evaluator and reproduces the published value exactly (validation/human-results/frontier_combinatorics.json). The record coverings come from the La Jolla Covering Repository (D. Gordon, Zenodo, CC BY 4.0). Unit tests check small coverings with known covering numbers and reject uncovered, repeated or out-of-range blocks.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`

## Sources

- benchmark: https://doi.org/10.5281/zenodo.10779736 (La Jolla Covering Repository (D. Gordon), Zenodo version 1.2 (2026-04-24, CC BY 4.0): coverdata.json entry C(19,8,5) lists size 335, lower bound 264 and the improvement history; covers.json holds the 335-block covering submitted by Marco Meoli on 2015-10-31.)
- paper: https://doi.org/10.1002/jcd.3180030404 (Gordon, Kuperberg and Patashnik (1995), New constructions for covering designs: covering numbers C(v,k,t), constructions and the tables the repository grew from.)
- benchmark: https://coveringrepository.com (Successor repository that tracks improvements after the La Jolla freeze (March 2026); its latest-improvement listings showed no entry for these parameters on 2026-10-12.)

## Limitations

- The La Jolla repository was frozen in March 2026; later improvements are tracked at coveringrepository.com, whose latest-improvement listings were checked on 2026-10-12 without an entry for C(19,8,5), but whose full table could not be queried (HTTP 403), so a recent unlisted improvement cannot be excluded.
- The lower bound 264 is the repository's; the evaluator reports the weaker Schoenheim bound for orientation only and does not certify optimality.
- The greedy seed has 480 blocks.
- The record construction is public, so a candidate can reproduce the record by embedding it; the score measures the returned object, not the search that produced it, and only a strictly smaller score than the best known value is a new result.
- Candidate Python runs inside the evaluator process as trusted research code; the evaluator recomputes every property of the returned object exactly and trusts nothing the candidate reports.
