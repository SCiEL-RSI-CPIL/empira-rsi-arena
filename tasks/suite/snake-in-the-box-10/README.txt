# Snake-in-the-box in the 10-cube

**Task id:** `snake-in-the-box-10`\
**Status:** ready\
**Direction:** maximize\
**Family:** `snake` (suite/_lib/arnew)\
**Areas:** theoretical computer science, mathematics

## Task

Construct a snake that is as long as possible in the hypercube Q_10: the graph whose vertices are the integers 0 to 2^10 - 1, two of them adjacent when they differ in exactly one bit. A snake is an induced path: consecutive vertices of the path are adjacent, and no two vertices that are not consecutive on it are adjacent.

## Candidate interface

construct() -> list of at least two distinct integers in [0, 2**10): the vertices of the path in order. Consecutive vertices must differ in exactly one bit, and any two non-consecutive ones in at least two bits. Score: number of edges of the path (the number of vertices minus one), maximized.

## The instance

The longest-induced-path problem in the 10-dimensional hypercube, with the longest snake on T. Ace's leaderboard as the reference.

## Why it matters

**Problem:** Construct the longest induced path (snake) in the 10-dimensional hypercube Q_10, measured in edges.

The snake-in-the-box problem comes from coding theory (Kautz 1958, error-detecting unit-distance Gray codes) and is a standard test of combinatorial search; the maximum length is proven only up to dimension 8, so every longer snake in dimension 9 and above is a new lower bound.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [New Snake-in-the-Box Records via Snakepit Surgery and Learned Construction](https://arxiv.org/abs/2607.15270): Find longer snakes (induced paths) in hypercubes of dimensions 9 to 13 with snakepit surgery and a learned constructor. The task is the same problem in dimension 10 with an exact verifier counting edges; the paper's value for dimension 10 was the task's first reference; the reference is now the longest snake on the leaderboard that the paper cites, which is longer in dimensions 11 to 13.

## Record

The longest snake known has 379 edges: T. Ace's leaderboard (https://www.minortriad.com/snake/, read 2026-11-15) shows a snake of that length, and this task's verifier replayed it. arXiv:2607.15270 reported the records of July 2026.

**Record validation:** reproduced. the leaderboard at minortriad.com/snake lists 379 for dimension 10 and shows a snake of that length, credited to: Orland, P., Fagan, L., et al., "A Census of New Snake-in-the-Box Records", arXiv:2607.15270v1 (2026). (full dataset at github, accessed on 17 July 2026); that transition sequence replayed by our verifier: induced path with 379 edges in Q10; the transition sequence in Appendix E of arXiv 2607.15270v3 replayed by our verifier: induced path with 379 edges in Q10; Table 1 of arXiv 2607.15270v3, the source of the first version, lists 379. Note: Best known, not proven; the leaderboard was read on 2026-11-15 and changes every few weeks.

## Data and requirements

No data files.

Requires: `py:numpy`.

## Limitations

- The records come from searches tuned for this problem (cube-surgery annealing, beam search with local rerouting).
- The records move: in 2026 the leaderboard recorded improvements in every dimension of this family, several within weeks of each other. The record is the state on the day of the record validation and will fall behind.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
