# Spherical code, 129 points in dimension 5

**Task id:** `sphere-d5-n129`\
**Status:** ready\
**Direction:** maximize\
**Family:** `sphere` (suite/_lib/arnew)\
**Areas:** mathematics, theoretical computer science, optimization

## Task

Place 129 points on the unit sphere in R^5 so that the smallest angle between two of them is as large as possible (a spherical code). The angle between two points is the angle at the centre of the sphere, the arccosine of the inner product of the two unit vectors.

## Candidate interface

construct() -> array of shape (129, 5): one point per row (the verifier scales every row to unit length; a zero or non-finite row is invalid). Score: smallest angle between two points in degrees, maximized.

## The instance

Spherical code with 129 points in R^5, scored by the smallest pairwise angle; the reference is the code listed for this size in Cohn's table of spherical codes.

## Why it matters

**Problem:** Find 129 points on the unit sphere of R^5 whose smallest pairwise angle is as large as possible, that is a spherical code of maximal minimum angle; the reference is the best code known for this size, as listed in Cohn's table of spherical codes.

Maximising the minimum angle of n points on a sphere is the Tammes problem (dimension 3, from the packing of pollen-grain apertures) and, in higher dimension, the spherical code problem of coding theory and sphere packing. Optimality is proven only for few n (n <= 14 and 24 in R^3, with a proof for 15 claimed in 2026, and a handful of special codes in R^4 and R^5), so the listed codes for larger n are the best known, not proven, and a larger minimum angle would be a new best code.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Finding and Investigating Exact Spherical Codes](https://doi.org/10.1080/10586458.2009.10128893): Find good spherical codes of N points in R^n by energy minimisation from many random starts (dimensions 3 to 8), prove that the codes found exist exactly by writing their inner products as algebraic numbers, and catalogue them. The task is the same problem for 129 points in R^5. The paper's searches reach 72 points in R^4 and 50 points in R^5 and report three improved codes in R^4 (40, 68 and 71 points); the task family goes to 130 points, so its larger sizes pose the paper's problem beyond the sizes the paper searched.

## Record

The code listed for this size in Cohn's table of spherical codes reaches 43.884295585 degrees (computed by this task's verifier from the table's coordinates, read 2026-11-15). Sloane's table of 1994-96, the source of the first version of this task, reaches 43.867206367 degrees after a local re-optimisation: 0.017 degrees less.

**Record validation:** reproduced. minimum angle of the coordinates in Cohn's table of spherical codes, computed by our verifier: 43.884295585; the table prints cosine 0.720741141724 for this size (rounded up to 12 places), the same angle to 1e-8 degrees; Sloane's table of 1994-96, the source of the first version: 43.867206148 as published, 43.867206367 after a local re-optimisation, 0.017 degrees below the record. Note: Best known, not proven; table read on 2026-11-15. The table credits this code to: Ilmari Vahteristo, New spherical codes by tuning an annealing solver, preprint, 2026, arXiv:2610.19988.

## Data and requirements

No data files are needed to run the task. The record comes from the coordinates at https://spherical-codes.org/data/5/129, which only the record validation reads (scripts/new_open_problems/prepare_new_data.py --family sphere).

Requires: `py:numpy`.

## Limitations

- The listed code is the best known one, found by computer search and not proven optimal (sizes that are proven have no task).
- The table is updated all the time: codes of this family were improved in 2022 to 2026. The record is the state on 2026-11-15 and will fall behind.
- A result counts as an improvement only above the record by more than 1e-5 degrees, a margin against numerical slack in the listed coordinates.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
