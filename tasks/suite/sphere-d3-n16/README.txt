# Spherical code, 16 points in dimension 3

**Task id:** `sphere-d3-n16`\
**Status:** ready\
**Direction:** maximize\
**Family:** `sphere` (suite/_lib/arnew)\
**Areas:** mathematics, chemistry, optimization

## Task

Place 16 points on the unit sphere in R^3 so that the smallest angle between two of them is as large as possible (the Tammes problem). The angle between two points is the angle at the centre of the sphere, the arccosine of the inner product of the two unit vectors.

## Candidate interface

construct() -> array of shape (16, 3): one point per row (the verifier scales every row to unit length; a zero or non-finite row is invalid). Score: smallest angle between two points in degrees, maximized.

## The instance

Spherical code with 16 points in R^3, scored by the smallest pairwise angle; the reference is the code listed for this size in Cohn's table of spherical codes.

## Why it matters

**Problem:** Place 16 points on the unit sphere in R^3 so that the smallest angular distance between two points is as large as possible (the Tammes problem, equivalently the densest packing of 16 equal circles on a sphere).

The Tammes problem asks how to spread points on a sphere as far apart as possible; it arose from the arrangement of pores on pollen grains and models the stereochemistry of atoms around a central atom, the placing of dimples, sensors and viewing directions, and spherical codes. Optimal arrangements are proven only up to 14 points and for 24 (a proof for 15 is claimed in a 2026 preprint), so for the sizes of this family the best known packings come from computer search and a larger minimum angle is a new best packing.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [The densest packing of equal circles on a sphere](https://doi.org/10.1107/S0108767390011370): Compute the densest known packings of equal circles on a sphere for a range of numbers of circles. The task is the same packing problem for 16 circles, scored by the minimum angle. The paper's list goes to 90 circles (as Wang 2009 reports; this paper itself was not opened), the task family to 130, and the task's reference packing comes from Cohn's table of spherical codes.
- [The closest packing of equal circles on a sphere](https://doi.org/10.1098/rspa.1986.0056): Find closest packings of circles on a sphere by minimising a repulsion energy and refining to exact solutions; it reports improved packings for 20 to 40 circles. The task is the same packing problem for 16 circles; the paper's improved packings cover 20 to 40 circles, the task family 16 to 130.

## Record

The code listed for this size in Cohn's table of spherical codes reaches 52.244395753 degrees (computed by this task's verifier from the table's coordinates, read 2026-11-15). Sloane's table of 1994-96, the source of the first version of this task, lists a code with the same angle.

**Record validation:** reproduced. minimum angle of the coordinates in Cohn's table of spherical codes, computed by our verifier: 52.244395753; the table prints cosine 0.612294616483 for this size (rounded up to 12 places), the same angle to 1e-8 degrees; Sloane's table of 1994-96, the source of the first version: 52.244395745 as published, 52.244395753 after a local re-optimisation, the same angle. Note: Best known, not proven; table read on 2026-11-15. The table credits this code to: K. Schütte and B. L. van der Waerden, Auf welcher Kugel haben 5, 6, 7, 8 oder 9 Punkte mit Mindestabstand Eins Platz?, Math. Ann. 123 (1951), 96–124, doi:10.1007/BF02054944.

## Data and requirements

No data files are needed to run the task. The record comes from the coordinates at https://spherical-codes.org/data/3/16, which only the record validation reads (scripts/new_open_problems/prepare_new_data.py --family sphere).

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
