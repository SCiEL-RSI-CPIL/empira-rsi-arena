# Thomson problem, N = 167

**Task id:** `thomson-n167`\
**Status:** ready\
**Direction:** minimize\
**Family:** `thomson` (suite/_lib/arnew)\
**Areas:** physics, mathematics, chemistry, optimization

## Task

Place 167 equal point charges on the unit sphere in R^3 so that their Coulomb energy, the sum of 1/distance over all pairs of points, is as small as possible (the Thomson problem).

## Candidate interface

construct() -> array of shape (167, 3): one point per row (the verifier scales every row to unit length; a zero or non-finite row is invalid, and so are two points that coincide). Score: sum over all pairs of points of 1/(Euclidean distance), minimized.

## The instance

Thomson problem for N=167 point charges on the unit sphere, with the smallest known Coulomb energy taken from the tabulated putative global minima.

## Why it matters

**Problem:** Find the arrangement of 167 unit point charges on the unit sphere with the smallest Coulomb energy, the sum of 1/distance over all pairs (the Thomson problem).

The Thomson problem is the classical model of repelling charges confined to a sphere, which Thomson posed in his 1904 atom model and which now serves as a model for spherical crystals and shells; its energy landscape has exponentially many local minima, so global minima beyond N around 100 are putative. Only a few N (2 to 6 and 12) are rigorously proven, so a lower energy would be a new putative global minimum.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Equilibrium configurations of N equal charges on a sphere](https://doi.org/10.1088/0305-4470/24/23/008): Find, by numerical minimisation, equilibrium configurations of N equal charges on a sphere and their energies. The task asks for the minimum-energy configuration of N charges (the sum of 1/distance); this paper's search tables are an early record source for small N, while the task's N is larger in most cases.
- [Structure and dynamics of spherical crystals characterized for the Thomson problem](https://doi.org/10.1103/PhysRevB.74.212101): Characterise putative global minima of the Thomson problem and the structure and dynamics of the resulting spherical crystals. The Cambridge Cluster Database table of global minima that carries the paper's name is the source of the smallest known energies (Wikipedia reproduces it); the task asks to lower the energy for a given N.
- [Defect motifs for spherical topologies](https://doi.org/10.1103/PhysRevB.79.224115): Study the defect motifs of low-energy configurations on the sphere, including the Thomson problem, extending the global-minimum search to larger N. Its configurations are also reproduced in the Cambridge Cluster Database; the task asks to lower the energy of a given N.

## Record

Smallest known energy 12753.46942975 (Wikipedia table, read 2026-11-03; the table carries a 'citation needed' tag).

**Record validation:** matches a pinned source. energy 12753.469429750 found in the pinned Wikipedia table; N > 150: optimiser reproduction not attempted. Note: The Wikipedia table carries a 'citation needed' tag.

## Data and requirements

No data files.

Requires: `py:numpy`.

## Limitations

- Only N = 2 to 6 and N = 12 have a published rigorous proof (a proof for N = 7 was announced in 2026 and is not peer reviewed); the other energies are putative minima that many groups have optimised for decades, so matching is the likely outcome.
- The record has 9 decimals; a result counts as an improvement only below it by more than 1e-8.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
