# Lowest-energy MMFF94 conformers of eight drug molecules

**Source collection:** [Molecular and materials design research: source-based research tasks](../../papers/science-chemistry_research.md)\
**Task id:** `science-chemistry-mmff94-drug-conformers`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

Find, for each of eight flexible drug molecules, the conformer with the lowest MMFF94 force-field energy while keeping the molecule's connectivity and stereochemistry. The evaluator recomputes the energy of the returned coordinates and compares it with minima from an extensive reference search.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

conformer(molecule) -> array of shape (n_atoms, 3): Cartesian coordinates in Angstrom of every atom, in the atom order of Chem.AddHs(Chem.MolFromSmiles(molecule['smiles'])) (RDKit 2026.03). molecule = {'name', 'smiles', 'atoms' (element symbols in that order), 'formal_charges', 'bonds' ([i, j, bond order])}. Called for eight drugs (imatinib, sildenafil, atorvastatin, oseltamivir, tamoxifen, simvastatin, (S)-verapamil, (R)-ranolazine); all calls together within 300 s, at most 4 threads. The evaluator computes the MMFF94 energy (RDKit, no cutoff, dielectric 1) of exactly the returned coordinates; every bond must lie within 0.70-1.35 times the sum of covalent radii, non-bonded atoms at least 0.6 A apart, and the stereochemistry assigned from the 3D geometry must reproduce the given isomeric SMILES. Score: mean MMFF94 energy above reference minima found by an extensive local search (kcal/mol; minimize; negative = lower)

## State of the art

No published global minima exist for these molecules under this protocol; the scored references come from the collection's own reference search (validation/chemistry-checks.json).

Reference value: no documented value for this finite instance (reviewed 2026-10-10).


## Labels

Scientific contribution: `method` (tier 2); artifact `construction`, domain `natural_science`, data `none`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `chemistry`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Find the lowest-energy conformations of flexible molecules on a molecular-mechanics potential energy surface.

Why it matters: Low-energy conformers are the starting point of property prediction, docking and spectroscopy; conformational search becomes difficult as the number of rotatable bonds grows, and missed minima bias downstream calculations.

- [Automated exploration of the low-energy chemical space with fast quantum chemical methods](https://doi.org/10.1039/C9CP06869D): Main research objective: Develop an automated workflow that finds low-energy conformers and rotamers of molecules. Task connection: The task scores how low an MMFF94 energy a conformer search reaches on eight drug molecules.
- [Low Mode Search. An Efficient, Automated Computational Method for Conformational Analysis](https://doi.org/10.1021/ja952478m): Main research objective: Introduce an efficient, automated global conformational search method for flexible molecules. Task connection: The task poses the same global conformational search problem under the fixed MMFF94 force field.

The cited papers' main objective is the underlying problem the task scores; the local data, splits and limits are stated in the task's limitations.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- MMFF94 is RDKit's implementation; the reference minima are the lowest energies found by chem_conformers.reference_search (ETKDGv3, MMFF94 minimisation, torsional basin hopping; three seeded runs per molecule and three more with larger settings for atorvastatin and simvastatin, whose first runs disagreed), not published values and not proven global minima. Each reference was reached by at least three runs and is attained by a witness geometry stored in validation/chemistry-checks.json.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:rdkit`

## Sources

- paper: https://doi.org/10.1039/C9CP06869D (Pracht, Bohle and Grimme (2020): automated search for low-energy conformers and rotamers (CREST).)
- paper: https://doi.org/10.1021/ja952478m (Kolossvary and Guida (1996): low-mode search for efficient global conformational analysis.)
- paper: https://doi.org/10.1021/acs.jcim.5b00654 (Riniker and Landrum (2015): ETKDG conformer generation, used by the seed and the reference search.)
- paper: https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6%3C490::AID-JCC1%3E3.0.CO;2-P (Halgren (1996): the MMFF94 force field (RDKit implementation used here).)
- benchmark: https://www.ebi.ac.uk/chembl/ (ChEMBL records CHEMBL941, CHEMBL192, CHEMBL1487, CHEMBL1229, CHEMBL83, CHEMBL1064, CHEMBL6966 and CHEMBL1404 for the structures.)

## Limitations

- The references are the lowest energies found by this collection's own reference search (ETKDGv3 sampling, MMFF94 minimisation and torsional basin hopping with three or more independent seeds; per-seed results in validation/chemistry-checks.json), not published values and not proven global minima; a negative score means the candidate found lower minima.
- MMFF94 in vacuum with a constant dielectric favours intramolecular hydrogen bonds and folded conformers; the force-field minimum is not the bioactive or solution conformer, so the task measures search, not conformational realism.
- Atom order and energies depend on the RDKit version (2026.03 here). Verapamil and ranolazine are racemic drugs of which one enantiomer is fixed. The time budget (300 s for all eight molecules) is checked only as a whole.
