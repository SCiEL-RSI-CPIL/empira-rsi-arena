"""Lowest-energy conformers of drug molecules under the fixed MMFF94 force field.

The candidate returns Cartesian coordinates (Angstrom) for every atom of each
molecule in the atom order of ``Chem.AddHs(Chem.MolFromSmiles(smiles))``. The
evaluator recomputes the MMFF94 energy (RDKit implementation, no cutoff,
constant dielectric 1) of exactly those coordinates, rejects geometries whose
bonds are broken or whose atoms collide, and rejects any geometry whose
stereocentres or double-bond configurations differ from the given molecule
(assigned from 3D and compared as canonical isomeric SMILES). The score is the
mean energy above reference minima (kcal/mol).

The references were found by ``reference_search`` (ETKDGv3 embedding of
thousands of conformers, MMFF94 minimisation and torsional basin hopping from
the lowest ones, repeated with independent seeds); they are best-found values
of that search, not proven global minima and not published numbers, so a
negative score means a lower minimum than the reference search found.
Structures are approved drugs as recorded in ChEMBL; for the racemic drugs
verapamil and ranolazine one enantiomer is fixed.
"""
from __future__ import annotations

import numpy as np

from .runtime import CandidateError, require

# name, ChEMBL record, isomeric SMILES, lowest MMFF94 energy (kcal/mol) found by reference_search over 3-6 seeded runs
MOLECULES = (
    ("imatinib", "CHEMBL941", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", 64.811559),
    ("sildenafil", "CHEMBL192", "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12", -13.476892),
    ("atorvastatin", "CHEMBL1487", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O", 29.242449),
    ("oseltamivir", "CHEMBL1229", "CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@H](NC(C)=O)[C@@H](N)C1", 15.378631),
    ("tamoxifen", "CHEMBL83", "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1", 102.920209),
    ("simvastatin", "CHEMBL1064", "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@H]21", 27.477374),
    ("(S)-verapamil", "CHEMBL6966 (S enantiomer)", "COc1ccc(CCN(C)CCC[C@@](C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC", 131.094382),
    ("(R)-ranolazine", "CHEMBL1404 (R enantiomer)", "COc1ccccc1OC[C@H](O)CN1CCN(CC(=O)Nc2c(C)cccc2C)CC1", 190.03353),
)
ROTATABLE = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"


def _chem():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Chem


def molecule(smiles):
    """The evaluator's molecule: explicit hydrogens appended by RDKit AddHs."""
    Chem = _chem()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES {smiles}")
    return Chem.AddHs(mol)


def description(name, smiles):
    """What the candidate receives for one molecule."""
    mol = molecule(smiles)
    return {"name": name, "smiles": smiles, "atoms": [a.GetSymbol() for a in mol.GetAtoms()],
            "formal_charges": [a.GetFormalCharge() for a in mol.GetAtoms()],
            "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()] for b in mol.GetBonds()]}


def _with_coordinates(mol, coordinates):
    Chem = _chem()
    from rdkit.Geometry import Point3D
    copy = Chem.Mol(mol)
    copy.RemoveAllConformers()
    conformer = Chem.Conformer(copy.GetNumAtoms())
    for i, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    copy.AddConformer(conformer, assignId=True)
    return copy


def mmff_energy(mol):
    from rdkit.Chem import AllChem
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    return float(AllChem.MMFFGetMoleculeForceField(mol, properties, confId=0).CalcEnergy())


def check_geometry(mol, smiles, value):
    """Return (coordinates, molecule with them) or raise CandidateError for an invalid geometry."""
    Chem = _chem()
    n = mol.GetNumAtoms()
    try:
        x = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise CandidateError("conformer() must return an (n_atoms, 3) array of numbers") from None
    require(x.shape == (n, 3), f"expected {n} atoms as rows (x, y, z); got shape {x.shape}")
    require(bool(np.all(np.isfinite(x))), "coordinates contain NaN or infinite values")
    table = Chem.GetPeriodicTable()
    radii = np.array([table.GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()])
    distance = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    bonded = np.zeros((n, n), bool)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded[i, j] = bonded[j, i] = True
        ratio = distance[i, j] / (radii[i] + radii[j])
        require(0.7 <= ratio <= 1.35, f"bond {i}-{j} has length {distance[i, j]:.3f} A (outside 0.70-1.35 x the covalent radii sum)")
    iu = np.triu_indices(n, 1)
    free = ~bonded[iu]
    closest = float(np.min(distance[iu][free])) if np.any(free) else float("inf")
    require(closest >= 0.6, f"two non-bonded atoms are {closest:.3f} A apart")
    placed = _with_coordinates(mol, x)
    probe = Chem.Mol(placed)
    Chem.AssignStereochemistryFrom3D(probe, replaceExistingTags=True)
    found = Chem.MolToSmiles(Chem.RemoveHs(probe))
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    require(found == expected, f"stereochemistry or connectivity changed: geometry corresponds to {found}, expected {expected}")
    return x, placed


def conformer_set_score(conformer):
    """Mean MMFF94 energy above the reference minima (kcal/mol); lower is better."""
    rows = []
    for name, record, smiles, reference in MOLECULES:
        mol = molecule(smiles)
        _, placed = check_geometry(mol, smiles, conformer(description(name, smiles)))
        energy = mmff_energy(placed)
        rows.append({"name": name, "chembl": record, "energy_kcal_mol": energy, "reference_kcal_mol": reference,
                     "excess_kcal_mol": None if reference is None else energy - reference, "atoms": mol.GetNumAtoms()})
    if any(r["reference_kcal_mol"] is None for r in rows):
        raise RuntimeError("reference energies are not recorded")
    excess = [r["excess_kcal_mol"] for r in rows]
    return float(np.mean(excess)), {"molecules": rows, "within_0_01_kcal": int(sum(e <= 0.01 for e in excess)),
                                    "below_reference": int(sum(e < -0.01 for e in excess))}


def minimise(mol, max_iterations=20000):
    """MMFF94 minimisation of every conformer of ``mol`` in place; returns energies (helper for candidates and references)."""
    from rdkit.Chem import AllChem
    result = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=max_iterations, mmffVariant="MMFF94")
    return [float(e) for _, e in result]


def reference_search(smiles, seed, conformers=2000, starts=30, steps=80, threads=4):
    """Best MMFF94 minimum found by ETKDGv3 sampling plus torsional basin hopping (documentation and checks)."""
    Chem = _chem()
    from rdkit.Chem import AllChem, rdMolTransforms
    rng = np.random.default_rng(seed)
    mol = molecule(smiles)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = int(threads)
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(conformers), params=params))
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=int(threads), maxIters=20000, mmffVariant="MMFF94")
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    pattern = Chem.MolFromSmarts(ROTATABLE)
    torsions = []
    for j, k in mol.GetSubstructMatches(pattern):
        a = next(n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k)
        b = next(n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j)
        torsions.append((a, j, k, b))

    def valid(m, cid):
        x = m.GetConformer(cid).GetPositions()
        try:
            check_geometry(molecule(smiles), smiles, x)
            return True
        except CandidateError:
            return False

    ranked = sorted((e, cid) for cid, (_, e) in zip(ids, results) if valid(mol, cid))
    best_energy, best_xyz = ranked[0][0], mol.GetConformer(ranked[0][1]).GetPositions()
    for energy, cid in ranked[:starts]:
        current = Chem.Mol(mol, confId=cid)
        current.GetConformer().SetId(0)
        current_energy = energy
        for _ in range(steps):
            trial = Chem.Mol(current)
            conf = trial.GetConformer()
            for t in rng.choice(len(torsions), size=min(len(torsions), int(rng.integers(1, 4))), replace=False):
                angle = rdMolTransforms.GetDihedralDeg(conf, *torsions[t]) + float(rng.uniform(-180.0, 180.0))
                rdMolTransforms.SetDihedralDeg(conf, *torsions[t], angle)
            field = AllChem.MMFFGetMoleculeForceField(trial, properties, confId=0)
            field.Minimize(maxIts=20000)
            trial_energy = field.CalcEnergy()
            if trial_energy < current_energy - 1e-9 or rng.random() < np.exp(-(trial_energy - current_energy) / 1.0):
                if valid(trial, 0):
                    current, current_energy = trial, trial_energy
                    if trial_energy < best_energy:
                        best_energy, best_xyz = trial_energy, trial.GetConformer().GetPositions()
    return float(best_energy), np.asarray(best_xyz)
