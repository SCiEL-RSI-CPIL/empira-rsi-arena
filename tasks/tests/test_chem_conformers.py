"""MMFF94 conformer task: geometry and stereochemistry checks, energies and the reference minima."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import chem_conformers as cf  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import seed_source, slow  # noqa: E402

NEEDS_RDKIT = unittest.skipUnless(importlib.util.find_spec("rdkit"), "RDKit is not installed")
OSELTAMIVIR = dict((m[0], m[2]) for m in cf.MOLECULES)["oseltamivir"]


def embedded(smiles, seed=3):
    from rdkit.Chem import AllChem
    mol = cf.molecule(smiles)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=5000)
    return mol.GetConformer().GetPositions()


@NEEDS_RDKIT
class GeometryCheckTests(unittest.TestCase):
    def test_minimised_conformer_is_accepted_and_energy_recomputed(self):
        mol = cf.molecule(OSELTAMIVIR)
        x = embedded(OSELTAMIVIR)
        _, placed = cf.check_geometry(mol, OSELTAMIVIR, x)
        energy = cf.mmff_energy(placed)
        self.assertTrue(np.isfinite(energy))
        self.assertAlmostEqual(cf.mmff_energy(cf.check_geometry(mol, OSELTAMIVIR, x.tolist())[1]), energy, places=9)

    def test_mirror_image_and_epimer_are_rejected(self):
        mol = cf.molecule(OSELTAMIVIR)
        mirror = embedded(OSELTAMIVIR) * np.array([-1.0, 1.0, 1.0])
        with self.assertRaises(CandidateError):
            cf.check_geometry(mol, OSELTAMIVIR, mirror)
        epimer = OSELTAMIVIR.replace("[C@@H](N)", "[C@H](N)")
        self.assertNotEqual(epimer, OSELTAMIVIR)
        with self.assertRaises(CandidateError):
            cf.check_geometry(mol, OSELTAMIVIR, embedded(epimer))

    def test_broken_or_malformed_geometries_are_rejected(self):
        mol = cf.molecule(OSELTAMIVIR)
        x = embedded(OSELTAMIVIR)
        stretched = x.copy()
        stretched[0] += 3.0
        for bad in (stretched, x[:-1], np.full_like(x, np.nan), np.zeros_like(x), "coordinates"):
            with self.assertRaises(CandidateError):
                cf.check_geometry(mol, OSELTAMIVIR, bad)

    def test_description_uses_the_evaluator_atom_order(self):
        description = cf.description("oseltamivir", OSELTAMIVIR)
        mol = cf.molecule(OSELTAMIVIR)
        self.assertEqual(description["atoms"], [a.GetSymbol() for a in mol.GetAtoms()])
        self.assertEqual(len(description["bonds"]), mol.GetNumBonds())


@NEEDS_RDKIT
class ReferenceTests(unittest.TestCase):
    def test_every_molecule_has_a_reference_and_fully_specified_stereochemistry(self):
        from rdkit import Chem
        for name, record, smiles, reference in cf.MOLECULES:
            self.assertIsInstance(reference, float, name)
            stereo = Chem.FindPotentialStereo(Chem.MolFromSmiles(smiles))
            self.assertTrue(all(str(s.specified) == "Specified" for s in stereo), name)

    # Measured at 27 s on one thread: runs only with ARENA_SLOW_TESTS=1.
    @slow
    def test_seed_is_valid_and_not_below_the_references(self):
        namespace = {}
        exec(seed_source("science-chemistry-mmff94-drug-conformers"), namespace)  # the package's seed
        score, details = cf.conformer_set_score(namespace["conformer"])
        self.assertGreater(score, 0.0)
        self.assertTrue(all(row["excess_kcal_mol"] > -1e-6 for row in details["molecules"]))


if __name__ == "__main__":
    unittest.main()
