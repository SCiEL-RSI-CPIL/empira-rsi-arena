"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def conformer(molecule):
    """ETKDGv3 embedding of 20 conformers, MMFF94 minimisation, lowest-energy conformer returned."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(molecule["smiles"]))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=20, params=params))
    energies = [energy for _, energy in AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=10000, mmffVariant="MMFF94")]
    best = int(np.argmin(energies))
    return mol.GetConformer(ids[best]).GetPositions()
# EVOLVE-BLOCK-END
