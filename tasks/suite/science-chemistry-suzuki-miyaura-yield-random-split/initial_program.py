"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def featurize(reactions):
    """Summed 512-bit Morgan count fingerprints (radius 2) of the molecules left of '>>' in each reaction SMILES."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    cache = {}
    rows = []
    for reaction in reactions:
        left = reaction["reaction_smiles"].split(">>")[0]
        if left not in cache:
            mol = Chem.MolFromSmiles(left)
            cache[left] = generator.GetCountFingerprintAsNumPy(mol).astype(float)
        rows.append(cache[left])
    return np.array(rows)


def predict_yields(train_reactions, train_yields, test_reactions, info):
    """Random forest on reactant-side count fingerprints (simple baseline); yields in percent."""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=0)
    model.fit(featurize(train_reactions), np.asarray(train_yields, dtype=float))
    return np.clip(model.predict(featurize(test_reactions)), 0.0, 100.0).tolist()
# EVOLVE-BLOCK-END
