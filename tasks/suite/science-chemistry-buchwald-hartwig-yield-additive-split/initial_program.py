"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

COMPONENTS = ("aryl_halide", "ligand", "base", "additive")


def featurize(reactions):
    """Concatenated 256-bit Morgan fingerprints (radius 2) of the four varied components."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    cache = {}

    def bits(smiles):
        if smiles not in cache:
            cache[smiles] = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)).astype(float)
        return cache[smiles]
    return np.array([np.concatenate([bits(r[c]) for c in COMPONENTS]) for r in reactions])


def predict_yields(train_reactions, train_yields, test_reactions, info):
    """Random forest on component fingerprints (simple baseline); yields in percent."""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=0)
    model.fit(featurize(train_reactions), np.asarray(train_yields, dtype=float))
    return np.clip(model.predict(featurize(test_reactions)), 0.0, 100.0).tolist()
# EVOLVE-BLOCK-END
