"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def fit_predict(train_smiles, train_y, valid_smiles, valid_y, test_smiles, info):
    # Baseline: random forest on 1024-bit Morgan fingerprints.
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    def fingerprints(smiles):
        out = np.zeros((len(smiles), 1024))
        for row, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                out[row, list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).GetOnBits())] = 1
        return out

    x = np.vstack([fingerprints(train_smiles), fingerprints(valid_smiles)])
    y = np.concatenate([train_y, valid_y])
    xt = fingerprints(test_smiles)
    if info["kind"] == "binary":
        return RandomForestClassifier(300, random_state=0, n_jobs=1).fit(x, y).predict_proba(xt)[:, 1]
    return RandomForestRegressor(300, random_state=0, n_jobs=1).fit(x, y).predict(xt)
# EVOLVE-BLOCK-END
