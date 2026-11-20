"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def featurize(smiles):
    """1024-bit Morgan (radius 2) fingerprints plus six RDKit descriptors."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            rows.append(np.zeros(1030))
            continue
        descriptors = [Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Descriptors.MolWt(mol),
                       Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol)]
        rows.append(np.concatenate([generator.GetFingerprintAsNumPy(mol).astype(float), descriptors]))
    return np.array(rows)


def predict(train_smiles, train_values, test_smiles, info):
    """Random forest on fingerprints and descriptors (simple baseline)."""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=0)
    model.fit(featurize(train_smiles), np.asarray(train_values, dtype=float))
    return model.predict(featurize(test_smiles)).tolist()
# EVOLVE-BLOCK-END
