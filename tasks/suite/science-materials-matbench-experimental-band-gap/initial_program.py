"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def element_table():
    """Six elemental properties from RDKit's periodic table, indexed by symbol."""
    from rdkit import Chem
    table = Chem.GetPeriodicTable()
    rows = {}
    for z in range(1, 104):
        symbol = table.GetElementSymbol(z)
        rows[symbol] = [z, table.GetAtomicWeight(z), table.GetRcovalent(z), table.GetRvdw(z), table.GetNOuterElecs(z),
                        table.GetDefaultValence(z)]
    return rows


def composition_features(composition, table):
    """Fraction-weighted mean, minimum, maximum and spread of the elemental properties."""
    elements = sorted(composition)
    amounts = np.array([composition[e] for e in elements], dtype=float)
    weights = amounts / amounts.sum()
    props = np.array([table[e] for e in elements], dtype=float)
    mean = weights @ props
    spread = np.sqrt(weights @ (props - mean) ** 2)
    return np.concatenate([mean, props.min(0), props.max(0), spread, [len(elements)]])


def fit_predict(train_inputs, train_targets, test_inputs, info):
    """Random forest on composition statistics (simple baseline)."""
    from sklearn.ensemble import RandomForestRegressor
    table = element_table()
    X = np.array([composition_features(x["composition"], table) for x in train_inputs])
    Xt = np.array([composition_features(x["composition"], table) for x in test_inputs])
    model = RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=0)
    model.fit(X, np.asarray(train_targets, dtype=float))
    return model.predict(Xt).tolist()
# EVOLVE-BLOCK-END
