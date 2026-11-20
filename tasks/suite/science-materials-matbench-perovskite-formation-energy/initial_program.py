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


def structure_features(structure, table):
    """Composition statistics plus density, volume per atom, cell shape and nearest-neighbour distances."""
    lattice = np.asarray(structure["lattice"], dtype=float)
    frac = np.asarray(structure["frac_coords"], dtype=float)
    species = structure["species"]
    props = np.array([table[s] for s in species], dtype=float)
    mean = props.mean(0)
    spread = props.std(0)
    volume = abs(np.linalg.det(lattice))
    mass = props[:, 1].sum()
    lengths = np.linalg.norm(lattice, axis=1)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    cart = frac @ lattice
    images = (frac[None, :, None, :] + shifts[None, None, :, :] - frac[:, None, None, :]) @ lattice
    distances = np.linalg.norm(images, axis=-1).reshape(len(species), -1)
    distances[distances < 1e-8] = np.inf
    nearest = distances.min(1)
    return np.concatenate([mean, props.min(0), props.max(0), spread,
                           [len(set(species)), len(species), volume / len(species), mass / volume,
                            lengths.min(), lengths.max(), nearest.mean(), nearest.min(), nearest.max(), np.ptp(cart[:, 2])]])


def fit_predict(train_inputs, train_targets, test_inputs, info):
    """Random forest on composition and simple structure statistics (simple baseline)."""
    from sklearn.ensemble import RandomForestRegressor
    table = element_table()
    X = np.array([structure_features(x, table) for x in train_inputs])
    Xt = np.array([structure_features(x, table) for x in test_inputs])
    model = RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=0)
    model.fit(X, np.asarray(train_targets, dtype=float))
    return model.predict(Xt).tolist()
# EVOLVE-BLOCK-END
