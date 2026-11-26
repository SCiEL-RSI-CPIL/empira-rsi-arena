"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

COLUMNS = ["number_of_total_atoms", "percent_atom_al", "percent_atom_ga", "percent_atom_in",
           "lattice_vector_1_ang", "lattice_vector_2_ang", "lattice_vector_3_ang",
           "lattice_angle_alpha_degree", "lattice_angle_beta_degree", "lattice_angle_gamma_degree"]
SPACEGROUPS = [12, 33, 167, 194, 206, 227]


def features(table):
    x = table[COLUMNS].to_numpy(dtype=float)
    volume = np.array([abs(np.linalg.det(np.array(v, dtype=float))) for v in table["lattice_vectors"]])
    onehot = np.stack([(table["spacegroup"] == g).to_numpy(dtype=float) for g in SPACEGROUPS], axis=1)
    return np.column_stack([x, volume / x[:, 0], onehot])


def fit_predict(train, test_inputs, time_budget_s):
    """One gradient-boosted tree model per target, fitted on log1p of the target."""
    x_train, x_test = features(train), features(test_inputs)
    out = []
    for target in ["formation_energy_ev_natom", "bandgap_energy_ev"]:
        model = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0)
        model.fit(x_train, np.log1p(train[target].to_numpy(dtype=float)))
        out.append(np.clip(np.expm1(model.predict(x_test)), 0.0, None))
    return np.column_stack(out)
# EVOLVE-BLOCK-END
