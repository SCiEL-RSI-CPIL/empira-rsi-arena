"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

HYDROGEN_EXCESS_KEV = 7288.971064
NEUTRON_EXCESS_KEV = 8071.31806


def _features(n, z):
    a = n + z
    pairing = np.where((n % 2 == 0) & (z % 2 == 0), 1.0, np.where((n % 2 == 1) & (z % 2 == 1), -1.0, 0.0))
    return np.column_stack([a, -a ** (2 / 3), -z * (z - 1) / a ** (1 / 3), -(n - z) ** 2 / a, pairing / np.sqrt(a)])


def predict_masses(training, queries):
    """Liquid-drop (Bethe-Weizsaecker) binding energy fitted by least squares; returns mass excesses in keV."""
    n = np.array([row["N"] for row in training], float)
    z = np.array([row["Z"] for row in training], float)
    excess = np.array([row["mass_excess_keV"] for row in training], float)
    binding = z * HYDROGEN_EXCESS_KEV + n * NEUTRON_EXCESS_KEV - excess
    coefficients, *_ = np.linalg.lstsq(_features(n, z), binding, rcond=None)
    qn = np.array([q[0] for q in queries], float)
    qz = np.array([q[1] for q in queries], float)
    predicted_binding = _features(qn, qz) @ coefficients
    return (qz * HYDROGEN_EXCESS_KEV + qn * NEUTRON_EXCESS_KEV - predicted_binding).tolist()
# EVOLVE-BLOCK-END
