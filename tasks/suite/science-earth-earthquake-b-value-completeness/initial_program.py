"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

BIN = 0.1


def estimate_b_and_mc(catalogues):
    """Maximum-curvature Mc plus 0.2 and the Aki-Utsu maximum-likelihood b above it (Woessner and Wiemer 2005)."""
    out = []
    for magnitudes in catalogues:
        values, counts = np.unique(np.round(magnitudes, 1), return_counts=True)
        mc = values[np.argmax(counts)] + 0.2
        above = magnitudes[magnitudes >= mc - 1e-9]
        b = np.log10(np.e) / max(above.mean() - (mc - BIN / 2), 1e-3) if len(above) > 1 else 1.0
        out.append([float(np.clip(b, 0.1, 3.0)), float(mc)])
    return np.array(out)
# EVOLVE-BLOCK-END
