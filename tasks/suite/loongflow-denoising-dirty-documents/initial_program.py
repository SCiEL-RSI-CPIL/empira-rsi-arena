"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def fit_predict(train, test_inputs, time_budget_s):
    """Lookup table: each noisy grey level maps to the mean clean intensity seen with it in training."""
    sums, counts = np.zeros(256), np.zeros(256)
    for noisy, clean in zip(train["noisy"], train["clean"]):
        sums += np.bincount(noisy.ravel(), weights=clean.ravel() / 255.0, minlength=256)
        counts += np.bincount(noisy.ravel(), minlength=256)
    table = np.where(counts > 0, sums / np.maximum(counts, 1), np.arange(256) / 255.0)
    return [table[image] for image in test_inputs["noisy"]]
# EVOLVE-BLOCK-END
