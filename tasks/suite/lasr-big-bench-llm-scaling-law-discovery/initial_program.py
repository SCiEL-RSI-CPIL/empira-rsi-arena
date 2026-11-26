"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

N_PARAMS = 2


def initial_params():
    return [0.3, 0.01]


def scaling_law(X, params):
    """Grade linear in the log of training compute (training steps x tokens per batch)."""
    E, a = params
    return E + a * np.log10(X["training_steps"] * X["batch_size"])
# EVOLVE-BLOCK-END
