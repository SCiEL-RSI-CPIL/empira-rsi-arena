"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def two_stream_coefficients(omega, g, mu0):
    """Delta-Eddington (Joseph, Wiscombe and Weinman 1976): Eddington coefficients after delta-scaling with f = g^2."""
    f = g ** 2
    omega_s = omega * (1 - f) / (1 - omega * f)
    g_s = (g - f) / (1 - f)
    gamma1 = (7 - omega_s * (4 + 3 * g_s)) / 4
    gamma2 = -(1 - omega_s * (4 - 3 * g_s)) / 4
    gamma3 = (2 - 3 * g_s * mu0) / 4
    return gamma1, gamma2, gamma3, f
# EVOLVE-BLOCK-END
