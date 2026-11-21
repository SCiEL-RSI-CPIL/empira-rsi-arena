"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def forecast_premium(history):
    """Prevailing-mean seed (the Welch-Goyal benchmark): average log equity premium since 1926-01."""
    premium = history["equity_premium"]
    return float(np.mean(premium[history["yyyymm"] >= 192601]))
# EVOLVE-BLOCK-END
