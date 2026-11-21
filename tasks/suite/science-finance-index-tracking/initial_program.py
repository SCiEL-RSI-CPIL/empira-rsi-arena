"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import nnls


def rebalance(state):
    """Seed: at week 145 buy the 10 stocks most correlated with the index, weighted by non-negative least squares of
    index returns on their returns (each at least 1%); then buy and hold."""
    if state["week"] != 145:
        return None
    index_returns = np.diff(np.log(state["index_prices"]))
    stock_returns = np.diff(np.log(state["stock_prices"]), axis=0)
    corr = np.array([np.corrcoef(stock_returns[:, j], index_returns)[0, 1] for j in range(stock_returns.shape[1])])
    chosen = np.argsort(-corr)[:state["K"]]
    coef, _ = nnls(stock_returns[:, chosen], index_returns)
    share = np.maximum(coef, 0.0) + 1e-12
    w = np.zeros(stock_returns.shape[1])
    w[chosen] = state["min_weight"] + (1.0 - state["min_weight"] * len(chosen)) * share / share.sum()
    return w
# EVOLVE-BLOCK-END
