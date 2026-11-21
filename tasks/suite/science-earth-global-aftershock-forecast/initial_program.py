"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

import numpy as np


def expected_count(a, p, magnitude, offset, length, c=0.05):
    """Reasenberg-Jones (1989) expected number of M >= 5 aftershocks in [offset, offset + length) days (b = 1)."""
    integral = ((offset + length + c) ** (1 - p) - (offset + c) ** (1 - p)) / (1 - p)
    return 10 ** (a + magnitude - 5.0) * integral


class PoissonReasenbergJones:
    def __init__(self, a, p, edges):
        self.a, self.p, self.edges = a, p, [int(e) for e in edges]

    def forecast(self, history, case):
        mean = expected_count(self.a, self.p, case["main_mag"], case["start"] - case["main_t"], case["window_days"])
        n = np.arange(self.edges[-1])
        log_pmf = n * math.log(mean) - mean - np.array([math.lgamma(k + 1.0) for k in n])
        pmf = np.exp(log_pmf)
        probs = [pmf[self.edges[i]:self.edges[i + 1]].sum() for i in range(len(self.edges) - 1)] + [max(0.0, 1.0 - pmf.sum())]
        probs = np.maximum(np.array(probs), 1e-6)
        return probs / probs.sum()


def fit_aftershock_model(history, training_cases, info):
    """Poisson maximum likelihood of the productivity a and decay p on the training cases (grid search)."""
    offsets = training_cases["start"] - training_cases["main_t"]
    counts = training_cases["count"]
    best = None
    for p in np.arange(0.75, 1.40, 0.05):
        for a in np.arange(-3.5, -1.0, 0.05):
            mean = expected_count(a, p, training_cases["main_mag"], offsets, training_cases["window_days"])
            ll = np.sum(counts * np.log(mean) - mean)
            if best is None or ll > best[0]:
                best = (ll, a, p)
    return PoissonReasenbergJones(best[1], best[2], info["bin_edges"])
# EVOLVE-BLOCK-END
