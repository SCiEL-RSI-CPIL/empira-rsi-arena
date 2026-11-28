"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

ESTIMATION_START, MAX_LAG = 196001, 6


def ahead(levels, kind, t, h):
    """Target h months after month index t: annualized average growth or the level change."""
    if kind == "growth":
        return 1200.0 / h * np.log(levels[t + h] / levels[t])
    return levels[t + h] - levels[t]


def forecast_target(history, horizons):
    """Seed: the benchmark direct autoregression with BIC lag selection (scores exactly 1)."""
    months, kind = history["yyyymm"], history["target_kind"]
    levels = history["panel"][:, history["series"].index(history["target"])]
    z = np.r_[np.nan, (1200.0 * np.diff(np.log(levels))) if kind == "growth" else np.diff(levels)]
    t = len(levels) - 1
    first = int(np.searchsorted(months, ESTIMATION_START)) + MAX_LAG
    out = []
    for h in horizons:
        rows = np.arange(first, t - h + 1)
        y = np.array([ahead(levels, kind, s, h) for s in rows])
        lags = np.column_stack([z[rows - j] for j in range(MAX_LAG)])
        now = np.array([z[t - j] for j in range(MAX_LAG)])
        best = None
        for p in range(MAX_LAG + 1):
            X = np.column_stack([np.ones(len(rows)), lags[:, :p]])
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            ssr = float(np.sum((y - X @ coef) ** 2))
            bic = len(y) * np.log(ssr / len(y)) + (p + 1) * np.log(len(y))
            if best is None or bic < best[0]:
                best = (bic, float(coef @ np.r_[1.0, now[:p]]))
        out.append(best[1])
    return np.array(out)
# EVOLVE-BLOCK-END
