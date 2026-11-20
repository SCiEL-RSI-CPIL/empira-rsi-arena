"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def gaussian_ratio(presence, background):
    """Per-covariate Gaussian log-density ratio of presences to background (naive Bayes)."""
    mu_p, sd_p = np.nanmean(presence, axis=0), np.nanstd(presence, axis=0) + 1e-6
    mu_b, sd_b = np.nanmean(background, axis=0), np.nanstd(background, axis=0) + 1e-6

    def score(x):
        x = np.where(np.isnan(x), mu_b, x)
        lp = -0.5 * ((x - mu_p) / sd_p) ** 2 - np.log(sd_p)
        lb = -0.5 * ((x - mu_b) / sd_b) ** 2 - np.log(sd_b)
        return np.sum(lp - lb, axis=1)
    return score


def predict_distributions(regions):
    out = {}
    for name, region in regions.items():
        keep = [i for i, c in enumerate(region["covariates"]) if c not in region["categorical"]]
        background = region["background"]["x"][:, keep]
        out[name] = {}
        for group in region["sites"].values():
            sites = group["x"][:, keep]
            for species in group["species"]:
                model = gaussian_ratio(region["presence"][species]["x"][:, keep], background)
                out[name][species] = model(sites)
    return out
# EVOLVE-BLOCK-END
