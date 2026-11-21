"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def predict_gmst(training, inputs):
    """Linear regression of the temperature anomaly on total effective radiative forcing."""
    years = inputs["forcing_year"]
    total = inputs["forcing"]["total"]
    fit = np.isin(years, training["year"])
    slope, intercept = np.polyfit(total[fit], training["gmst"], 1)
    return intercept + slope * total[np.isin(years, inputs["predict_years"])]
# EVOLVE-BLOCK-END
