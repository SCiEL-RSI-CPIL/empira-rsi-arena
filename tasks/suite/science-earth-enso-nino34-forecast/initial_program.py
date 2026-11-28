"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

LEADS = 12


class DampedPersistence:
    """Nino 3.4 anomaly persistence damped by lag regressions fitted per calendar month."""

    def __init__(self, climatology, damping):
        self.climatology, self.damping = climatology, damping

    def forecast(self, history):
        month = int(history["month"][-1]) - 1
        anomaly = history["nino34"][-1] - self.climatology[month]
        targets = (month + np.arange(1, LEADS + 1)) % 12
        return self.climatology[targets] + anomaly * self.damping[month]


def fit_enso_model(training):
    nino, month = np.asarray(training["nino34"]), np.asarray(training["month"]) - 1
    recent = slice(len(nino) - 360, len(nino))
    climatology = np.array([nino[recent][month[recent] == m].mean() for m in range(12)])
    anomaly = nino - climatology[month]
    damping = np.zeros((12, LEADS))
    for m in range(12):
        starts = np.where(month[:-LEADS] == m)[0]
        for lead in range(1, LEADS + 1):
            x, y = anomaly[starts], anomaly[starts + lead]
            damping[m, lead - 1] = np.dot(x, y) / np.dot(x, x)
    return DampedPersistence(climatology, damping)
# EVOLVE-BLOCK-END
