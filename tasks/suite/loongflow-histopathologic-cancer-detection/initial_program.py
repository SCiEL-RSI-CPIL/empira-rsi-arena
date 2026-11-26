"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def features(images):
    """Colour statistics of the whole crop and of its central 32 x 32 region (where the label is defined)."""
    x = images.astype(np.float32) / 255.0
    centre = x[:, 8:40, 8:40, :]
    parts = [x.mean(axis=(1, 2)), x.std(axis=(1, 2)), centre.mean(axis=(1, 2)), centre.std(axis=(1, 2))]
    grey = centre.mean(axis=3).reshape(len(x), -1)
    parts.append(np.stack([(grey < t).mean(axis=1) for t in (0.3, 0.45, 0.6, 0.75, 0.9)], axis=1))
    return np.concatenate(parts, axis=1)


def fit_predict(train, test_inputs, time_budget_s):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features(train["images"]))
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(x_train, train["labels"])
    return model.predict_proba(scaler.transform(features(test_inputs["images"])))[:, 1]
# EVOLVE-BLOCK-END
