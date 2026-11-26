"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def fit_predict(train, test_inputs, time_budget_s):
    """Word TF-IDF features and one logistic regression per label."""
    vectorizer = TfidfVectorizer(max_features=20000, sublinear_tf=True, strip_accents="unicode", min_df=2)
    x_train = vectorizer.fit_transform(train["comment_text"])
    x_test = vectorizer.transform(test_inputs["comment_text"])
    scores = np.zeros((x_test.shape[0], len(CLASSES)))
    for j, name in enumerate(CLASSES):
        model = LogisticRegression(C=4.0, solver="liblinear")
        model.fit(x_train, train[name].to_numpy())
        scores[:, j] = model.predict_proba(x_test)[:, 1]
    return scores
# EVOLVE-BLOCK-END
