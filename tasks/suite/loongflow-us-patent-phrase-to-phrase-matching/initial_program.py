"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_predict(train, test_inputs, time_budget_s):
    """Cosine similarity of character 2-4-gram TF-IDF vectors of anchor and target (no training)."""
    phrases = np.concatenate([train["anchor"], train["target"], test_inputs["anchor"], test_inputs["target"]])
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(phrases)
    a = vectorizer.transform(test_inputs["anchor"])
    b = vectorizer.transform(test_inputs["target"])
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()
# EVOLVE-BLOCK-END
