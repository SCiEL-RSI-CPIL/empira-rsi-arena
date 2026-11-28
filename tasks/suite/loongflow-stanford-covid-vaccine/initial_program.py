"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]


def fit_predict(train, test_inputs, time_budget_s):
    """Mean target per (nucleotide, predicted loop type) pair over the scored training positions."""
    sums, counts = {}, {}
    for _, row in train.iterrows():
        values = np.array([row[t] for t in TARGETS], dtype=float).T  # (seq_scored, 3)
        for i in range(int(row["seq_scored"])):
            key = (row["sequence"][i], row["predicted_loop_type"][i])
            sums[key] = sums.get(key, 0.0) + values[i]
            counts[key] = counts.get(key, 0) + 1
    overall = sum(sums.values()) / sum(counts.values())
    out = np.zeros((len(test_inputs), int(test_inputs["seq_scored"].iloc[0]), 3))
    for k, (_, row) in enumerate(test_inputs.iterrows()):
        for i in range(out.shape[1]):
            key = (row["sequence"][i], row["predicted_loop_type"][i])
            out[k, i] = sums[key] / counts[key] if key in counts else overall
    return out
# EVOLVE-BLOCK-END
