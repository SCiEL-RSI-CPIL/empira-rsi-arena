"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def solve(model):
    # Baseline: stop HiGHS at its first feasible solution and return it (zeros clipped to the bounds if it found none).
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", 30.0)
    h.setOptionValue("mip_max_improving_sols", 1)
    h.readModel(model["mps_path"])
    h.run()
    x = np.array(h.getSolution().col_value, dtype=float)
    if len(x) != model["num_cols"]:
        x = np.zeros(model["num_cols"])
    return np.clip(x, model["col_lower"], model["col_upper"])
# EVOLVE-BLOCK-END
