"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def script():
    # ABC commands run after `read <circuit>`; the network they leave must consist of LUTs with at most 6 inputs.
    # A plain mapping to 3-input LUTs: a valid starting point that leaves room for improvement.
    return "strash; if -K 3"
# EVOLVE-BLOCK-END
