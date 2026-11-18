"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def script():
    # ABC commands run after `read <circuit>`; the network they leave must consist of LUTs with at most 6 inputs.
    return "strash; dc2; if -K 6"
# EVOLVE-BLOCK-END
