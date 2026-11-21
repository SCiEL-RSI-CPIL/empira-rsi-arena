"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """A single quarter-wave layer of the low-index material centred in the band."""
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    return ["L"], [centre / (4.0 * problem["materials"]["L"])]
# EVOLVE-BLOCK-END
