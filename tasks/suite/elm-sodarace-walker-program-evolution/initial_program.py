"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def make_square(wc, x0, y0, x1, y1):
    """Four corner joints of the square with opposite corners (x0, y0) and (x1, y1)."""
    return wc.add_joint(x0, y0), wc.add_joint(x0, y1), wc.add_joint(x1, y1), wc.add_joint(x1, y0)


def make_walker():
    # ELM's hand-designed square seed: passive edges plus three oscillating springs to a centre joint.
    wc = walker_creator()
    corners = make_square(wc, 0, 0, 10, 10)
    centre = wc.add_joint(5, 5)
    for k in range(len(corners) - 1):
        wc.add_muscle(corners[k], corners[k + 1])
    wc.add_muscle(corners[3], corners[0])
    wc.add_muscle(corners[3], centre)
    wc.add_muscle(corners[0], centre, 5.0, 0.0)
    wc.add_muscle(corners[1], centre, 10.0, 0.0)
    wc.add_muscle(corners[2], centre, 2.0, 0.0)
    return wc.get_walker()
# EVOLVE-BLOCK-END
