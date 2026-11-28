"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def design(problem):
    """A centred dielectric rod of radius 0.2 lattice constants (the textbook TM band-gap crystal)."""
    size = problem["pixels"]
    cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
    return cell
# EVOLVE-BLOCK-END
