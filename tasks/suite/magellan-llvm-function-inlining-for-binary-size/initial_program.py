"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def should_inline(features):
    # Naive starting policy, as in Magellan's search: never inline (LLVM still performs mandatory inlining).
    return False
# EVOLVE-BLOCK-END
