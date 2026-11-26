"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def solve(question, llm):
    """Single-pass planning: the prompt already states the required plan format."""
    return llm(question, max_tokens=8192)
# EVOLVE-BLOCK-END
