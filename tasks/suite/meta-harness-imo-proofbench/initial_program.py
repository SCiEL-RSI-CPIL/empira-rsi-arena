"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def solve(question, llm):
    """Single-pass proof writing."""
    system = ("You are an expert olympiad mathematician. Write a complete and rigorous but concise proof (at most about 1,200 words) "
              "and justify every step.")
    return llm(question, system=system, max_tokens=8192)
# EVOLVE-BLOCK-END
