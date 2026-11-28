"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def solve(question, llm):
    """Answer the request directly; the whole reply is checked against its output constraints."""
    system = "Follow every formatting and content constraint in the request exactly. Reply with the response only."
    return llm(question, system=system, max_tokens=4096)
# EVOLVE-BLOCK-END
