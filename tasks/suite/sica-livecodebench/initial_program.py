"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def solve(question, llm):
    """One-shot code generation; the evaluator executes the last ```python block of the reply."""
    return llm(question, max_tokens=8192)
# EVOLVE-BLOCK-END
