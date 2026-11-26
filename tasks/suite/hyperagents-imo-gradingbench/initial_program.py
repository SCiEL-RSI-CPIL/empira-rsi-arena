"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def solve(question, llm):
    """Zero-shot chain of thought; the evaluator reads the text after the last 'Answer:' marker."""
    prompt = question + "\n\nThink step by step but keep the reasoning brief (at most about 300 words), then finish with a final line of the form 'Answer: <answer>'."
    return llm(prompt, max_tokens=4096)
# EVOLVE-BLOCK-END
