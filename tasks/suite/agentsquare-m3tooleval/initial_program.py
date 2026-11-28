"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def act(instruction, observation, valid_actions, history, llm):
    """Chat-format tool-use agent (CodeAct's M3ToolEval loop): the task prompt, then alternating replies and tool results."""
    messages = [{"role": "user", "content": instruction}]
    for action, seen, _ in history:
        messages.append({"role": "assistant", "content": action})
        messages.append({"role": "user", "content": seen})
    return llm(messages, max_tokens=512)
# EVOLVE-BLOCK-END
