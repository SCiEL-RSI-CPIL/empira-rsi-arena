"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
WINDOW = 4


import re


def _squash(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _pick(reply, valid_actions):
    text = reply.strip().splitlines()[0].strip().strip("`'\".* ") if reply.strip() else ""
    if text.lower().startswith("action:"):
        text = text[7:].strip()
    for action in valid_actions:
        if _squash(text) == _squash(action):
            return action
    for action in sorted(valid_actions, key=len, reverse=True):
        if _squash(action) in _squash(text):
            return action
    return text or (valid_actions[0] if valid_actions else "look")


def act(instruction, observation, valid_actions, history, llm):
    """Single-call agent: the instruction, a short window of recent steps and the current observation; one action back."""
    lines = []
    for action, seen, _ in history[-WINDOW:]:
        lines.append("Action: " + action)
        lines.append("Observation: " + seen[:1500])
    prompt = instruction + "\n\n"
    if lines:
        prompt += "Recent steps:\n" + "\n".join(lines) + "\n\n"
    prompt += "Current observation:\n" + observation[:4000] + "\n\n"
    if valid_actions:
        prompt += "Valid actions: " + ", ".join(valid_actions) + "\n"
    prompt += "Reply with the next action only, with no other text."
    return _pick(llm(prompt, max_tokens=60), valid_actions)
# EVOLVE-BLOCK-END
