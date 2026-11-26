"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import re


def _clean(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def solve(user_query, llm, untrusted_llm):
    """PAPILLON-style pipeline: redact with the trusted model, ask the untrusted model, answer with the trusted model."""
    request = _clean(llm("Rewrite the user's request for an external assistant: keep the task, but remove names, contact details and other "
                         "personal information. Return only the rewritten request. /no_think\n\nUser request:\n" + user_query, max_tokens=1024))
    external = untrusted_llm(request, max_tokens=2048)
    return _clean(llm("Respond to the user's request. A related request was answered by an external assistant; use it if helpful.\n\n"
                      "User request:\n" + user_query + "\n\nRelated request:\n" + request + "\n\nExternal answer:\n" + external
                      + "\n\nWrite only the final response. /no_think", max_tokens=2048))
# EVOLVE-BLOCK-END
