"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def _passages(documents):
    return "\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(documents))


def _last_line(text):
    lines = [line.strip().strip('"') for line in str(text).strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def solve(claim, search, llm):
    """GEPA's HoverMultiHop structure: retrieve, summarise, write the next query; three hops of 7, 7 and 10 documents."""
    hop1 = search(claim, 7)
    summary1 = llm(f"Claim: {claim}\n\nPassages:\n{_passages(hop1)}\n\nSummarise briefly what these passages say about the "
                   "entities in the claim and what information is still missing to verify it.", max_tokens=1024)
    query2 = _last_line(llm(f"Claim: {claim}\n\nWhat we know so far: {summary1}\n\nWrite one short Wikipedia search query that "
                            "finds a document still needed to verify the claim. Reply with the query only.", max_tokens=256)) or claim
    hop2 = search(query2, 7)
    summary2 = llm(f"Claim: {claim}\n\nEarlier summary: {summary1}\n\nNew passages:\n{_passages(hop2)}\n\nUpdate the summary: "
                   "what do we now know, and which entity or document is still missing?", max_tokens=1024)
    query3 = _last_line(llm(f"Claim: {claim}\n\nSummary 1: {summary1}\n\nSummary 2: {summary2}\n\nWrite one short Wikipedia "
                            "search query for the remaining missing document. Reply with the query only.", max_tokens=256)) or claim
    hop3 = search(query3, 10)
    return hop1 + hop2 + hop3
# EVOLVE-BLOCK-END
