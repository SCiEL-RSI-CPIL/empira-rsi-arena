"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import re

FENCED_FILE = re.compile(r"^(?P<path>[^\n`]*?)[ \t]*\n```[^\n]*\n(?P<body>.*?)^```", re.M | re.S)


def _prompt(exercise):
    lines = [exercise["instructions"].strip(), "",
             "Implement this " + exercise["language"] + " exercise by editing the files below. Keep the names and signatures "
             "that the stubs declare, because hidden unit tests use them, and use only the language's standard library.",
             "Reply with the complete new content of every file you change: the file path on its own line, then the whole "
             "file in one fenced code block."]
    for path, text in exercise["files"].items():
        lines += ["", path, "```", text.rstrip("\n"), "```"]
    for path, text in exercise.get("context", {}).items():
        lines += ["", path + " (read-only)", "```", text.rstrip("\n"), "```"]
    return "\n".join(lines)


def _match_path(label, paths):
    label = label.strip().strip("*`#:").strip()
    return next((path for path in paths if label == path or label.endswith(path)), None)


def solve(exercise, llm):
    """One model call per exercise that rewrites the stub files."""
    reply = llm(_prompt(exercise), max_tokens=8192)
    edits = {}
    for match in FENCED_FILE.finditer(reply):
        path = _match_path(match.group("path"), exercise["files"])
        if path:
            edits[path] = match.group("body")
    if not edits:
        blocks = re.findall(r"```[^\n]*\n(.*?)```", reply, flags=re.S)
        editable = [path for path in exercise["files"] if not path.endswith("Cargo.toml")]
        if blocks and len(editable) == 1:
            edits[editable[0]] = blocks[-1]
    return edits
# EVOLVE-BLOCK-END
