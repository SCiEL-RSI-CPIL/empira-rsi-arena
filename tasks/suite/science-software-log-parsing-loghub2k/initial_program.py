"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Seed for log parsing: group messages by their token sequence with digit-bearing tokens masked.

Every whitespace-separated token that contains a digit becomes the wildcard <*>; messages with the same masked token
sequence form one group. Real parsers (Drain, IPLoM, Spell) also merge sequences that differ in variable words.
"""


def parse(dataset):
    groups, labels = {}, []
    for message in dataset["messages"]:
        key = " ".join("<*>" if any(ch.isdigit() for ch in token) else token for token in message.split())
        labels.append(groups.setdefault(key, len(groups)))
    return labels
# EVOLVE-BLOCK-END
