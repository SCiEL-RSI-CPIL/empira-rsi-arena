"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Seed for the coverage-guided fuzzing tasks: random printable strings (no coverage feedback).

The corpus mixes short and long strings over printable ASCII plus newline, drawn from a seeded generator.
"""
import random
import string

ALPHABET = string.printable


def fuzz(target):
    rng = random.Random(0)
    corpus = []
    for _ in range(min(300, target["max_inputs"])):
        length = rng.choice([1, 2, 4, 8, 16, 32, 64, 128])
        corpus.append("".join(rng.choice(ALPHABET) for _ in range(min(length, target["max_chars"]))))
    return corpus
# EVOLVE-BLOCK-END
