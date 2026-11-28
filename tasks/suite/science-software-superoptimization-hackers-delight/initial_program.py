"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Seed for the superoptimisation task: return the reference program after dead-code elimination.

The references contain no dead code, so the seed scores the reference total; better programs need search
(enumeration or stochastic rewriting, checked on test inputs before they are returned).
"""


def optimize(problem):
    program = [list(instruction) for instruction in problem["reference"]]
    live = {program[-1][0]}
    kept = []
    for instruction in reversed(program):
        if instruction[0] in live:
            kept.append(instruction)
            live.update(a for a in instruction[2:] if isinstance(a, str))
    return kept[::-1]
# EVOLVE-BLOCK-END
