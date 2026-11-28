"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Seed for spectrum-based fault localisation: the Tarantula risk formula (Jones and Harrold, 2005).

Suspiciousness of a block = failed ratio / (failed ratio + passed ratio), where the failed (passed) ratio is the
share of failing (passing) tests that reach the block; blocks reached by no failing test score 0.
"""


def localize(spectrum):
    failed, passed = spectrum["failed"], spectrum["passed"]
    scores = []
    for ef, ep in zip(spectrum["ef"], spectrum["ep"]):
        if ef == 0:
            scores.append(0.0)
            continue
        fail_ratio = ef / failed
        pass_ratio = ep / passed if passed else 0.0
        scores.append(fail_ratio / (fail_ratio + pass_ratio))
    return scores
# EVOLVE-BLOCK-END
