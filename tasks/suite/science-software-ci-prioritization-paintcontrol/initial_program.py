"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Seed for the CI test-prioritisation tasks: tests that failed in their latest previous execution come first.

Tests never executed before come next, tests that passed last time follow; ties keep ascending id order.
"""

LAST = {}


def prioritize(cycle, history):
    if history:
        for test in history[-1]["tests"]:
            LAST[test["id"]] = 1.0 if test["failed"] else 0.0
    ids = [test["id"] for test in cycle["tests"]]
    return sorted(ids, key=lambda t: (-LAST.get(t, 0.5), t))
# EVOLVE-BLOCK-END
