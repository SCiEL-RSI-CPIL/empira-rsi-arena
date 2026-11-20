"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
"""Seed for the mutation-based test-generation tasks: random calls drawn from the API description.

Calls cycle through the API functions; arguments are random values of the described kinds (no mutant feedback).
"""
import random
import string

WORDS = ["a", "ab", "abc", "x y", "Hello world.", "", " ", "-", "1/3", "0.5", "http://a/b?c=d#e", "%20", "+"]


def _value(kind, rng):
    if isinstance(kind, dict):
        (name, spec), = kind.items()
        if name == "options":
            key = rng.choice(sorted(spec))
            return {key: _value(spec[key], rng)}
        if name == "strings":
            return [_value("str", rng) for _ in range(spec)]
        return rng.choice(spec)
    if kind == "str":
        return rng.choice(WORDS) if rng.random() < 0.6 else "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 20)))
    if kind in ("int", "opt_int"):
        return rng.randint(-2, 40)
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "float":
        return rng.uniform(-10, 10)
    if kind == "fraction":
        return rng.choice(["1/3", "0.5", "-2", "3e2", "7", "22/7", "0"])
    if kind == "spec":
        return rng.choice(["", ".2f", "10", ">8", "e", "%", "g"])
    if kind == "number":
        return rng.choice([rng.randint(-5, 5), rng.choice(["1/3", "0.5", "-2"]), 0.25])
    if kind == "numbers":
        return [rng.choice([rng.randint(-5, 9), "1/3", 2.5]) for _ in range(rng.randint(0, 6))]
    if kind == "pairs":
        return [[_value("str", rng), _value("str", rng)] for _ in range(rng.randint(0, 3))]
    raise ValueError(kind)


def generate_tests(subject):
    rng = random.Random(0)
    calls = sorted(subject["api"])
    tests = []
    for i in range(subject["max_tests"]):
        call = calls[i % len(calls)]
        tests.append({"call": call, "args": [_value(kind, rng) for kind in subject["api"][call]]})
    return tests
# EVOLVE-BLOCK-END
