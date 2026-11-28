"""Fuzzing and mutation subjects: mutant generation, in-sandbox loading, probes, mutation analysis and input checks."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite/_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import software_data, software_mutate as mutate, software_pymod as pm, software_sandbox  # noqa: E402
from autoresearch_suite.runtime import CandidateError, load_program  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

PREPARED = all((software_data.data_dir() / f"pymod/{kind}-{name}.json").is_file()
               for kind, names in (("fuzz", pm.FUZZ), ("mutation", pm.MUTATION)) for name in names)
TINY = '''"""Module docstring."""
LIMIT = 3


class Box:
    def size(self, x):
        """Doc."""
        if x < LIMIT and not x == 0:
            return x * 2
        return -x


def clamp(values):
    out = []
    for v in values:
        if v > 10:
            break
        out.append(v + 0)
    return out
'''


def harness(entry, args):
    return software_sandbox.execute(software_sandbox.harness_source(pm.HARNESS), entry, args, cpu_seconds=60)


class Mutants(unittest.TestCase):
    def test_operators_scope_and_trivial_equivalence(self):
        found = mutate.mutants(TINY)
        operators = {m["operator"] for m in found}
        self.assertEqual(operators, {"ROR", "UOD", "AOR", "CRP", "BCR", "LCR"})
        self.assertFalse(any("XXDoc.XX" in m["replacement"] for m in found))  # docstrings are not mutated
        self.assertFalse(any(m["unit"] == "LIMIT" for m in found))  # module-level code is not mutated
        # v + 0 -> v - 0 compiles to different code, but deleting the unary minus is a real change
        self.assertTrue(any(m["operator"] == "UOD" and "return x" in m["replacement"] for m in found))
        for m in found:
            lines = TINY.splitlines(keepends=True)
            compile("".join(lines[:m["start"] - 1]) + m["replacement"] + "".join(lines[m["end"]:]), "<m>", "exec")
        self.assertEqual(found, mutate.mutants(TINY))

    def test_methods_keep_their_indentation(self):
        method = [m for m in mutate.mutants(TINY) if m["unit"] == "Box.size"][0]
        self.assertTrue(method["replacement"].startswith("    def size"))


class Harness(unittest.TestCase):
    files = {"pkg/__init__.py": "from .core import parse\n",
             "pkg/core.py": "def parse(text):\n    if text.startswith('a'):\n        return 1\n    elif text:\n        return 2 if len(text) > 3 else 3\n    raise ValueError(text)\n"}

    def test_probes_count_blocks_and_hits(self):
        source = software_sandbox.harness_source(pm.HARNESS) + (
            "\nFUZZ_DRIVERS['pkg'] = lambda imp: imp('pkg').parse\n")
        run = lambda inputs: software_sandbox.execute(source, "coverage", [self.files, ["pkg/core.py"], "pkg", inputs], cpu_seconds=30)
        empty = run([])
        self.assertEqual((empty["hits"], empty["probes"]), ([], 7))  # entry, if/else x2 (elif), ternary x2
        self.assertEqual(len(run(["a"])["hits"]), 2)
        self.assertEqual(len(run(["a", "bcdef", "b", ""])["hits"]), 7)

    def test_mutation_analysis_kills_by_outcome(self):
        source = software_sandbox.harness_source(pm.HARNESS) + (
            "\nMUTATION_DRIVERS['pkg'] = lambda imp, call, args: (lambda: imp('pkg').parse(*args))\n")
        mutants = [[4, 4, "    elif not text:\n"], [5, 5, "        return 2 if len(text) > 4 else 3\n"]]
        tests = [{"call": "parse", "args": ["bcdef"]}, {"call": "parse", "args": ["bcde"]}]
        result = software_sandbox.execute(source, "mutation_analysis", [self.files, "pkg/core.py", "pkg", tests, mutants], cpu_seconds=30)
        self.assertEqual(result["killed"], [0, 1])
        weak = software_sandbox.execute(source, "mutation_analysis", [self.files, "pkg/core.py", "pkg", tests[:1], mutants], cpu_seconds=30)
        self.assertEqual(weak["killed"], [0, -1])


class InputChecks(unittest.TestCase):
    def test_valid_and_invalid_tests(self):
        good = [{"call": "wrap", "args": ["a b c", 3, {"max_lines": None, "placeholder": "~"}]}, {"call": "dedent", "args": ["  x"]}]
        self.assertEqual(pm.check_tests("textwrap", good, 30), good)
        bad = [[], [{"call": "nope", "args": []}], [{"call": "dedent", "args": ["x", "y"]}],
               [{"call": "wrap", "args": ["a", 3, {"width": 4}]}], [{"call": "wrap", "args": ["a", True, {}]}],
               [{"call": "dedent", "args": ["x" * 501]}], good * 16]
        for tests in bad:
            with self.subTest(tests=str(tests)[:60]), self.assertRaises(CandidateError):
                pm.check_tests("textwrap", tests, 30)

    def test_fraction_inputs_are_bounded(self):
        ok = [{"call": "binary", "args": ["1/3", "**", 50]}, {"call": "format", "args": ["2.5e10", ".3f"]}]
        self.assertEqual(pm.check_tests("fractions", ok, 30), ok)
        for args in (["1e999", "**", 2], ["1/3", "**", 5000], ["1/3", "+", "1" * 31]):
            with self.subTest(args=args), self.assertRaises(CandidateError):
                pm.check_tests("fractions", [{"call": "binary", "args": args}], 30)
        with self.assertRaises(CandidateError):
            pm.check_tests("fractions", [{"call": "format", "args": ["1", "10000"]}], 30)


@unittest.skipUnless(PREPARED, "run scripts/prepare_software.py --only pymod")
class PreparedSubjects(unittest.TestCase):
    def candidate(self, name, source):
        path = Path(tempfile.mkdtemp()) / name
        path.write_text(source)
        return load_program(path)

    def test_mutant_counts_and_seed_scores(self):
        counts = {name: len(software_data.load_json(f"pymod/mutation-{name}.json")["mutants"]) for name in pm.MUTATION}
        self.assertEqual(counts, {"textwrap": 132, "urllib.parse": 396, "fractions": 453, "statistics": 583})
        seed = seed_source("science-software-mutation-textwrap")
        score, details = pm.mutation_score(self.candidate("seed.py", seed), "textwrap")
        self.assertEqual(details["usable_tests"], 30)
        self.assertTrue(0.2 < score < 0.6)

    def test_fuzz_seed_and_invalid_corpora(self):
        seed = seed_source("science-software-fuzz-tomllib")
        score, details = pm.fuzz_score(self.candidate("seed.py", seed), "tomllib")
        self.assertTrue(50 < score < details["probes"])
        for name, source in {"long.py": "def fuzz(target):\n    return ['x' * 2001]\n",
                             "many.py": "def fuzz(target):\n    return ['x'] * 1001\n",
                             "bytes.py": "def fuzz(target):\n    return [[1, 2]]\n"}.items():
            with self.subTest(name), self.assertRaises(CandidateError):
                pm.fuzz_score(self.candidate(name, source), "tomllib")


if __name__ == "__main__":
    unittest.main()
