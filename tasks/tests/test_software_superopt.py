"""Superoptimisation: IR semantics against Z3, C parsing, program checks, equivalence proofs and the prepared kernels."""
from __future__ import annotations

import importlib.util
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite/_lib"))
sys.path.insert(0, str(ROOT / "tests"))
from autoresearch_suite import software_data, software_superopt as so  # noqa: E402
from autoresearch_suite.runtime import CandidateError, load_program  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

HAVE_Z3 = importlib.util.find_spec("z3") is not None
PREPARED = (software_data.data_dir() / "superopt/problems.json").is_file()
C_SOURCE = """// comment
#include <stdint.h>
int32_t p99(int32_t x, int32_t y) {
  uint32_t o1 = (uint32_t) x >> 3;
  int32_t o2 = x >> 3;
  int32_t o3 = o1 < (uint32_t) y ? -o2 : ~y;
  return !o3 && (x / y);
}
"""


class Semantics(unittest.TestCase):
    @unittest.skipUnless(HAVE_Z3, "needs z3-solver")
    def test_interpreter_matches_z3_for_every_operation(self):
        import z3
        rng = random.Random(5)
        values = so.SPECIAL + [rng.getrandbits(32) for _ in range(20)]
        for op, arity in so.ARITY.items():
            names = ["a", "b", "c"][:arity]
            for _ in range(60):
                args = [rng.choice(values) for _ in names]
                program = [["r", op, *names]]
                expected = z3.simplify(so.z3_term(program, names, [z3.BitVecVal(v, 32) for v in args])).as_long()
                self.assertEqual(so.run(program, names, args), expected, (op, args))

    def test_division_and_shift_edge_cases(self):
        self.assertEqual(so.OPS["udiv"](7, 0), so.MASK)
        self.assertEqual(so.OPS["sdiv"](so.MASK, 0), 1)  # -1 / 0
        self.assertEqual(so.OPS["srem"](so.MASK - 6, 0), so.MASK - 6)
        self.assertEqual(so.OPS["ashr"](0x80000000, 40), so.MASK)
        self.assertEqual(so.OPS["shl"](1, 32), 0)

    def test_c_parsing_follows_c_types(self):
        problem = so.parse_c("p99", C_SOURCE)
        ops = [i[1] for i in problem["reference"]]
        self.assertEqual(problem["inputs"], ["x", "y"])
        self.assertEqual(ops[:3], ["lshr", "ashr", "ult"])
        self.assertIn("select", ops)
        self.assertIn("sdiv", ops)
        for x, y in ((5, 3), (-40 & so.MASK, 7), (123456, so.MASK)):
            o1, o2 = x >> 3, so.OPS["ashr"](x, 3)
            o3 = (-o2 & so.MASK) if o1 < y else (~y & so.MASK)
            expected = int(o3 == 0 and so.OPS["sdiv"](x, y) != 0)
            self.assertEqual(so.run(problem["reference"], ["x", "y"], [x, y]), expected)


class Programs(unittest.TestCase):
    def test_structural_checks(self):
        good = [["a", "sub", "x", 1], ["r", "and", "x", "a"]]
        self.assertEqual(so.check_program(good, ["x"]), good)
        bad = [[], [["a", "foo", "x", 1]], [["a", "sub", "x"]], [["x", "neg", "x"]], [["a", "add", "b", 1]],
               [["a", "add", "x", 2 ** 32]], [["a", "add", "x", 1.0]], [["a", "add", "x", True]], [["1a", "neg", "x"]],
               [["a", "neg", "x"]] * 2, [["t%d" % i, "neg", "x"] for i in range(65)], "program"]
        for program in bad:
            with self.subTest(program=str(program)[:50]), self.assertRaises(CandidateError):
                so.check_program(program, ["x"])

    @unittest.skipUnless(HAVE_Z3, "needs z3-solver")
    def test_equivalence(self):
        reference = [["a", "sub", "x", 1], ["b", "and", "a", "x"], ["c", "eq", "b", 0], ["d", "ne", "x", 0], ["r", "and", "c", "d"]]
        shorter = [["a", "sub", "x", 1], ["b", "xor", "x", "a"], ["r", "ugt", "b", "a"]]
        self.assertEqual(so.equivalent(reference, shorter, ["x"]), "proven")
        self.assertEqual(so.equivalent(reference, [["r", "eq", "x", 1]], ["x"]), "counterexample")
        # differs only at x = 0x40000000, which random tests rarely hit; the proof finds it
        sneaky = reference[:-1] + [["e", "ne", "x", 0x40000000], ["f", "and", "c", "d"], ["r", "and", "f", "e"]]
        self.assertEqual(so.equivalent(reference, sneaky, ["x"], tests=10), "counterexample")


@unittest.skipUnless(HAVE_Z3 and PREPARED, "needs z3-solver and scripts/prepare_software.py --only superopt")
class PreparedKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems = {p["name"]: p for p in software_data.load_json("superopt/problems.json")}

    def test_kernels_compute_the_named_functions(self):
        self.assertEqual(sum(len(p["reference"]) for p in self.problems.values()), 135)
        rng = random.Random(11)
        for _ in range(2000):
            x = rng.getrandbits(32) >> rng.randint(0, 31)
            self.assertEqual(so.run(self.problems["p23"]["reference"], ["x"], [x]), bin(x).count("1"))
            self.assertEqual(so.run(self.problems["p18"]["reference"], ["x"], [x]), int(x != 0 and x & (x - 1) == 0))
            self.assertEqual(so.run(self.problems["p13"]["reference"], ["x"], [x]), (x != 0) * (1 if x < 2 ** 31 else so.MASK))
            if 0 < x <= 2 ** 31:
                self.assertEqual(so.run(self.problems["p24"]["reference"], ["x"], [x]), 1 << (x - 1).bit_length())

    def test_scores(self):
        directory = Path(tempfile.mkdtemp())
        (directory / "seed.py").write_text(seed_source("science-software-superoptimization-hackers-delight"))
        self.assertEqual(so.superopt_score(load_program(directory / "seed.py"))[0], 135.0)
        better = ("BETTER = {'p16': [['c', 'uge', 'x', 'y'], ['r', 'select', 'c', 'x', 'y']],\n"
                  "          'p18': [['a', 'sub', 'x', 1], ['b', 'xor', 'x', 'a'], ['r', 'ugt', 'b', 'a']],\n"
                  "          'p01': [['r', 'sub', 'x', 1]]}\n"
                  "def optimize(problem):\n    return BETTER.get(problem['name'], problem['reference'])\n")
        (directory / "better.py").write_text(better)
        score, details = so.superopt_score(load_program(directory / "better.py"))
        self.assertEqual(score, 135 - 3 - 7)
        self.assertEqual(details["per_kernel"]["p01"]["status"], "counterexample")
        cases = {"malformed.py": "def optimize(problem):\n    return [['r', 'popcnt', 'x']]\n",
                 "short.py": "def optimize(problem):\n    return []\n",
                 "missing.py": "def superoptimize(problem):\n    return []\n"}
        for name, source in cases.items():
            (directory / name).write_text(source)
            with self.subTest(name), self.assertRaises(CandidateError):
                so.superopt_score(load_program(directory / name))


if __name__ == "__main__":
    unittest.main()
