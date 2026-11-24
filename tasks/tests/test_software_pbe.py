"""Programming by example: SyGuS parsing, grammar conformance, string semantics against Z3 and the prepared tasks."""
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
from autoresearch_suite import software_data, software_pbe as pbe  # noqa: E402
from autoresearch_suite.runtime import CandidateError, load_program  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

HAVE_Z3 = importlib.util.find_spec("z3") is not None
PREPARED = (software_data.data_dir() / "pbe/tasks.json").is_file()
TINY = '''; comment
(set-logic SLIA)
(synth-fun f ((name String)) String
    ((Start String (ntString))
     (ntString String (name " " "say ""hi"""
                       (str.++ ntString ntString)
                       (str.substr ntString ntInt ntInt)))
     (ntInt Int (0 1 -1 (+ ntInt ntInt) (str.len ntString) (str.indexof ntString ntString ntInt)))))
(declare-var name String)
(constraint (= (f "ab cd") "ab"))
(constraint (= (f "x y")
               "x"))
(check-synth)
'''


class Parsing(unittest.TestCase):
    def test_benchmark_and_strings(self):
        function, examples = pbe.parse_benchmark(TINY)
        self.assertEqual(function["params"], [["name", "String"]])
        self.assertEqual(examples, [[["ab cd"], "ab"], [["x y"], "x"]])
        self.assertIn(("str", 'say "hi"'), function["grammar"]["ntString"][1])
        self.assertIn(-1, function["grammar"]["ntInt"][1])

    def test_conformance(self):
        function, _ = pbe.parse_benchmark(TINY)
        grammar = function["grammar"]
        ok = pbe.sexp('(str.substr name 0 (str.indexof name " " 0))')[0]
        self.assertTrue(pbe.conforms(ok, "Start", grammar))
        for text in ['(str.at name 0)', '(str.substr name 0 2 3)', '(str.substr name "a" 1)', '(str.++ name "x")', "name2", "5"]:
            with self.subTest(text=text):
                self.assertFalse(pbe.conforms(pbe.sexp(text)[0], "Start", grammar))

    def test_semantics_edge_cases(self):
        cases = {'(str.indexof "abc" "" 3)': 3, '(str.indexof "abc" "" 4)': -1, '(str.replace "abc" "" "x")': "xabc",
                 '(str.substr "abc" (- 1) 2)': "", '(str.at "abc" 3)': "", '(str.to.int "0012")': 12, '(str.to.int "")': -1,
                 '(str.to.int "-3")': -1, '(int.to.str (- 2))': "", '(str.contains "abc" "")': True, '(str.prefixof "ab" "abc")': True}
        for text, value in cases.items():
            with self.subTest(text=text):
                self.assertEqual(pbe.evaluate(pbe.sexp(text)[0], {}), value)
        with self.assertRaises(pbe.EvalError):
            deep = "name"
            for _ in range(20):
                deep = f"(str.++ {deep} {deep})"
            pbe.evaluate(pbe.sexp(deep)[0], {"name": "abcdefgh"})

    @unittest.skipUnless(HAVE_Z3, "needs z3-solver")
    def test_interpreter_matches_z3_on_random_terms(self):
        import z3
        rng = random.Random(3)
        strings = ["", "a", "ab-cd", "x y z", "12", "007", "a.b.c", "--", "Hello World"]
        ints = [-2, -1, 0, 1, 2, 3, 5, 9]

        def term(sort, depth):
            if depth == 0 or rng.random() < 0.3:
                return '"' + rng.choice(strings) + '"' if sort == "String" else (lambda v: str(v) if v >= 0 else f"(- {-v})")(rng.choice(ints))
            if sort == "String":
                op = rng.choice(["str.++", "str.replace", "str.at", "int.to.str", "str.substr"])
                args = {"str.++": ["String", "String"], "str.replace": ["String"] * 3, "str.at": ["String", "Int"],
                        "int.to.str": ["Int"], "str.substr": ["String", "Int", "Int"]}[op]
            else:
                op = rng.choice(["+", "-", "str.len", "str.to.int", "str.indexof"])
                args = {"+": ["Int", "Int"], "-": ["Int", "Int"], "str.len": ["String"], "str.to.int": ["String"],
                        "str.indexof": ["String", "String", "Int"]}[op]
            return "(" + " ".join([op] + [term(a, depth - 1) for a in args]) + ")"
        checked = 0
        for _ in range(400):
            sort = rng.choice(["String", "Int"])
            text = term(sort, 3)
            solver = z3.Solver()
            solver.from_string(f"(declare-fun out () {sort}) (assert (= out {text}))")
            self.assertEqual(solver.check(), z3.sat)
            out = solver.model()[z3.String("out") if sort == "String" else z3.Int("out")]
            expected = out.as_string() if sort == "String" else out.as_long()
            self.assertEqual(pbe.evaluate(pbe.sexp(text)[0], {}), expected, text)
            checked += 1
        self.assertEqual(checked, 400)


@unittest.skipUnless(PREPARED, "run scripts/prepare_software.py --only pbe")
class PreparedTasks(unittest.TestCase):
    def candidate(self, name, source):
        path = Path(tempfile.mkdtemp()) / name
        path.write_text(source)
        return load_program(path)

    def test_tasks_and_seed(self):
        tasks = software_data.load_json("pbe/tasks.json")
        self.assertEqual(len(tasks), 27)
        self.assertTrue(all(t["examples"] and t["held_out"] for t in tasks))
        seed = seed_source("science-software-pbe-strings-sygus")
        score, details = pbe.pbe_score(self.candidate("seed.py", seed))
        self.assertEqual(details["solved"], 8)
        identity = "def synthesize(task):\n    return task['params'][0][0]\n"
        self.assertEqual(pbe.pbe_score(self.candidate("identity.py", identity))[0], 0.0)

    def test_invalid_candidates(self):
        cases = {"number.py": "def synthesize(task):\n    return 3\n",
                 "short.py": "def synthesize(task):\n    raise ValueError('none')\n",
                 "missing.py": "def solve(task):\n    return ''\n"}
        for name, source in cases.items():
            with self.subTest(name), self.assertRaises(CandidateError):
                pbe.pbe_score(self.candidate(name, source))
        rejected = pbe.pbe_score(self.candidate("bad.py", "def synthesize(task):\n    return '(str.at name'\n"))[1]
        self.assertEqual(rejected["rejected"], 27)


if __name__ == "__main__":
    unittest.main()
