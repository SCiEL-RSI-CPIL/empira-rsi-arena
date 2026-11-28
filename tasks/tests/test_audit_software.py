"""Software-engineering evaluators: limits that do not depend on the machine, resumable preparation, hidden answers."""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "suite/_lib")]
from autoresearch_suite import software_data, software_harness_pymod as pymod_harness  # noqa: E402
from autoresearch_suite import software_harness_quixbugs as quixbugs_harness, software_quixbugs as quixbugs  # noqa: E402
from autoresearch_suite import software_pymod, software_sandbox, software_sbfl as sbfl, software_superopt  # noqa: E402
from autoresearch_suite.runtime import CandidateError, load_program  # noqa: E402

LOOP = "def f(n):\n    i = 0\n    while i < n:\n        i += 1\n    return i\n"
# Twenty microseconds of C-level work per iteration, so the time of a call is far from its step count.
SLOW = "def f(n):\n    s = 0\n    for i in range(n):\n        s += sum(range(2000)) * 0\n    return s\n"
DRIVER = "\nMUTATION_DRIVERS['pkg'] = lambda imp, call, args: (lambda: imp('pkg').f(*args))\n"
# Sets the limit of the original call just above its own run time: the situation of a slow machine.
TIGHT = DRIVER + '''
def _tight(files, tests, mutants):
    global CPU_SECONDS
    start = time.process_time()
    outcomes(files, "pkg", tests)
    CPU_SECONDS = 2 * max(time.process_time() - start, 0.02)
    return mutation_analysis(files, "pkg.py", "pkg", tests, mutants)
'''


def harness(entry, args, extra=DRIVER):
    return software_sandbox.execute(software_sandbox.harness_source(software_pymod.HARNESS) + extra, entry, args, cpu_seconds=30)


def candidate(source, name="candidate.py"):
    path = Path(tempfile.mkdtemp()) / name
    path.write_text(source)
    return load_program(path, name[:-3])


class ModuleStepBudget(unittest.TestCase):
    def test_loops_are_cut_by_their_step_count(self):
        limit = pymod_harness.STEP_LIMIT
        tests = [{"call": "f", "args": [limit - 2]}, {"call": "f", "args": [limit]}]  # one step for the entry
        first, second = harness("outcomes", [{"pkg.py": LOOP}, "pkg", tests])
        self.assertEqual((first[0], second), ("ok", ["timeout"]))
        endless = LOOP.replace("i < n", "i < n or True")
        result = harness("mutation_analysis", [{"pkg.py": LOOP}, "pkg.py", "pkg", [{"call": "f", "args": [3]}], [[1, 5, endless]]])
        self.assertEqual((result["killed"], result["usable_tests"]), ([0], 1))

    def test_a_variant_that_is_only_slower_is_not_killed_on_a_slow_machine(self):
        variant = SLOW.replace("range(n)", "range(n * 4)")
        result = harness("_tight", [{"pkg.py": SLOW}, [{"call": "f", "args": [2000]}], [[1, 5, variant]]], TIGHT)
        self.assertEqual((result["usable_tests"], result["killed"]), (1, [-1]))
        different = SLOW.replace("return s", "return s + 1")
        result = harness("_tight", [{"pkg.py": SLOW}, [{"call": "f", "args": [2000]}], [[1, 5, different]]], TIGHT)
        self.assertEqual(result["killed"], [0])

    def test_a_time_signal_after_the_end_of_a_call_does_not_escape(self):
        # One long C-level operation that ends in an exception: the signal can only be handled after the call.
        late = lambda: sum(itertools.chain(range(3_000_000), ["x"]))
        for _ in range(3):
            self.assertIn(pymod_harness._run(late, 10 ** 9, 0.01), (["timeout"], ["raise", "TypeError"]))
        program = quixbugs_harness._load(quixbugs_harness._compile(
            "import itertools\ndef f(n):\n    return sum(itertools.chain(range(n), ['x']))\n"), "f", [0])
        for _ in range(3):
            self.assertEqual(quixbugs_harness._output(program, [3_000_000], [0], 10 ** 9, slack=0.001), [False, None])

    def test_fuzzing_inputs_are_cut_by_their_step_count(self):
        source = software_sandbox.harness_source(software_pymod.HARNESS) + "\nFUZZ_DRIVERS['pkg'] = lambda imp: (lambda s: imp('pkg').f(len(s)))\n"
        files = {"pkg.py": LOOP + "\n\ndef g():\n    return 1\n"}
        runs = [software_sandbox.execute(source, "coverage", [files, ["pkg.py"], "pkg", ["abc", "x" * 30], 20], cpu_seconds=30) for _ in range(2)]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual((runs[0]["timeouts"], runs[0]["probes"]), (1, 4))  # f, loop body, loop exit, g
        self.assertEqual(len(runs[0]["hits"]), 3)

    def test_try_finally_is_instrumented_and_probe_numbers_are_kept(self):
        import ast
        code = "def f(x):\n    try:\n        y = x\n    finally:\n        z = 1\n    try:\n        y = 2\n    except ValueError:\n        y = 3\n    return y\n"
        visitor = pymod_harness._Instrument()
        compile(ast.fix_missing_locations(visitor.visit(ast.parse(code))), "<t>", "exec")
        self.assertEqual(visitor.count, 6)  # entry; try body, finally; try body, else, handler
        files = {"pkg/__init__.py": "from .core import parse\n",
                 "pkg/core.py": "def parse(text):\n    if text.startswith('a'):\n        return 1\n    elif text:\n        return 2 if len(text) > 3 else 3\n    raise ValueError(text)\n"}
        source = software_sandbox.harness_source(software_pymod.HARNESS) + "\nFUZZ_DRIVERS['pkg'] = lambda imp: imp('pkg').parse\n"
        result = software_sandbox.execute(source, "coverage", [files, ["pkg/core.py"], "pkg", ["a", "bcdef", "b", ""]], cpu_seconds=30)
        self.assertEqual((result["probes"], len(result["hits"])), (7, 7))


class ProgramStepLimits(unittest.TestCase):
    COUNT = "def f(n):\n    for i in range(n):\n        pass\n    return n\n"

    def test_the_limit_is_an_exact_step_count(self):
        # f(1000) spends 1001 steps: one for the function entry and one per loop iteration.
        self.assertEqual(quixbugs_harness.outputs(self.COUNT, "f", [[1000], [1000]], [1001, 1000]), [[True, 1000], [False, None]])
        recursive = "def f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\n"
        self.assertEqual(quixbugs_harness.outputs(recursive, "f", [[50], [50]], [51, 50]), [[True, 50], [False, None]])

    def test_tests_after_a_failed_run_are_not_run_when_asked(self):
        endless = "def f(n):\n    while n:\n        pass\n    return n\n"
        self.assertEqual(quixbugs_harness.outputs(endless, "f", [[0], [1], [0]], [500] * 3), [[True, 0], [False, None], [True, 0]])
        self.assertEqual(quixbugs_harness.outputs(endless, "f", [[0], [1], [0]], [500] * 3, True), [[True, 0], [False, None], [False, None]])

    def test_only_one_kept_test_has_a_larger_limit(self):
        self.assertEqual(quixbugs.step_limit("levenshtein", ["rosettacode", "raisethysword"]), 25_000_000)
        self.assertEqual(quixbugs.step_limit("levenshtein", ["a", "b"]), quixbugs.STEP_LIMIT)
        self.assertEqual(quixbugs.step_limit("gcd", [35, 21]), 2_000_000)

    def test_judging_uses_step_limits_and_keeps_expected_values_out_of_the_child(self):
        tests = [{"input": [n], "expected": 2 * n} for n in range(8)]
        program = {"name": "double", "entry": "double", "source": "def double(x):\n    return x + 1\n",
                   "visible": tests[:2], "held_out": tests[2:4], "generated": tests[4:]}
        self.assertEqual(quixbugs.judge_patch(program, "def double(x):\n    return x * 2\n")["status"], "repaired")
        started = time.time()
        self.assertEqual(quixbugs.judge_patch(program, "def double(x):\n    while True:\n        pass\n")["status"], "unrepaired")
        self.assertLess(time.time() - started, 10)
        reader = ("def double(x):\n    import sys\n    frame = sys._getframe(1)\n    while frame is not None:\n"
                  "        for value in frame.f_locals.values():\n            if isinstance(value, dict) and 'expected' in value:\n"
                  "                return value['expected']\n        frame = frame.f_back\n    return x + 1\n")
        self.assertEqual(quixbugs.run_tests(reader, "double", tests[2:]), [False] * 6)  # x + 1 is wrong from 2 on

    def test_statements_inside_a_match_block_count_as_edits(self):
        buggy = "def gcd(a, b):\n    if b == 0:\n        return a\n    else:\n        return gcd(a % b, b)\n"
        rewrite = ("def gcd(a, b):\n    match 0:\n        case _:\n            a, b = abs(a), abs(b)\n            while b:\n"
                   "                a, b = b, a % b\n            return a\n    if b == 0:\n        return a\n    else:\n        return gcd(a % b, b)\n")
        self.assertEqual(quixbugs.statement_edits(buggy, rewrite), 6)  # match, case and four statements
        tests = [{"input": [35, 21], "expected": 7}]
        program = {"name": "gcd", "entry": "gcd", "source": buggy, "visible": tests, "held_out": tests, "generated": tests}
        self.assertEqual(quixbugs.judge_patch(program, rewrite)["status"], "rejected")
        cases = "def f(x):\n    match x:\n        case 1 if x:\n            return 1\n        case _:\n            return 2\n"
        self.assertEqual(quixbugs.statement_edits(cases, cases.replace("return 2", "return 3")), 1)
        self.assertEqual(quixbugs.statement_edits(cases, cases.replace("case 1 if x", "case 2 if x")), 1)
        star = "def f(x):\n    try:\n        y = 1\n    except* ValueError:\n        y = 2\n    return y\n"
        self.assertEqual(quixbugs.statement_edits(star, star.replace("y = 2", "y = 3")), 1)

    def test_the_fast_tracer_records_the_lines_of_plain_tracing(self):
        source = ("def helper(k):\n    if k % 3 == 0:\n        return 1\n    return 0\n\n\ndef f(n):\n    total = 0\n"
                  "    for i in range(n):\n        total += helper(i)\n    if n > 100:\n        total = -1\n    return total\n")
        tests = [{"input": [0], "expected": 0}, {"input": [7], "expected": 3}, {"input": [500], "expected": -1}]
        function = quixbugs_harness._load(quixbugs_harness._compile(source), "f", [10 ** 9])

        def plain(n):
            lines = set()

            def tracer(frame, event, arg):
                if frame.f_code.co_filename == "<program>":
                    if event in ("line", "call"):
                        lines.add(frame.f_lineno)
                    return tracer
            sys.settrace(tracer)
            try:
                function(n)
            finally:
                sys.settrace(None)
            return sorted(lines)
        traced = quixbugs_harness.trace_tests(source, "f", tests, [10 ** 6] * 3)
        self.assertEqual(traced, [[True, plain(0)], [True, plain(7)], [True, plain(500)]])
        # A run over its limit is traced for its first steps only and is not a passed test.
        endless = "def f(n):\n    while True:\n        n += 1\n    return n\n"
        self.assertEqual(quixbugs_harness.trace_tests(endless, "f", [{"input": [0], "expected": 0}], [10 ** 6]), [[False, [1, 2, 3]]])


class ResumableDerivation(unittest.TestCase):
    def test_finished_steps_are_kept_and_the_result_does_not_depend_on_the_division(self):
        calls = []

        def work(key):
            calls.append(key)
            return {"key": key, "pair": (1, 2)}
        whole = [software_data.Steps()(k, lambda k=k: work(k)) for k in "abc"]
        directory = Path(tempfile.mkdtemp()) / "partial"
        first = software_data.Steps(directory, time.monotonic() + 60)
        self.assertEqual(first("a", lambda: work("a")), whole[0])
        stopped = software_data.Steps(directory, time.monotonic() - 1)  # the time limit has passed
        self.assertEqual(stopped("a", lambda: self.fail("a finished step ran again")), whole[0])
        with self.assertRaises(software_data.Unfinished):
            stopped("b", lambda: work("b"))
        resumed = software_data.Steps(directory)
        self.assertEqual([resumed(k, lambda k=k: work(k)) for k in "abc"], whole)
        self.assertEqual(calls, ["a", "b", "c", "a", "b", "c"])
        self.assertEqual(whole[0]["pair"], [1, 2])  # every result has passed through JSON
        self.assertFalse(list(directory.glob("*.part")))

    def test_the_preparation_script_keeps_and_clears_partial_work(self):
        spec = importlib.util.spec_from_file_location("prepare_software_under_test", ROOT / "scripts/prepare_software.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        script.PARTIAL = Path(tempfile.mkdtemp()) / "software-partial"
        runs = []

        class Family:
            RESUMABLE = True

            @staticmethod
            def derive(raw, steps):
                return {"x.json": json.dumps([steps(k, lambda k=k: runs.append(k) or k) for k in ("p", "q")]).encode()}
        with self.assertRaises(software_data.Unfinished):
            script.derive("toy", Family, {"source": b"raw"}, time.monotonic() - 1)
        (script.PARTIAL / "toy-0000000000000000").mkdir(parents=True)  # steps of other sources or older code
        files, work = script.derive("toy", Family, {"source": b"raw"}, None)
        self.assertEqual((json.loads(files["x.json"]), runs), (["p", "q"], ["p", "q"]))
        self.assertEqual([p.name for p in script.PARTIAL.iterdir()], [work.name])
        self.assertNotEqual(script.code_stamp({"source": b"raw"}), script.code_stamp({"source": b"other"}))
        plain = type("Plain", (), {"derive": staticmethod(lambda raw: {"y": b"1"})})
        self.assertEqual(script.derive("plain", plain, {}, None), ({"y": b"1"}, None))


class FaultOrder(unittest.TestCase):
    TARANTULA = ("def localize(s):\n    return [0.0 if ef == 0 else (ef / s['failed']) / (ef / s['failed'] + (ep / s['passed'] if s['passed'] else 0.0))\n"
                 "            for ef, ep in zip(s['ef'], s['ep'])]\n")

    def setUp(self):
        rng = random.Random(5)
        faults = []
        for _ in range(30):
            ef = [rng.randint(0, 6) for _ in range(40)]
            block = rng.choice([b for b, value in enumerate(ef) if value])
            faults.append({"subject": "textwrap", "mutant": 0, "operator": "AOR", "block": block, "failed": 6, "passed": 9,
                           "ef": ef, "ep": [rng.randint(0, 9) for _ in range(40)]})
        self.faults = faults
        self.original = sbfl.software_data.load_json
        sbfl.software_data.load_json = lambda name: {"blocks": {"textwrap": 40}, "pool_size": 15, "faults": faults}

    def tearDown(self):
        sbfl.software_data.load_json = self.original

    def test_a_formula_scores_the_same_while_positions_change_between_evaluations(self):
        formula = candidate(self.TARANTULA, "formula.py")
        first, details = sbfl.sbfl_score(formula)
        self.assertEqual(first, sbfl.sbfl_score(formula)[0])
        self.assertEqual(first, details["baseline_mean_exam"]["tarantula"])
        by_position = candidate("def localize(s):\n    return [float(-b) for b in range(s['blocks'])]\n", "position.py")
        self.assertNotEqual(sbfl.sbfl_score(by_position)[0], sbfl.sbfl_score(by_position)[0])

    def test_counting_the_calls_no_longer_finds_the_faulty_block(self):
        table = [f["block"] for f in self.faults]
        lookup = candidate(f"import random\nTABLE = {table!r}\nCALLS = [0]\n\n\ndef localize(s):\n    index = CALLS[0]\n    CALLS[0] += 1\n"
                           "    order = list(range(s['blocks']))\n    random.Random(index).shuffle(order)\n"
                           "    position = order.index(TABLE[index])\n    return [1.0 if b == position else 0.0 for b in range(s['blocks'])]\n", "lookup.py")
        # With the earlier seeds (the fault's index, faults in list order) this candidate ranked every faulty block first.
        earlier = statistics.fmean(sbfl.exam(lookup.localize(view), position)
                                   for view, position in (sbfl.shuffled(f, i) for i, f in enumerate(self.faults)))
        self.assertEqual(earlier, 1 / 40)
        self.assertGreater(sbfl.sbfl_score(lookup)[0], 0.25)
        fault = self.faults[0]
        self.assertEqual(sbfl.shuffled(fault, 4), sbfl.shuffled(fault, 4))
        self.assertNotIn("block", sbfl.shuffled(fault, 4)[0])


class StructuralChecks(unittest.TestCase):
    def test_an_operation_that_is_not_a_name_is_the_candidates_error(self):
        for program in ([["a", ["add"], "x", 1]], [["a", {"op": 1}, "x", 1]], [["a", None, "x", 1]]):
            with self.assertRaises(CandidateError):
                software_superopt.check_program(program, ["x"])
        self.assertEqual(len(software_superopt.check_program([["a", "sub", "x", 1], ["b", "and", "x", "a"]], ["x"])), 2)


@unittest.skipUnless((software_data.data_dir() / "pymod/mutation-textwrap.json").is_file(), "run scripts/prepare_software.py --only pymod")
class PreparedSubjects(unittest.TestCase):
    def test_mutation_analysis_repeats_exactly_and_slow_mutants_stay_cheap(self):
        data = software_data.load_json("pymod/mutation-textwrap.json")
        tests = [{"call": "wrap", "args": ["Mr. alpha kilo- end. two  spaces", 12, {"max_lines": 2}]},
                 {"call": "shorten", "args": ["two  spaces kilo- c end. \n \n xxxxxxxxxxxxxx \n", 12, {}]},
                 {"call": "dedent", "args": ["  a\n  b"]}]
        mutants = [[m["start"], m["end"], m["replacement"]] for m in data["mutants"][60:100]]  # includes mutants that loop
        args = [data["files"], data["target"], "textwrap", software_pymod.check_tests("textwrap", tests, 30), mutants]
        started = time.time()
        first = harness("mutation_analysis", args, "")
        self.assertEqual(first, harness("mutation_analysis", args, ""))
        self.assertLess(time.time() - started, 15)
        self.assertEqual(first["usable_tests"], 3)
        self.assertTrue(any(k >= 0 for k in first["killed"]) and any(k < 0 for k in first["killed"]))


if __name__ == "__main__":
    unittest.main()
