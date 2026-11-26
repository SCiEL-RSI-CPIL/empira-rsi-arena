"""Frontier-CS polyomino packing harness (package or_csp): limits, compilation and the seed score."""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "suite" / "_lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import or_csp_frontier as F  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
import suite_test_support  # noqa: E402

try:
    COMPILER = F.find_gnu_compiler()
except RuntimeError:
    COMPILER = None
HAVE_DATA = (F.DATA / "chk.cc").is_file() and len(list(F.DATA.glob("*.in"))) == 70
PARAMS = {"n_cases": 70, "time_limit_s": 2.0, "memory_mb": 256, "workers": 4}


def seed_source():
    return suite_test_support.seed_module("coral-frontier-cs-polyominoes-packing").SOLUTION_CPP


@unittest.skipUnless(COMPILER, "a GNU g++ is required")
class LimitTests(unittest.TestCase):
    def build(self, tmp, name, source):
        path = Path(tmp) / f"{name}.cc"
        path.write_text(source)
        ok, log = F.compile_cpp(COMPILER, path, Path(tmp) / name)
        self.assertTrue(ok, log)
        return Path(tmp) / name

    def test_launcher_enforces_cpu_and_reports_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "runner.c").write_text(F.RUNNER_C)
            import subprocess
            subprocess.run([COMPILER, "-x", "c", "-O2", "-o", str(Path(tmp) / "runner"), str(Path(tmp) / "runner.c")], check=True)
            (Path(tmp) / "in.txt").write_text("1\n")
            cases = {
                "spin": ("int main(){ volatile unsigned long x = 0; for(;;) x++; }", {"time_limit_exceeded"}),
                "crash": ("#include <cstdlib>\nint main(){ std::abort(); }", {"runtime_error"}),
                "hog": ("#include <vector>\n#include <cstdio>\nint main(){ std::vector<char> v(400u << 20, 1); "
                        "unsigned s = 0; for (size_t i = 0; i < v.size(); i += 4096) s += v[i]; std::printf(\"%u\", s); }",
                        {"memory_limit_exceeded", "runtime_error"}),
                "fine": ("#include <cstdio>\nint main(){ int n; std::scanf(\"%d\", &n); std::printf(\"%d\\n\", n + 1); }", {"ran"}),
            }
            for name, (source, expected) in cases.items():
                with self.subTest(program=name):
                    binary = self.build(tmp, name, source)
                    status, cpu, rss = F.run_case(Path(tmp) / "runner", binary, Path(tmp) / "in.txt", Path(tmp) / f"{name}.out",
                                                  1.0, 2.0, 256 << 20)
                    self.assertIn(status, expected)
                    if name == "fine":
                        self.assertEqual((Path(tmp) / "fine.out").read_text(), "2\n")

    def test_compile_errors_and_missing_source_are_invalid(self):
        if not HAVE_DATA:
            self.skipTest("prepared Frontier-CS data absent: run scripts/prepare_or_csp.py")
        for module in (types.SimpleNamespace(SOLUTION_CPP="int main( {"), types.SimpleNamespace(OTHER="x"),
                       types.SimpleNamespace(SOLUTION_CPP=42)):
            with self.assertRaises(CandidateError):
                F.polyomino_score(module, PARAMS)


@unittest.skipUnless(COMPILER and HAVE_DATA, "a GNU g++ and the prepared Frontier-CS data are required")
class SeedTests(unittest.TestCase):
    def test_seed_score_is_reproducible(self):
        score, details = F.polyomino_score(types.SimpleNamespace(SOLUTION_CPP=seed_source()), PARAMS)
        self.assertEqual(details["status_counts"], {"accepted": 70})
        self.assertAlmostEqual(score, 43.296286262857144, places=9)  # the seed is deterministic and far from the limits


if __name__ == "__main__":
    unittest.main()
