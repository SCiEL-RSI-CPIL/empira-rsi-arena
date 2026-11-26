"""LLVM inlining-for-size proxy (package or_csp): interactive-channel host and scoring."""
from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "suite" / "_lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "suite" / "_lib"))

from autoresearch_suite import or_csp_llvm as M  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402

try:
    TOOLS, VERSION = M.find_llvm()
except RuntimeError:
    TOOLS = VERSION = None
PROGRAM = r"""
#include <stdio.h>
static int sq(int x) { return x * x; }
static int add3(int a, int b, int c) { return a + b + c; }
int big(int n) { int s = 0; for (int i = 0; i < n; i++) { s += sq(i) ^ (s >> 3); if (s & 1) s += add3(i, n, s); } return s; }
int user1(int a) { return big(a) + sq(a); }
int user2(int a) { return big(a + 1) - add3(a, a, a); }
int main(int argc, char **argv) { printf("%d %d\n", user1(argc), user2(argc)); return 0; }
"""


class DecodeTests(unittest.TestCase):
    def test_tensor_decoding(self):
        self.assertEqual(M._decode({"type": "int64_t", "shape": [1]}, struct.pack("<q", -35)), -35)
        self.assertEqual(M._decode({"type": "float", "shape": [2]}, struct.pack("<2f", 1.5, 2.0)), [1.5, 2.0])


@unittest.skipUnless(TOOLS, "an LLVM >= 17 with the interactive ML inliner is required")
class ChannelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        (self.work / "t.c").write_text(PROGRAM)
        subprocess.run([TOOLS["clang"], "-Oz", "-Xclang", "-disable-llvm-optzns", "-emit-llvm", "-c", str(self.work / "t.c"),
                        "-o", str(self.work / "t.bc")], check=True)

    def tearDown(self):
        self.tmp.cleanup()

    def size(self, tag, policy=None, include_default=False):
        out = self.work / f"t.{tag}.bc"
        cmd = [TOOLS["opt"], "-passes=default<O2>", str(self.work / "t.bc"), "-o", str(out)]
        stats = None
        if policy is None:
            subprocess.run(cmd, check=True)
        else:
            base = str(self.work / f"chan_{tag}")
            flags = ["-enable-ml-inliner=release", f"-inliner-interactive-channel-base={base}"]
            if include_default:
                flags.append("-inliner-interactive-include-default")
            stats = M.run_policy(cmd[:1] + flags + cmd[1:], base, policy, time_limit=60)
        subprocess.run([TOOLS["llc"], "-O2", "-filetype=obj", str(out), "-o", str(self.work / f"t.{tag}.o")], check=True)
        return M.text_size(TOOLS, self.work / f"t.{tag}.o"), stats

    def test_replaying_the_default_decisions_reproduces_the_default_build(self):
        reference, _ = self.size("default")
        seen = []

        def replay(features):
            seen.append(features)
            return bool(features["inlining_default"])
        replayed, stats = self.size("replay", replay, include_default=True)
        self.assertEqual(replayed, reference)
        self.assertGreater(stats["decisions"], 0)
        self.assertIn("cost_estimate", seen[0])
        candidate_view = []
        never, stats = self.size("never", lambda features: candidate_view.append(features) or False)
        self.assertEqual(stats["inlined"], 0)
        self.assertGreaterEqual(never, reference)
        self.assertNotIn("inlining_default", candidate_view[0])  # candidates never see LLVM's own decision

    def test_bad_policies_are_invalid_and_never_hang(self):
        for policy in (lambda f: 1 / 0, lambda f: "yes", lambda f: 0.5):
            with self.assertRaises(CandidateError):
                self.size("bad", policy)

    def test_compiler_failure_is_reported(self):
        base = str(self.work / "chan_fail")
        with self.assertRaises(RuntimeError):
            M.run_policy([TOOLS["opt"], "-passes=default<Oz>", "-enable-ml-inliner=release", f"-inliner-interactive-channel-base={base}",
                          str(self.work / "t.bc"), "-o", str(self.work / "x.bc")], base, lambda f: False, time_limit=30)


@unittest.skipUnless(TOOLS and (M.DATA / "sqlite-amalgamation-3530400.zip").is_file(), "LLVM and the prepared SQLite archive are required")
class ScoreTests(unittest.TestCase):
    def test_never_inline_seed_is_larger_than_the_default(self):
        params = {"archive": "sqlite-amalgamation-3530400.zip", "folder": "sqlite-amalgamation-3530400"}
        score, details = M.inlining_size_score(types.SimpleNamespace(should_inline=lambda features: False), params)
        self.assertGreater(score, 1.0)
        self.assertEqual(details["inlined"], 0)
        self.assertGreater(details["decisions"], 1000)


if __name__ == "__main__":
    unittest.main()
