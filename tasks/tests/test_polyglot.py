"""Tests of the polyglot evaluator package (Aider Polyglot coding agents).

Data-dependent tests skip when ``python scripts/prepare_polyglot.py`` has not
been run; the container tests skip unless Docker and the evaluator image
(``python scripts/prepare_polyglot.py --build-image``) are available. Nothing
here calls a model.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import polyglot_agent, polyglot_data, polyglot_image, polyglot_runner  # noqa: E402
from autoresearch_suite.runtime import CandidateError  # noqa: E402
from suite_test_support import seed_source  # noqa: E402

DGM_SMALL = ["go__dominoes", "cpp__all-your-base", "python__dominoes", "java__sgf-parsing", "javascript__robot-name",
             "rust__variable-length-quantity", "python__beer-song", "go__book-store", "javascript__bottle-song", "rust__bowling"]
HAVE_DATA = (polyglot_data.data_dir() / "manifest.json").is_file()


def image_or_none():
    if not HAVE_DATA:
        return None
    try:
        return polyglot_image.require_image()["tag"]
    except RuntimeError:
        return None


class FakeLLM:
    """Stands in for llm.LLM: scripted replies or errors, usage counters."""

    def __init__(self, replies=(), error=None):
        self.replies, self.error, self.budget, self.calls = list(replies), error, 0.05, []

    def __call__(self, prompt, **options):
        self.calls.append((prompt, options))
        if self.error:
            raise self.error
        return self.replies.pop(0) if self.replies else ""

    def usage(self):
        return {"model": "fake", "requests": len(self.calls), "cached_responses": 0, "cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


SETTINGS = {"agent_cpu_seconds": 30, "agent_seconds": 60}


class SummaryTests(unittest.TestCase):
    def test_pytest(self):
        ok = polyglot_runner.summary("python", "x\n============================== 12 passed in 0.03s ==============================\n")
        self.assertEqual(ok, {"passed": 12, "failed": 0, "ok": True})
        bad = polyglot_runner.summary("python", "=========== 1 failed, 11 passed in 0.10s ===========")
        self.assertFalse(bad["ok"])
        self.assertFalse(polyglot_runner.summary("python", "")["ok"])
        self.assertFalse(polyglot_runner.summary("python", "============ 3 skipped in 0.01s ============")["ok"])

    def test_cargo(self):
        text = ("test result: ok. 0 passed; 0 failed; 0 ignored\n"
                "test result: ok. 31 passed; 0 failed; 0 ignored\n"
                "test result: ok. 0 passed; 0 failed; 0 ignored\n")
        self.assertEqual(polyglot_runner.summary("rust", text), {"passed": 31, "failed": 0, "ok": True})
        self.assertEqual(polyglot_runner.summary("rust", "test result: FAILED. 30 passed; 1 failed;"), {"passed": 30, "failed": 1, "ok": False})

    def test_go(self):
        self.assertTrue(polyglot_runner.summary("go", "ok  \tdominoes\t0.055s\n")["ok"])
        self.assertFalse(polyglot_runner.summary("go", "--- FAIL: TestX (0.00s)\nFAIL\nFAIL\tdominoes\t0.003s\n")["ok"])
        self.assertFalse(polyglot_runner.summary("go", "")["ok"])

    def test_jest(self):
        self.assertTrue(polyglot_runner.summary("javascript", "Tests:       1 skipped, 7 passed, 8 total\n")["ok"])
        self.assertFalse(polyglot_runner.summary("javascript", "Tests:       1 failed, 7 passed, 8 total\n")["ok"])

    def test_catch2(self):
        self.assertTrue(polyglot_runner.summary("cpp", "All tests passed (17 assertions in 17 test cases)\n")["ok"])
        failing = "test cases: 17 | 16 passed | 1 failed\nassertions: 17 | 16 passed | 1 failed\n"
        self.assertEqual(polyglot_runner.summary("cpp", failing), {"passed": 16, "failed": 1, "ok": False})

    def test_gradle(self):
        text = "SgfParsingTest > emptyInput() PASSED\n\nSgfParsingTest > twoNodes() PASSED\n\nBUILD SUCCESSFUL in 3s\n"
        self.assertEqual(polyglot_runner.summary("java", text), {"passed": 2, "failed": 0, "ok": True})
        self.assertFalse(polyglot_runner.summary("java", "SgfParsingTest > a() PASSED\nSgfParsingTest > b() FAILED\n")["ok"])
        self.assertEqual(polyglot_runner.summary("java", "BUILD FAILED in 2s\n"), {"passed": 0, "failed": 1, "ok": False})
        mixed = "T > a() PASSED\nT > b() FAILED\nT > c() FAILED\n3 tests completed, 2 failed\nBUILD FAILED in 4s\n"
        self.assertEqual(polyglot_runner.summary("java", mixed), {"passed": 1, "failed": 2, "ok": False})

    def test_bundle_tar(self):
        data = polyglot_runner.bundle_tar("demo", {"a.py": (b"x = 1\n", 0o644), "gradlew": (b"#!/bin/sh\n", 0o755)}, "python")
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            members = {m.name: m for m in tar.getmembers()}
            self.assertEqual(sorted(members), ["demo/a.py", "demo/gradlew", "run.sh"])
            self.assertEqual(members["demo/gradlew"].mode, 0o755)
            self.assertIn(b"pytest", tar.extractfile("run.sh").read())


@unittest.skipUnless(HAVE_DATA, "prepared polyglot data missing (python scripts/prepare_polyglot.py)")
class DataTests(unittest.TestCase):
    def test_manifest_and_dgm_small_subset(self):
        manifest = polyglot_data.manifest()
        self.assertEqual(len(manifest["exercises"]), 225)
        self.assertEqual(polyglot_data.subset_ids("small"), DGM_SMALL)
        self.assertEqual(len(polyglot_data.subset_ids("medium")), 50)
        heldout = polyglot_data.subset_ids("heldout")
        self.assertEqual(len(heldout), 166)
        self.assertFalse(set(heldout) & (set(DGM_SMALL) | set(polyglot_data.subset_ids("medium"))))
        self.assertEqual(len(set(DGM_SMALL) | set(polyglot_data.subset_ids("medium")) | set(heldout)), 225)
        with self.assertRaises(CandidateError):
            polyglot_data.subset_ids("large")

    def test_visible_inputs_hide_tests_and_references(self):
        for exercise_id in DGM_SMALL:
            exercise = polyglot_data.Exercise(exercise_id)
            exercise.verify()
            visible = exercise.visible()
            self.assertEqual(sorted(visible), ["context", "files", "id", "instructions", "language"])
            self.assertTrue(visible["instructions"].strip())
            hidden = set(exercise.files.get("test", [])) | set(exercise.files.get("example", [])) | set(exercise.files.get("editor", []))
            self.assertFalse(hidden & (set(visible["files"]) | set(visible["context"])), exercise_id)
            self.assertEqual(sorted(visible["files"]), sorted(exercise.files["solution"]))
        java = polyglot_data.Exercise("java__sgf-parsing").visible()
        self.assertEqual(sorted(java["files"]), ["src/main/java/SgfParsing.java"])
        self.assertEqual(sorted(java["context"]), ["build.gradle"])

    def test_bundle_overrides_solution_and_enables_java_tests(self):
        exercise = polyglot_data.Exercise("java__sgf-parsing")
        files = exercise.bundle({"src/main/java/SgfParsing.java": "class X {}\n"})
        self.assertEqual(files["src/main/java/SgfParsing.java"][0], b"class X {}\n")
        self.assertNotIn(b"@Disabled(", files["src/test/java/SgfParsingTest.java"][0])
        self.assertEqual(files["gradlew"][1], 0o755)
        self.assertFalse(any(rel.startswith((".meta/", ".docs/")) for rel in files))
        self.assertIn("src/main/java/SgfNode.java", files)  # helper files reach the test run, not the agent

    def test_reference_solution_mapping(self):
        mapped, remove = polyglot_data.Exercise("cpp__circular-buffer").reference_solution()
        self.assertEqual(sorted(mapped), ["circular_buffer.h"])
        self.assertEqual(remove, ["circular_buffer.cpp"])
        mapped, _ = polyglot_data.Exercise("java__poker").reference_solution()
        self.assertEqual(sorted(mapped), ["src/main/java/Card.java", "src/main/java/Hand.java", "src/main/java/Poker.java"])
        mapped, _ = polyglot_data.Exercise("rust__decimal").reference_solution()
        self.assertEqual(sorted(mapped), ["Cargo.toml", "src/lib.rs"])
        self.assertIn("num-bigint", mapped["Cargo.toml"])


@unittest.skipUnless(HAVE_DATA, "prepared polyglot data missing (python scripts/prepare_polyglot.py)")
class AgentTests(unittest.TestCase):
    def setUp(self):
        self.exercise = polyglot_data.Exercise("python__beer-song")

    def test_validate_files(self):
        stub = self.exercise.stubs()["beer_song.py"]
        self.assertEqual(polyglot_agent.validate_files({"beer_song.py": stub}, self.exercise), {})
        self.assertEqual(polyglot_agent.validate_files({"beer_song.py": "x = 1\n"}, self.exercise), {"beer_song.py": "x = 1\n"})
        for bad in ({"beer_song_test.py": "x"}, {"beer_song.py": 3}, {"beer_song.py": "a\x00b"},
                    {"beer_song.py": "x" * (polyglot_agent.MAX_FILE_CHARS + 1)}, ["beer_song.py"]):
            with self.assertRaises(CandidateError):
                polyglot_agent.validate_files(bad, self.exercise)

    def run_agent(self, source, llm):
        return polyglot_agent.run_agent(source, self.exercise, llm, SETTINGS)

    def test_agent_runs_in_sandbox_with_model_calls(self):
        source = ("def solve(exercise, llm):\n"
                  "    reply = llm('write ' + exercise['id'], max_tokens=99999, temperature=0.5)\n"
                  "    return {'beer_song.py': reply}\n")
        llm = FakeLLM(["def recite(start, take=1):\n    return []\n"])
        files, record = self.run_agent(source, llm)
        self.assertEqual(files, {"beer_song.py": "def recite(start, take=1):\n    return []\n"})
        self.assertEqual(record["agent_status"], "ok")
        self.assertEqual(llm.calls[0][1]["max_tokens"], polyglot_agent.MAX_TOKENS_CAP)
        self.assertEqual(llm.calls[0][1]["temperature"], 0.5)

    def test_agent_cannot_read_prepared_tests(self):
        path = self.exercise.root / "beer_song_test.py"
        source = f"def solve(exercise, llm):\n    return {{'beer_song.py': open({str(path)!r}).read()}}\n"
        files, record = self.run_agent(source, FakeLLM())
        self.assertIsNone(files)
        self.assertEqual(record["agent_status"], "crash")
        self.assertIn("PermissionError", record["agent_error"])

    def test_contract_violations_invalidate(self):
        for source in ("def solve(exercise, llm):\n    return None\n", "def solve(exercise):\n    return {}\n",
                       "solve = None\n"):
            with self.assertRaises(CandidateError):
                self.run_agent(source, FakeLLM())

    def test_budget_errors_reach_the_agent(self):
        source = ("def solve(exercise, llm):\n    try:\n        llm('x')\n    except RuntimeError as error:\n"
                  "        return {'beer_song.py': 'error = ' + repr(str(error)) + '\\n'}\n    return {}\n")
        llm = FakeLLM(error=CandidateError("per-evaluation model budget of $0.05 exhausted after 3 requests ($0.0500 spent)"))
        files, record = self.run_agent(source, llm)
        self.assertIn("budget of this exercise", files["beer_song.py"])
        self.assertTrue(record["model_errors"])

    def test_service_failures_stop_the_evaluation(self):
        source = "def solve(exercise, llm):\n    llm('x')\n    return {}\n"
        llm = FakeLLM(error=CandidateError("model request failed after retries: timed out"))
        with self.assertRaises(RuntimeError):
            self.run_agent(source, llm)

    def test_reserved_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.py"
            path.write_text("def solve(exercise, llm):\n    return {}\n\n__polyglot_solve__ = 1\n")
            spec = importlib.util.spec_from_file_location("polyglot_candidate", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with self.assertRaises(CandidateError):
                polyglot_agent.check_candidate(module)


@unittest.skipUnless(HAVE_DATA, "prepared polyglot data missing (python scripts/prepare_polyglot.py)")
class SeedTests(unittest.TestCase):
    def test_seed_parses_whole_file_replies(self):
        namespace = {}
        exec(compile(seed_source("dgm-aider-polyglot"), "seed.py", "exec"), namespace)
        exercise = polyglot_data.Exercise("cpp__all-your-base").visible()
        reply = "Here you go.\n\n**all_your_base.h**\n```cpp\n#pragma once\n```\n\nall_your_base.cpp\n```cpp\nint x;\n```\n"
        llm = FakeLLM([reply])
        edits = namespace["solve"](exercise, llm)
        self.assertEqual(edits, {"all_your_base.h": "#pragma once\n", "all_your_base.cpp": "int x;\n"})
        prompt = llm.calls[0][0]
        self.assertIn("all_your_base.h", prompt)
        self.assertIn("CMakeLists.txt (read-only)", prompt)
        self.assertNotIn("all_your_base_test", prompt)


@unittest.skipUnless(image_or_none(), "Docker or the polyglot image is unavailable (python scripts/prepare_polyglot.py --build-image)")
class ContainerTests(unittest.TestCase):
    """Reference solutions pass and stubs fail in the offline image (one exercise per language)."""

    def test_reference_passes_and_stub_fails(self):
        image = image_or_none()
        for exercise_id in ("python__beer-song", "go__dominoes", "rust__bowling", "javascript__bottle-song", "cpp__all-your-base",
                            "java__sgf-parsing"):
            exercise = polyglot_data.Exercise(exercise_id)
            gold = polyglot_runner.reference_run(image, exercise)
            self.assertTrue(gold["resolved"], (exercise_id, gold["output_tail"][-800:]))
            stub = polyglot_runner.run_exercise(image, exercise, {})
            self.assertFalse(stub["resolved"], exercise_id)


if __name__ == "__main__":
    unittest.main()
