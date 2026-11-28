"""Hidden test runs of the polyglot evaluator: one fresh container per exercise, without network.

The test commands are those of Aider's benchmark harness, which DGM's
``polyglot/constants.py`` repeats: ``pytest`` (DGM adds ``-rA --tb=long``),
``cargo test -- --include-ignored``, ``go test ./...``, Aider's ``npm-test.sh``
(link the preinstalled ``node_modules``, turn ``xtest(`` into ``test(``, ``npm run
test``), Aider's ``cpp-test.sh`` (CMake with ``EXERCISM_RUN_ALL_TESTS``, ``make``)
and ``./gradlew test`` after removing ``@Disabled`` from the Java tests (Aider does
this; DGM's harness does not, so there only each class's first Java test runs).

An exercise counts as solved (pass@1) when the test command exits with status 0,
as in both harnesses, and, as an extra guard against a solution that ends the
test process early with status 0, the test framework's own summary must report
at least one passed and no failed test. The container gets the exercise files on
standard input as a tar stream (no host directory is mounted), runs with
``--network none``, no capabilities, a CPU, memory and process limit, and is
killed after ``seconds`` (Aider's harness allows 180 s per test run).
"""
from __future__ import annotations

import io
import re
import subprocess
import tarfile
import time
import uuid

# Test commands of Aider's benchmark harness (Aider-AI/aider at f5a512ba656ab109356f6ab53aaf039423c1d189: benchmark/benchmark.py,
# npm-test.sh and cpp-test.sh; Apache-2.0), as repeated in DGM's polyglot/constants.py (jennyzzt/dgm, Apache-2.0).
TEST_COMMANDS = {
    "python": "pytest -rA --tb=long",
    "rust": "cargo test -- --include-ignored",
    "go": "go test ./...",
    "javascript": ("set -e\n[ ! -e node_modules ] && ln -s /npm-install/node_modules .\n"
                   "[ ! -e package-lock.json ] && ln -s /npm-install/package-lock.json .\n"
                   "sed -i 's/\\bxtest(/test(/g' *.spec.js\nnpm run test"),
    "cpp": "set -e\n[ ! -d \"build\" ] && mkdir build\ncd build\ncmake -DEXERCISM_RUN_ALL_TESTS=1 -G \"Unix Makefiles\" ..\nmake",
    "java": "./gradlew test",
}
OUTPUT_TAIL = 6000
DOCKER_LIMITS = ("--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "1024")


def bundle_tar(slug, files, language):
    """Tar stream: ``run.sh`` (the test command) and ``<slug>/<files>``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        entries = [("run.sh", TEST_COMMANDS[language].encode() + b"\n", 0o755)]
        entries += [(f"{slug}/{rel}", data, mode) for rel, (data, mode) in sorted(files.items())]
        for name, data, mode in entries:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), mode, 0
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _counts(text, words):
    found = {}
    for number, word in re.findall(r"(\d+) (" + "|".join(words) + r")\b", text):
        found[word] = found.get(word, 0) + int(number)
    return found


def summary(language, output):
    """The test framework's own verdict: ``{"passed": n, "failed": n, "ok": bool}`` (counts as the framework reports them)."""
    lines = output.splitlines()
    passed = failed = 0
    seen = False
    if language == "python":
        finals = [line for line in lines if re.match(r"^=+ .*\b(passed|failed|error|errors|no tests ran)\b.* =+$", line)]
        if finals:
            seen = True
            counts = _counts(finals[-1], ("passed", "failed", "errors", "error"))
            passed, failed = counts.get("passed", 0), counts.get("failed", 0) + counts.get("errors", 0) + counts.get("error", 0)
    elif language == "rust":
        for ok, p, f in re.findall(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed", output, flags=re.M):
            seen = True
            passed, failed = passed + int(p), failed + (int(f) or int(ok != "ok"))
    elif language == "go":
        oks = [line for line in lines if re.match(r"^ok\s+\S+", line)]
        fails = [line for line in lines if re.match(r"^(FAIL|--- FAIL|panic:)", line)]
        seen = bool(oks or fails)
        passed, failed = len(oks), len(fails)
    elif language == "javascript":
        tests = [line for line in lines if re.match(r"^Tests:\s+", line)]
        if tests:
            seen = True
            counts = _counts(tests[-1], ("passed", "failed"))
            passed, failed = counts.get("passed", 0), counts.get("failed", 0)
    elif language == "cpp":
        match = re.findall(r"All tests passed \((\d+) assertions? in (\d+) test cases?\)", output)
        failing = re.findall(r"test cases?:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*(\d+) failed", output)
        if match:
            seen, passed = True, int(match[-1][1])
        if failing:
            seen, passed, failed = True, int(failing[-1][1]), int(failing[-1][2]) or 1
    elif language == "java":
        results = re.findall(r"^\S.* > .* (PASSED|FAILED)$", output, flags=re.M)
        seen = bool(results) or "BUILD FAILED" in output
        passed, failed = results.count("PASSED"), results.count("FAILED") or int("BUILD FAILED" in output)
    return {"passed": passed, "failed": failed, "ok": bool(seen and passed >= 1 and failed == 0)}


def run_tests(image, exercise_id, slug, language, files, *, seconds=180, cpus=2.0, memory_gb=3.0):
    """Run the hidden tests of one exercise; returns the record (``resolved`` is the pass@1 outcome)."""
    name = "artask-polyglot-" + uuid.uuid4().hex[:16]
    command = ["docker", "run", "-i", "--rm", "--name", name, *DOCKER_LIMITS, "--cpus", str(cpus),
               "--memory", f"{memory_gb}g", "--memory-swap", f"{memory_gb}g", image, "bash", "-c",
               f"cd /work && tar -xf - && cd {slug} && bash /work/run.sh 2>&1"]
    data = bundle_tar(slug, files, language)
    started = time.time()
    timed_out = False
    try:
        proc = subprocess.run(command, input=data, capture_output=True, timeout=seconds)
        code, output, error = proc.returncode, proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as expired:
        timed_out = True
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=60)
        code, error = None, ""
        output = (expired.stdout or b"").decode("utf-8", "replace") + "\nTests timed out!"
    elapsed = time.time() - started
    if code == 125 or (code is not None and code != 0 and not output.strip() and error.strip()):
        raise RuntimeError(f"docker could not run the tests of {exercise_id}: {error.strip()[-500:]}")
    verdict = summary(language, output)
    resolved = code == 0 and verdict["ok"]
    return {"id": exercise_id, "resolved": bool(resolved), "exit_code": code, "timed_out": timed_out, "test_seconds": round(elapsed, 2),
            "framework_summary": verdict, "exit0_without_summary": bool(code == 0 and not verdict["ok"]),
            "output_tail": output[-OUTPUT_TAIL:]}


def run_exercise(image, exercise, solution=None, remove=(), **limits):
    """Test ``solution`` (``{path: text}`` overriding the stubs) of a prepared ``polyglot_data.Exercise``."""
    files = exercise.bundle(solution, remove=remove)
    return run_tests(image, exercise.id, exercise.slug, exercise.language, files, **limits)


def reference_run(image, exercise, **limits):
    """Trusted check: Exercism's reference solution must pass the hidden tests."""
    mapped, remove = exercise.reference_solution()
    return run_exercise(image, exercise, mapped, remove=remove, **limits)
