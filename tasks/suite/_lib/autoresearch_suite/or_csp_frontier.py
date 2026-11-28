"""Frontier-CS algorithmic problem 0, "Pack the Polyominoes (Reflections Allowed)" (package or_csp).

The CORAL example examples/frontier_cs_algo/0 delegates grading to the
Frontier-CS judge (a go-judge Docker service). This module runs the same
ingredients locally from github.com/FrontierCS/Frontier-CS at commit
e6ef46eb4848d99c3c88505d0084284b291bba76 (MIT): the 70 test inputs of
algorithmic/problems/0/testdata and the unchanged testlib checker chk.cc
(compiled with GNU g++). The candidate's C++ program is compiled like the judge
compiles C++17 (``g++ -O2 -pipe -std=gnu++17``; ``-static -s`` are dropped
because macOS cannot link statically) and run per case with a 2 s CPU limit,
a 4 s wall-clock limit and a 256 MiB memory limit. A case's score is the
``Ratio:`` the checker reports when it accepts the output (total cells / W x H),
0 otherwise; the problem score is the mean ratio x 100, as in the judge engine
(algorithmic/judge/src/judge_engine.js).

Isolation: the candidate's Python file is never executed in this process (see
``solution_source``). The compiled program runs with the user's own privileges
under resource limits, in an empty working directory, without the file-system
and process isolation of the judge's go-judge sandbox. The verdicts are kept
out of its reach: the checker is compiled and the inputs are read (and, for the
prepared data, compared with the data lock) before the program first runs,
every process it starts is killed when its case ends (Linux: the launcher is a
child subreaper), and the checks run after the last case from fresh copies.
Not closed: while it runs, a program can change the files of the cases that
run alongside it (their outputs, the launcher binary), and it can write any
file the user can write, for example the task library, which a later
evaluation would use.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from arena_task_paths import data_root
from . import software_sandbox
from .runtime import CandidateError
from .sandboxed import defines

DATA = data_root() / "or_csp" / "frontier_cs_0"
LOCK = Path(__file__).resolve().parents[3] / "catalog" / "or_csp-data-lock.json"
CPP_FLAGS = ["-O2", "-pipe", "-std=gnu++17"]
RATIO = re.compile(r"Ratio: ([\d.]+)")
NO_SOURCE = "candidate must define SOLUTION_CPP, the C++17 source of a complete program"
# Appended to the candidate's Python source in the sandboxed child: hands back the value the module gave SOLUTION_CPP.
SOURCE_HARNESS = '''

def __or_csp_solution_cpp__():
    return globals().get("SOLUTION_CPP")
'''
# Launcher: run the solution in a fresh grandchild with resource limits (no Python preexec_fn, safe with threads) and
# write that grandchild's exit status and usage, as the last line, to an inherited pipe. Linux keeps a process's
# peak RSS across exec, so the usage of a process forked from the evaluator would include the evaluator's memory; the
# grandchild is forked from this small launcher instead. The launcher enforces the wall-clock limit itself and, on
# Linux, is a child subreaper: when the solution ends, every process it left behind (also one that started its own
# session) is re-parented to the launcher, killed and reaped, so no process of the candidate outlives its case, and
# the CPU time of the reaped processes is reported with the solution's own (RLIMIT_CPU is per process only).
RUNNER_C = r"""
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/wait.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
static volatile sig_atomic_t expired = 0;
static void on_alarm(int sig) { (void)sig; expired = 1; }
static void cap(int what, rlim_t soft, rlim_t hard) {
    struct rlimit old, lim;
    if (getrlimit(what, &old) != 0) return;
    if (old.rlim_max != RLIM_INFINITY && hard > old.rlim_max) hard = old.rlim_max;
    if (soft > hard) soft = hard;
    lim.rlim_cur = soft; lim.rlim_max = hard;
    setrlimit(what, &lim);
}
static double reaped_cpu = 0.0;  /* CPU seconds of the descendants this launcher reaps (go-judge counts them too) */
static void reap(int options) {
    struct rusage usage;
    while (wait4(-1, NULL, options, &usage) > 0)
        reaped_cpu += usage.ru_utime.tv_sec + usage.ru_stime.tv_sec + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6;
}
static void reap_all(pid_t group) {
    kill(-group, SIGKILL);
#ifdef __linux__
    char path[64];
    snprintf(path, sizeof path, "/proc/%d/task/%d/children", (int)getpid(), (int)getpid());
    for (int round = 0; round < 5000; round++) {
        FILE *list = fopen(path, "r");
        long child;
        if (list) {
            while (fscanf(list, "%ld", &child) == 1) kill((pid_t)child, SIGKILL);
            fclose(list);
        }
        reap(WNOHANG);
        if (waitpid(-1, NULL, WNOHANG) < 0 && errno == ECHILD) return;
        usleep(1000);
    }
#else
    reap(WNOHANG);
#endif
}
int main(int argc, char **argv) {
    if (argc < 7) return 125;
    rlim_t cpu = (rlim_t)atol(argv[1]), mem = (rlim_t)atoll(argv[2]), out = (rlim_t)atoll(argv[3]);
    double wall = atof(argv[4]);
    int report = atoi(argv[5]);
#ifdef __linux__
    prctl(PR_SET_CHILD_SUBREAPER, 1);
#endif
    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_handler = on_alarm;  /* no SA_RESTART: wait4 returns EINTR when the wall-clock limit expires */
    sigaction(SIGALRM, &action, NULL);
    pid_t pid = fork();
    if (pid < 0) return 125;
    if (pid == 0) {
        close(report);
        setpgid(0, 0);
        cap(RLIMIT_CPU, cpu, cpu + 1);
        cap(RLIMIT_FSIZE, out, out);
        cap(RLIMIT_STACK, mem, mem);
#ifdef __linux__
        cap(RLIMIT_AS, mem, mem);
#endif
        execv(argv[6], argv + 6);
        _exit(126);
    }
    setpgid(pid, pid);
    struct itimerval timer;
    memset(&timer, 0, sizeof timer);
    timer.it_value.tv_sec = (time_t)wall;
    timer.it_value.tv_usec = (suseconds_t)((wall - (double)(time_t)wall) * 1e6);
    setitimer(ITIMER_REAL, &timer, NULL);
    int status = 0;
    struct rusage usage;
    for (;;) {
        pid_t done = wait4(pid, &status, 0, &usage);
        if (done == pid) break;
        if (done < 0 && errno == EINTR) {
            if (expired) kill(pid, SIGKILL);
            continue;
        }
        return 125;
    }
    memset(&timer, 0, sizeof timer);
    setitimer(ITIMER_REAL, &timer, NULL);
    reap_all(pid);
    dprintf(report, "\n%d %d %ld.%06ld %ld.%06ld %ld %d %.6f\n", WIFEXITED(status) ? WEXITSTATUS(status) : -1,
            WIFSIGNALED(status) ? WTERMSIG(status) : 0, (long)usage.ru_utime.tv_sec, (long)usage.ru_utime.tv_usec,
            (long)usage.ru_stime.tv_sec, (long)usage.ru_stime.tv_usec, (long)usage.ru_maxrss, (int)expired, reaped_cpu);
    return 0;
}
"""


def find_gnu_compiler():
    """A GNU g++ (bits/stdc++.h needs libstdc++); Apple's g++ is clang and is skipped."""
    names = [f"g++-{v}" for v in range(20, 9, -1)] + ["g++"]
    for name in names:
        path = shutil.which(name) or (f"/opt/homebrew/bin/{name}" if Path(f"/opt/homebrew/bin/{name}").exists() else None)
        if not path:
            continue
        try:
            banner = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "Free Software Foundation" in banner:
            return path
    raise RuntimeError("no GNU g++ found (on macOS install Homebrew gcc, which provides g++-NN)")


def solution_source(module):
    """The C++ source a candidate provides in SOLUTION_CPP.

    The evaluator hands over the candidate unexecuted (``sandboxed.run``: a module that carries the file's path
    only). Its Python source runs in the child process of software_sandbox, which returns the value of SOLUTION_CPP,
    so no statement of the candidate file is executed in this process. A module object that already has the
    attribute is trusted code (the tests) and is read directly.
    """
    if getattr(module, "__candidate_source__", False):
        if not defines(module, "SOLUTION_CPP"):
            raise CandidateError(NO_SOURCE)
        try:
            text = Path(module.__file__).read_text(encoding="utf-8")
            source = software_sandbox.execute(text + SOURCE_HARNESS, "__or_csp_solution_cpp__", cpu_seconds=30, memory_mb=1024,
                                              filename=Path(module.__file__).name)
        except (OSError, ValueError) as error:
            raise CandidateError(f"cannot read the candidate file: {error}") from None
        except software_sandbox.SandboxFailure as failure:
            raise CandidateError(f"the candidate module failed in the sandbox ({failure.kind}): {failure}"[:2000]) from None
    else:
        source = getattr(module, "SOLUTION_CPP", None)
    if type(source) is not str or "main" not in source:
        raise CandidateError(NO_SOURCE)
    return source


def compile_cpp(compiler, source_path, binary, *, timeout=120, extra=()):
    try:
        proc = subprocess.run([compiler, *CPP_FLAGS, *extra, "-o", str(binary), str(source_path)], capture_output=True, text=True,
                              errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"the compiler did not finish within {timeout} s"
    return proc.returncode == 0, (proc.stderr or proc.stdout)[-1500:]


def run_case(runner, binary, case_in, out_path, cpu_s, wall_s, memory_bytes, output_bytes=128 << 20, cwd=None):
    """Run one case; returns (status, cpu_seconds, max_rss_bytes).

    The CPU seconds are the solution's own and those of every process it started (on Linux the launcher reaps the
    ones the solution left behind and adds their usage; the judge's cgroup counts them as well); the peak resident
    set is the solution process's own.

    The launcher's report is the last line on a pipe it inherits (the solution does not), so text a program writes
    there through /proc comes before it. The launcher stops the program at ``wall_s``; the kill of its process group
    here, a few seconds later, only backs that up.
    """
    read_fd, write_fd = os.pipe()
    tail = bytearray()

    def drain():
        while True:
            chunk = os.read(read_fd, 1 << 16)
            if not chunk:
                return
            tail.extend(chunk)
            del tail[:-4096]

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    killed = threading.Event()
    try:
        with open(case_in, "rb") as stdin, open(out_path, "wb") as stdout:
            proc = subprocess.Popen([str(runner), str(int(cpu_s)), str(memory_bytes), str(output_bytes), repr(float(wall_s)),
                                     str(write_fd), str(binary)], stdin=stdin, stdout=stdout, stderr=subprocess.DEVNULL,
                                    start_new_session=True, pass_fds=(write_fd,), cwd=cwd)
        os.close(write_fd)
        write_fd = None

        def kill():
            killed.set()
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # the launcher's own session
            except ProcessLookupError:
                pass

        timer = threading.Timer(wall_s + 5.0, kill)
        timer.start()
        try:
            proc.wait()
        finally:
            timer.cancel()
        reader.join(timeout=5.0)
    finally:
        if write_fd is not None:
            os.close(write_fd)
        if not reader.is_alive():  # a reader still blocked keeps its descriptor (a writer outlived the launcher)
            os.close(read_fd)
    try:
        code, sig, utime, stime, maxrss, expired, reaped = bytes(tail).decode("ascii").strip().splitlines()[-1].split()
        code, sig, maxrss, expired = int(code), int(sig), int(maxrss), int(expired)
        cpu = float(utime) + float(stime) + float(reaped)
    except (IndexError, UnicodeDecodeError, ValueError):
        return ("time_limit_exceeded" if killed.is_set() else "runtime_error"), 0.0, 0
    rss = maxrss if sys.platform == "darwin" else maxrss * 1024
    if killed.is_set() or expired or cpu > cpu_s or sig in (signal.SIGKILL, signal.SIGXCPU):
        return "time_limit_exceeded", cpu, rss
    if rss > memory_bytes:
        return "memory_limit_exceeded", cpu, rss
    if sig or code != 0:
        return "runtime_error", cpu, rss
    return "ran", cpu, rss


def verify_prepared(data, contents):
    """Raise RuntimeError unless every file in ``contents`` ({name: bytes}) matches the data lock.

    The candidate's program runs with the user's privileges and could rewrite the prepared inputs or the checker
    source for later evaluations; the files are therefore read once, before any program of the candidate runs, and
    compared with the SHA-256 recorded in catalog/or_csp-data-lock.json. Without the catalog (the library used outside
    this repository) nothing is compared.
    """
    if not LOCK.is_file():
        return
    entries = {entry["path"].split("/", 1)[1]: entry for entry in json.loads(LOCK.read_text())["files"]
               if entry["path"].startswith("frontier_cs_0/")}
    for name, payload in contents.items():
        entry = entries.get(name)
        if entry is None or len(payload) != entry["size_bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"prepared file {Path(data) / name} does not match the data lock; rerun scripts/prepare_or_csp.py")


def check_case(checker, case_in, out_path, case_ans):
    try:
        proc = subprocess.run([str(checker), str(case_in), str(out_path), str(case_ans)], capture_output=True, text=True,
                              errors="replace", timeout=60)
    except subprocess.TimeoutExpired:  # the judge gives its checker 10 CPU seconds; no verdict means no score
        return 0.0, "the checker did not finish within 60 s"
    message = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = RATIO.search(message) if proc.returncode in (0, 7) else None
    return (float(match.group(1)) if match else 0.0), message.strip()[:300]


def polyomino_score(module, params, data_dir=None):
    """Mean checker ratio x 100 over the test cases (maximize).

    The checker is compiled before any program of the candidate runs. After the last case, when no process of the
    candidate is left, every accepted run is checked in a fresh directory by a fresh copy of that checker binary
    against the inputs as they were read at the start, so a program cannot replace the checker or an input.
    """
    source = solution_source(module)
    data = Path(data_dir or DATA)
    cases = sorted((p for p in data.glob("*.in")), key=lambda p: int(p.stem))
    if len(cases) != int(params["n_cases"]):
        raise FileNotFoundError(f"expected {params['n_cases']} cases in {data}; run scripts/prepare_or_csp.py")
    inputs = {case.stem: case.read_bytes() for case in cases}
    checker_sources = {name: (data / name).read_bytes() for name in ("chk.cc", "testlib.h")}
    if data_dir is None:
        verify_prepared(data, {**{f"{stem}.in": payload for stem, payload in inputs.items()}, **checker_sources})
    compiler = find_gnu_compiler()
    with tempfile.TemporaryDirectory(prefix="or_csp_frontier_") as tmp:
        tmp = Path(tmp)
        build, run, check, cwd = (tmp / name for name in ("build", "run", "check", "cwd"))
        for directory in (build, run, cwd):
            directory.mkdir()
        for name, payload in checker_sources.items():
            (build / name).write_bytes(payload)
        ok, log = compile_cpp(compiler, build / "chk.cc", build / "chk", extra=("-I", str(build)))
        if not ok:
            raise RuntimeError("could not build the official checker: " + log)
        checker = (build / "chk").read_bytes()
        (build / "runner.c").write_text(RUNNER_C)
        if subprocess.run([compiler, "-x", "c", "-O2", "-o", str(build / "runner"), str(build / "runner.c")], capture_output=True).returncode:
            raise RuntimeError("could not build the case launcher")
        (build / "solution.cpp").write_text(source, encoding="utf-8", errors="replace")
        ok, log = compile_cpp(compiler, build / "solution.cpp", build / "solution")
        if not ok:
            raise CandidateError("compilation failed: " + log)

        def one(stem):
            (run / f"{stem}.in").write_bytes(inputs[stem])
            status, cpu, rss = run_case(build / "runner", build / "solution", run / f"{stem}.in", run / f"{stem}.out",
                                        float(params["time_limit_s"]), 2.0 * float(params["time_limit_s"]),
                                        int(params["memory_mb"]) << 20, cwd=cwd)
            return {"case": int(stem), "status": status, "ratio": 0.0, "cpu_seconds": round(cpu, 3),
                    "max_rss_mb": round(rss / 2 ** 20, 1), "checker": ""}

        workers = max(1, min(int(params.get("workers", 4)), os.cpu_count() or 1))
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(one, inputs))
        # Every launcher has reaped its solution's processes: from here on no code of the candidate runs.
        check.mkdir()
        (check / "chk").write_bytes(checker)
        (check / "chk").chmod(0o700)
        (check / "placeholder.ans").write_text("zzz\n")  # every testdata/*.ans holds this placeholder; chk.cc never reads it

        def judge(row):
            stem = str(row["case"])
            if row["status"] == "ran":
                (check / f"{stem}.in").write_bytes(inputs[stem])
                ratio, message = check_case(check / "chk", check / f"{stem}.in", run / f"{stem}.out", check / "placeholder.ans")
                row.update(status="accepted" if ratio > 0 else "rejected", ratio=ratio, checker=message[:200])
            (run / f"{stem}.out").unlink(missing_ok=True)
            return row

        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(judge, rows))
    score = 100.0 * sum(r["ratio"] for r in rows) / len(rows)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return score, {"cases": rows, "status_counts": counts, "compiler": compiler, "flags": " ".join(CPP_FLAGS)}
