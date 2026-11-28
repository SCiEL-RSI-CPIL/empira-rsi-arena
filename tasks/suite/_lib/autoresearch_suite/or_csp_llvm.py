"""Function inlining for binary size through LLVM's MLGO interactive inliner (package or_csp).

Proxy for the feature-based setting of Magellan's inlining-for-size task: a policy
over the features of LLVM's MLInlineAdvisor decides every eligible call site
(legality and mandatory inlining stay with LLVM). ``opt`` runs the standard O2
pipeline on bitcode compiled with ``clang -Oz`` (minsize functions) with
``-enable-ml-inliner=release -inliner-interactive-channel-base=...``; this module
hosts the two named pipes of LLVM's interactive model runner (the training-log
format of llvm/lib/Analysis/TrainingLogger.cpp: a JSON header with tensor specs,
then per decision a JSON observation line, the raw feature tensors and a newline;
the host answers with the int64 advice). The workload is the sqlite3 command-line
shell built from the SQLite amalgamation (public domain); the reward is the text
section size of the linked binary relative to LLVM's default inlining heuristic.
"""
from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from arena_task_paths import data_root
from .runtime import CandidateError

DATA = data_root() / "or_csp" / "sqlite"
TYPES = {"int64_t": "q", "int32_t": "i", "int8_t": "b", "uint8_t": "B", "float": "f", "double": "d", "int16_t": "h",
         "uint16_t": "H", "uint32_t": "I", "uint64_t": "Q"}


def find_llvm():
    """opt/llc/clang/llvm-size of one LLVM installation that supports the interactive inliner."""
    roots = [os.environ.get("LLVM_BIN"), "/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"]
    if shutil.which("opt"):
        roots.append(str(Path(shutil.which("opt")).parent))
    for root in filter(None, roots):
        tools = {name: Path(root) / name for name in ("opt", "llc", "clang", "llvm-size")}
        if not all(path.exists() for path in tools.values()):
            continue
        help_text = subprocess.run([str(tools["opt"]), "--help-hidden"], capture_output=True, text=True).stdout
        if "inliner-interactive-channel-base" in help_text:
            version = subprocess.run([str(tools["opt"]), "--version"], capture_output=True, text=True).stdout
            return {k: str(v) for k, v in tools.items()}, " ".join(version.split()[:4])
    raise RuntimeError("no LLVM installation with the interactive ML inliner (LLVM >= 17; set LLVM_BIN)")


class _Channel:
    """Byte reader over the outbound pipe that never blocks past the compiler's lifetime."""

    def __init__(self, fd, process, deadline):
        self.fd, self.process, self.deadline, self.buffer = fd, process, deadline, bytearray()

    def _fill(self):
        while True:
            if time.monotonic() > self.deadline:
                raise TimeoutError("compiler session exceeded its time limit")
            ready = select.select([self.fd], [], [], 0.2)[0]
            if ready:
                chunk = os.read(self.fd, 1 << 16)
                if chunk:
                    self.buffer += chunk
                    return True
            elif self.process.poll() is not None:
                return False

    def line(self):
        while b"\n" not in self.buffer:
            if not self._fill():
                return None
        end = self.buffer.index(b"\n") + 1
        data, self.buffer = bytes(self.buffer[:end]), self.buffer[end:]
        return data

    def take(self, size):
        while len(self.buffer) < size:
            if not self._fill():
                raise RuntimeError("compiler closed the channel mid-observation")
        data, self.buffer = bytes(self.buffer[:size]), self.buffer[size:]
        return data


def _decode(spec, raw):
    fmt = "<" + TYPES[spec["type"]] * math.prod(spec["shape"])
    values = struct.unpack(fmt, raw)
    return values[0] if len(values) == 1 else list(values)


def run_policy(opt_cmd, base, policy, *, time_limit=300.0):
    """Run ``opt_cmd`` (which names the channel ``base``) answering every decision with ``policy``."""
    for suffix in (".in", ".out"):
        Path(base + suffix).unlink(missing_ok=True)
        os.mkfifo(base + suffix, 0o600)
    deadline = time.monotonic() + time_limit
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(opt_cmd, stdout=subprocess.DEVNULL, stderr=errors)
        out_fd = os.open(base + ".out", os.O_RDWR)  # also a writer, so reads never see a spurious EOF
        in_fd = None
        try:
            while in_fd is None:  # the compiler opens its inbound pipe first
                try:
                    in_fd = os.open(base + ".in", os.O_WRONLY | os.O_NONBLOCK)
                except OSError as error:
                    if error.errno != errno.ENXIO:
                        raise
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("compiler exited before opening the channel")
                    time.sleep(0.01)
            fcntl.fcntl(in_fd, fcntl.F_SETFL, fcntl.fcntl(in_fd, fcntl.F_GETFL) & ~os.O_NONBLOCK)
            channel = _Channel(out_fd, process, deadline)
            header = json.loads(channel.line())
            specs, advice = header["features"], header["advice"]
            sizes = [math.prod(s["shape"]) * struct.calcsize(TYPES[s["type"]]) for s in specs]
            decisions = inlined = 0
            policy_seconds = 0.0
            while True:
                line = channel.line()
                if line is None:
                    break
                event = json.loads(line)
                if "context" in event:
                    event = json.loads(channel.line())
                features = {spec["name"]: _decode(spec, channel.take(size)) for spec, size in zip(specs, sizes)}
                channel.take(1)
                started = time.perf_counter()
                try:
                    answer = policy(features)
                except CandidateError:
                    raise
                except Exception as error:  # noqa: BLE001 - a crashing policy is an invalid candidate
                    raise CandidateError(f"should_inline raised {type(error).__name__}: {error}") from error
                policy_seconds += time.perf_counter() - started
                if not (isinstance(answer, bool) or type(answer).__name__ in ("bool", "bool_")
                        or (isinstance(answer, int) and answer in (0, 1))):
                    raise CandidateError(f"should_inline must return a bool, got {type(answer).__name__}")
                decision = int(bool(answer))  # read once: the count below and the advice sent are the same value
                decisions += 1
                inlined += decision
                os.write(in_fd, struct.pack("<" + TYPES[advice["type"]], decision))
            process.wait(timeout=max(1.0, deadline - time.monotonic()))
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            os.close(out_fd)
            if in_fd is not None:
                os.close(in_fd)
        errors.seek(0)
        if process.returncode != 0:
            raise RuntimeError("opt failed: " + errors.read().decode(errors="replace")[-800:])
    return {"decisions": decisions, "inlined": inlined, "policy_seconds": round(policy_seconds, 3)}


def text_size(tools, binary):
    out = subprocess.run([tools["llvm-size"], "-A", str(binary)], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] in ("__text", ".text"):
            return int(parts[1])
    raise RuntimeError("no text section in " + str(binary))


def build_shell(tools, work, tag, policy=None):
    """Optimise, code-generate and link the sqlite3 shell; returns (text bytes, session stats)."""
    objects, stats = [], {"decisions": 0, "inlined": 0, "policy_seconds": 0.0}
    for unit in ("sqlite3", "shell"):
        opt = [tools["opt"], "-passes=default<O2>", str(work / f"{unit}.bc"), "-o", str(work / f"{unit}.{tag}.bc")]
        if policy is None:
            subprocess.run(opt, check=True, capture_output=True)
        else:
            base = str(work / f"channel_{tag}_{unit}")
            session = run_policy(opt[:1] + ["-enable-ml-inliner=release", f"-inliner-interactive-channel-base={base}"] + opt[1:], base, policy)
            for key in stats:
                stats[key] += session[key]
        subprocess.run([tools["llc"], "-O2", "-filetype=obj", str(work / f"{unit}.{tag}.bc"), "-o", str(work / f"{unit}.{tag}.o")],
                       check=True, capture_output=True)
        objects.append(str(work / f"{unit}.{tag}.o"))
    libraries = ["-lm"] + (["-lpthread", "-ldl"] if sys.platform.startswith("linux") else [])
    subprocess.run([tools["clang"], "-o", str(work / f"sqlite3.{tag}"), *objects, *libraries], check=True, capture_output=True)
    return text_size(tools, work / f"sqlite3.{tag}"), stats


def inlining_size_score(module, params, data_dir=None):
    """Text size of the policy-built sqlite3 binary / text size with LLVM's default heuristic (minimise)."""
    if not callable(getattr(module, "should_inline", None)):
        raise CandidateError("candidate must define should_inline(features) -> bool")
    archive = Path(data_dir or DATA) / params["archive"]
    if not archive.is_file():
        raise FileNotFoundError(f"{archive} missing; run scripts/prepare_or_csp.py")
    tools, version = find_llvm()
    with tempfile.TemporaryDirectory(prefix="or_csp_llvm_") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(archive) as bundle:
            for name in ("sqlite3.c", "shell.c", "sqlite3.h", "sqlite3ext.h"):
                (work / name).write_bytes(bundle.read(f"{params['folder']}/{name}"))
        for unit in ("sqlite3", "shell"):
            subprocess.run([tools["clang"], "-Oz", "-Xclang", "-disable-llvm-optzns", "-emit-llvm", "-c", str(work / f"{unit}.c"),
                            "-I", str(work), "-o", str(work / f"{unit}.bc")], check=True, capture_output=True)
        reference, _ = build_shell(tools, work, "default")
        size, stats = build_shell(tools, work, "candidate", module.should_inline)
    return size / reference, {"text_bytes": size, "default_heuristic_text_bytes": reference, "llvm": version,
                              "target": sys.platform, **stats}
