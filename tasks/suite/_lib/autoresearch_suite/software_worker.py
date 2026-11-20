"""Child process for software_sandbox: runs one untrusted function call under a policy.

Started as ``python -s software_worker.py <job_fd> <out_fd>`` with a clean
environment and an empty temporary working directory. It reads one framed JSON
job, applies OS resource limits, installs a Python audit hook, executes the
given source as a fresh module and calls one entry function. Oracle requests
and the final result are framed JSON messages on ``out_fd``.

This file must stay self-contained (standard library only): it never imports
the evaluator package, so candidate code cannot reach evaluator internals.
The policy is defence in depth for honest program search (no file writes, no
reads outside the Python installation, no processes, sockets, foreign-function
or subinterpreter access). It is not a security boundary against deliberately
hostile native code; run untrusted evaluations inside a container as well.
"""
import json
import os
import struct
import sys
import traceback

HEADER = struct.Struct(">Q")
BLOCKED_PREFIXES = (
    "socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty",
    "os.kill", "os.killpg", "os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace", "os.truncate",
    "os.chmod", "os.chown", "os.chflags", "os.lchflags", "os.lchmod", "os.lchown", "os.link", "os.symlink",
    "os.mkdir", "os.mkfifo", "os.mknod", "os.chdir", "os.fchdir", "os.chroot",
    "os.setxattr", "os.removexattr", "os.utime", "os.startfile", "shutil.", "ctypes.", "pty.", "webbrowser.",
    "urllib.Request", "http.client.", "ftplib.", "smtplib.", "poplib.", "imaplib.", "nntplib.", "telnetlib.",
    "sqlite3.", "resource.setrlimit", "resource.prlimit", "fcntl.", "winreg.", "msvcrt.", "_winapi.",
)
BLOCKED_MODULES = {
    "socket", "_socket", "ssl", "_ssl", "select", "selectors", "subprocess", "_posixsubprocess", "multiprocessing",
    "_multiprocessing", "concurrent.futures.process", "ctypes", "_ctypes", "cffi", "_cffi_backend", "urllib.request",
    "http.client", "ftplib", "smtplib", "webbrowser", "pty", "_interpreters", "_xxsubinterpreters",
    "_xxinterpchannels", "_interpchannels", "_interpqueues", "concurrent.interpreters", "_testcapi",
    "_testinternalcapi", "_testsinglephase", "_testmultiphase", "sqlite3", "_sqlite3", "fcntl",
}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
# Standard-library modules loaded before the policy because on some Python versions they import a blocked
# module at module level (platform imports subprocess on Python 3.10, and NumPy imports platform). The blocked
# module is then removed from sys.modules and its operations stay refused by the audit hook.
PRELOADED_STDLIB = ("platform",)


def read_frame(fd):
    def exact(n):
        chunks = []
        while n:
            chunk = os.read(fd, min(n, 1 << 20))
            if not chunk:
                raise EOFError("parent closed the job channel")
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)
    (size,) = HEADER.unpack(exact(HEADER.size))
    return json.loads(exact(size).decode("utf-8"))


def write_frame(fd, message):
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")
    data = HEADER.pack(len(payload)) + payload
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def apply_limits(limits):
    import resource
    cpu = int(limits.get("cpu_seconds") or 0)
    if cpu > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 2))
    memory = int(limits.get("memory_mb") or 0)
    if memory > 0 and sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (memory << 20, memory << 20))
    for name, value in (("RLIMIT_FSIZE", 0), ("RLIMIT_CORE", 0)):
        if hasattr(resource, name):
            try:
                resource.setrlimit(getattr(resource, name), (value, value))
            except (ValueError, OSError):
                pass


def install_policy(job):
    roots = sorted({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix) if p})
    roots += [os.path.realpath(p) for p in job.get("allow_read", [])]
    roots = tuple(r.rstrip(os.sep) + os.sep for r in roots)
    blocked = BLOCKED_MODULES | set(job.get("blocked_modules", []))
    for name in PRELOADED_STDLIB:
        __import__(name)
    for name in list(sys.modules):
        if name in blocked or name.split(".")[0] in blocked:
            del sys.modules[name]

    def readable(path):
        if isinstance(path, int):
            return True
        try:
            real = os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return False
        return real in ("/dev/null", "/dev/urandom") or real.startswith(roots)

    def hook(event, args):
        if event == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & WRITE_FLAGS)
            if writing or not readable(path):
                raise PermissionError(f"sandbox: file access denied ({path!r}, mode {mode!r})")
        elif event == "import":
            name = str(args[0])
            if name in blocked or name.split(".")[0] in blocked:
                raise ImportError(f"sandbox: import of {name} is not allowed")
        elif event in ("os.listdir", "os.scandir", "glob.glob", "glob.glob/2"):
            path = args[0] if args else "."
            if not readable(path if path not in (None, "") else "."):
                raise PermissionError(f"sandbox: directory listing denied ({path!r})")
        elif event.startswith(BLOCKED_PREFIXES):
            raise PermissionError(f"sandbox: operation {event} is not allowed")

    sys.addaudithook(hook)


class Oracle:
    """Callable proxy: oracle(request) sends one framed request and waits for the parent's answer."""

    def __init__(self, job_fd, out_fd):
        self._job_fd, self._out_fd = job_fd, out_fd

    def __call__(self, request):
        write_frame(self._out_fd, {"type": "oracle", "request": request})
        reply = read_frame(self._job_fd)
        if reply.get("error"):
            raise RuntimeError(reply["error"])
        return reply.get("response")


def main():
    job_fd, out_fd = int(sys.argv[1]), int(sys.argv[2])
    if sys.path and os.path.realpath(sys.path[0]) == os.path.realpath(os.path.dirname(os.path.abspath(__file__))):
        sys.path.pop(0)  # never expose the evaluator package directory
    job = read_frame(job_fd)
    apply_limits(job.get("limits", {}))
    import random
    random.seed(int(job.get("seed", 0)))
    sys.setrecursionlimit(int(job.get("recursion_limit", 10000)))
    install_policy(job)
    try:
        module = type(sys)("candidate")
        module.__dict__["__builtins__"] = __builtins__
        code = compile(job["source"], job.get("filename", "<candidate>"), "exec")
        exec(code, module.__dict__)
        sys.modules["candidate"] = module
        function = module.__dict__.get(job["entry"])
        if not callable(function):
            raise AttributeError(f"entry point {job['entry']}() is missing or not callable")
        args = list(job.get("args", []))
        if job.get("oracle"):
            args.append(Oracle(job_fd, out_fd))
        value = function(*args)
        message = {"type": "result", "ok": True, "value": value}
        try:
            write_frame(out_fd, message)
        except (TypeError, ValueError) as error:
            write_frame(out_fd, {"type": "result", "ok": False, "error": f"result is not JSON-serialisable: {error}"})
    except BaseException as error:  # noqa: BLE001 - every failure is reported to the parent
        text = "".join(traceback.format_exception(type(error), error, error.__traceback__))[-3000:]
        write_frame(out_fd, {"type": "result", "ok": False, "error": f"{type(error).__name__}: {error}", "traceback": text})


if __name__ == "__main__":
    main()
