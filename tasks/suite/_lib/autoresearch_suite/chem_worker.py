"""Child process of chem_sandbox: one candidate call that cannot see the evaluator's data.

Started as ``python -s chem_worker.py <job_fd> <out_fd>`` with a clean
environment in an empty temporary working directory. It reads one framed JSON
job, applies a CPU-time limit, installs a Python audit hook, executes the
candidate source as a fresh module and calls one entry function with the
inputs the evaluator chose to hand over (for prediction tasks: training inputs
with labels and test inputs without labels).

Policy (defence in depth for honest program search, not a security boundary):

* files may be read only inside the Python installation, a few system
  directories and the working directory, and never inside the hidden roots
  (the prepared data directory and the evaluator package);
* files may be written only inside the working directory;
* no process creation and no sockets.

Unlike software_worker.py this policy keeps ``ctypes`` and native extension
loading available, because scipy, scikit-learn and torch need them; a hostile
candidate could use them to bypass the hook, so adversarial evaluations still
need an operating-system sandbox or container. Standard library only: this
file never imports the evaluator package.
"""
import json
import os
import struct
import sys
import traceback

HEADER = struct.Struct(">Q")
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
SYSTEM_ROOTS = ("/usr", "/System", "/Library", "/etc", "/private/etc", "/dev", "/proc", "/sys", "/opt",
                "/private/var/db/timezone", "/var/db/timezone")
BLOCKED_EVENTS = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
                  "os.forkpty", "os.kill", "os.killpg", "pty.", "webbrowser.", "urllib.Request")
PATH_EVENTS = ("os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace", "os.mkdir", "os.chmod",
               "os.truncate", "os.link", "os.symlink", "shutil.rmtree", "shutil.move", "shutil.copyfile")


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
    view = memoryview(HEADER.pack(len(payload)) + payload)
    while view:
        view = view[os.write(fd, view):]


def apply_limits(limits):
    import resource
    cpu = int(limits.get("cpu_seconds") or 0)
    if cpu > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
    for name, value in (("RLIMIT_CORE", 0), ("RLIMIT_FSIZE", 1 << 30)):
        if hasattr(resource, name):
            try:
                resource.setrlimit(getattr(resource, name), (value, value))
            except (ValueError, OSError):
                pass


def _root(path):
    return os.path.realpath(path).rstrip(os.sep) + os.sep


def install_policy(job, workdir):
    prefixes = {sys.prefix, sys.base_prefix, sys.exec_prefix}
    readable_roots = tuple(sorted({_root(p) for p in prefixes if p} | {_root(p) for p in SYSTEM_ROOTS} | {_root(workdir)}))
    writable_roots = (_root(workdir),)
    hidden_roots = tuple(_root(p) for p in job.get("hidden", []))

    def resolve(path):
        try:
            return os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return None

    def allowed(path, roots):
        real = resolve(path)
        if real is None:
            return False
        if real == "/dev/null":
            return True
        probe = real.rstrip(os.sep) + os.sep
        if any(probe.startswith(h) for h in hidden_roots):
            return False
        return any(probe.startswith(r) for r in roots)

    def hook(event, args):
        if event == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            if isinstance(path, int) or path is None:
                return
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & WRITE_FLAGS)
            if not allowed(path, writable_roots if writing else readable_roots):
                raise PermissionError(f"sandbox: file access denied ({path!r}, mode {mode!r})")
        elif event in ("os.listdir", "os.scandir", "glob.glob", "glob.glob/2"):
            path = args[0] if args and args[0] not in (None, "") else "."
            if not isinstance(path, int) and not allowed(path, readable_roots):
                raise PermissionError(f"sandbox: directory listing denied ({path!r})")
        elif event.startswith(PATH_EVENTS):
            for path in args[:2]:
                if isinstance(path, (str, bytes, os.PathLike)) and not allowed(path, writable_roots):
                    raise PermissionError(f"sandbox: operation {event} denied ({path!r})")
        elif event.startswith(BLOCKED_EVENTS):
            raise PermissionError(f"sandbox: operation {event} is not allowed")

    sys.addaudithook(hook)


def main():
    job_fd, out_fd = int(sys.argv[1]), int(sys.argv[2])
    here = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:] = [p for p in sys.path if p and os.path.realpath(p) != here]  # never expose the evaluator package
    job = read_frame(job_fd)
    workdir = os.getcwd()
    apply_limits(job.get("limits", {}))
    import random
    seed = int(job.get("seed", 0))
    random.seed(seed)
    try:
        import numpy
        numpy.random.seed(seed)
    except ImportError:
        pass
    sys.setrecursionlimit(int(job.get("recursion_limit", 10000)))
    install_policy(job, workdir)
    try:
        module = type(sys)("candidate")
        module.__dict__["__builtins__"] = __builtins__
        exec(compile(job["source"], job.get("filename", "<candidate>"), "exec"), module.__dict__)
        sys.modules["candidate"] = module
        function = module.__dict__.get(job["entry"])
        if not callable(function):
            raise AttributeError(f"entry point {job['entry']}() is missing or not callable")
        value = function(*job.get("args", []))
        if hasattr(value, "tolist"):
            value = value.tolist()
        try:
            write_frame(out_fd, {"type": "result", "ok": True, "value": value})
        except (TypeError, ValueError) as error:
            write_frame(out_fd, {"type": "result", "ok": False, "error": f"result is not JSON-serialisable: {error}"})
    except BaseException as error:  # noqa: BLE001 - every failure is reported to the parent
        text = "".join(traceback.format_exception(type(error), error, error.__traceback__))[-3000:]
        write_frame(out_fd, {"type": "result", "ok": False, "error": f"{type(error).__name__}: {error}", "traceback": text})


if __name__ == "__main__":
    main()
