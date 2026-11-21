"""Child process for earth_sandbox: runs one candidate call without access to held-out data.

Started as ``python -s earth_worker.py <job_fd> <out_fd>`` with a clean environment and an
empty temporary working directory (on macOS inside ``sandbox-exec``, which denies reading
the prepared data and raw caches and all network access at the kernel level). It reads one
framed JSON job, applies resource limits, installs an audit hook, executes the candidate
source (followed by trusted harness code, if any) as a fresh module and calls one entry
function. Arrays travel as base64-encoded raw buffers, so values are transferred exactly.

Unlike software_worker.py this policy lets scientific libraries load (numpy, scipy,
scikit-learn, pandas), because their imports need ctypes and subprocess. The audit hook
still refuses file reads outside the Python installation and the working directory, file
writes outside the working directory, process creation and sockets. It is defence in depth
for honest program search, not a boundary against deliberately hostile native code; the
kernel sandbox covers native readers on macOS only.

This file must stay self-contained (standard library, plus numpy for array transport).
"""
import base64
import json
import os
import struct
import sys
import traceback

HEADER = struct.Struct(">Q")
BLOCKED_EVENTS = (
    "socket.", "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty",
    "os.kill", "os.killpg", "os.chroot", "os.chdir", "os.fchdir", "pty.", "webbrowser.", "urllib.Request",
    "http.client.", "ftplib.", "smtplib.", "poplib.", "imaplib.", "nntplib.", "telnetlib.", "resource.setrlimit",
    "resource.prlimit", "_winapi.", "msvcrt.", "winreg.", "sqlite3.connect",
)
WRITE_EVENTS = ("os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace", "os.truncate", "os.mkdir",
                "os.chmod", "os.chown", "os.link", "os.symlink", "os.utime", "shutil.", "os.mkfifo", "os.mknod")
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


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


def decode(value):
    """JSON transport form -> Python objects with numpy arrays."""
    if isinstance(value, dict):
        if "__nd__" in value:
            import numpy
            raw = base64.b64decode(value["__nd__"])
            return numpy.frombuffer(raw, dtype=numpy.dtype(value["dtype"])).reshape(value["shape"]).copy()
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def encode(value, depth=0):
    """Python/numpy objects -> JSON transport form; anything else is a contract violation."""
    if depth > 20:
        raise TypeError("output nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("output contains NaN or infinity")
        return value
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        if isinstance(value, numpy.ndarray):
            if value.dtype.kind not in "biuf":
                raise TypeError(f"arrays must be numeric, got dtype {value.dtype}")
            array = numpy.ascontiguousarray(value, dtype=numpy.float64 if value.dtype.kind == "f" else numpy.int64)
            return {"__nd__": base64.b64encode(array.tobytes()).decode("ascii"), "dtype": array.dtype.str,
                    "shape": list(array.shape)}
        if isinstance(value, numpy.generic):
            return encode(value.item(), depth + 1)
    if isinstance(value, (list, tuple)):
        return [encode(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("output dictionaries need string keys")
        return {key: encode(item, depth + 1) for key, item in value.items()}
    raise TypeError(f"unsupported output type {type(value).__name__}")


def apply_limits(limits):
    import resource
    cpu = int(limits.get("cpu_seconds") or 0)
    if cpu > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 2))
    memory = int(limits.get("memory_mb") or 0)
    if memory > 0 and sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (memory << 20, memory << 20))
    if hasattr(resource, "RLIMIT_CORE"):
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass


def install_policy(workdir):
    roots = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix) if p}
    roots |= {os.path.realpath(p) for p in sys.path if p and "site-packages" in p}
    readable_roots = tuple(sorted(r.rstrip(os.sep) + os.sep for r in roots | {workdir}))
    writable_root = workdir.rstrip(os.sep) + os.sep

    def resolve(path):
        if isinstance(path, int):
            return None
        try:
            return os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return ""

    def readable(path):
        real = resolve(path)
        return real is None or real in ("/dev/null", "/dev/urandom", "/dev/random") or (real + os.sep).startswith(readable_roots)

    def writable(path):
        real = resolve(path)
        return real is None or real == "/dev/null" or (real + os.sep).startswith(writable_root)

    def hook(event, args):
        if event == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & WRITE_FLAGS)
            if not (writable(path) if writing else readable(path)):
                raise PermissionError(f"sandbox: file access denied ({path!r})")
        elif event in ("os.listdir", "os.scandir", "glob.glob", "glob.glob/2"):
            path = args[0] if args and args[0] not in (None, "") else "."
            if not readable(path):
                raise PermissionError(f"sandbox: directory listing denied ({path!r})")
        elif event.startswith(WRITE_EVENTS):
            targets = [a for a in args if isinstance(a, (str, bytes, os.PathLike))]
            if not all(writable(a) for a in targets):
                raise PermissionError(f"sandbox: operation {event} outside the working directory")
        elif event.startswith(BLOCKED_EVENTS):
            raise PermissionError(f"sandbox: operation {event} is not allowed")

    sys.addaudithook(hook)


class Oracle:
    """oracle(request) sends one framed request to the trusted parent and returns its decoded answer."""

    def __init__(self, job_fd, out_fd):
        self._job_fd, self._out_fd = job_fd, out_fd

    def __call__(self, request):
        write_frame(self._out_fd, {"type": "oracle", "request": encode(request)})
        reply = read_frame(self._job_fd)
        if reply.get("error"):
            raise RuntimeError(reply["error"])
        return decode(reply.get("response"))


def main():
    job_fd, out_fd = int(sys.argv[1]), int(sys.argv[2])
    here = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or ".") != here]  # never expose the evaluator package
    job = read_frame(job_fd)
    apply_limits(job.get("limits", {}))
    import random
    random.seed(int(job.get("seed", 0)))
    sys.setrecursionlimit(int(job.get("recursion_limit", 10000)))
    import numpy  # noqa: F401 - loaded before the policy so that array transport always works
    install_policy(os.path.realpath(os.getcwd()))
    try:
        module = type(sys)("candidate")
        module.__dict__["__builtins__"] = __builtins__
        sys.modules["candidate"] = module
        exec(compile(job["source"], job.get("filename", "<candidate>"), "exec"), module.__dict__)
        if job.get("harness"):
            exec(compile(job["harness"], "<harness>", "exec"), module.__dict__)
        function = module.__dict__.get(job["entry"])
        if not callable(function):
            raise AttributeError(f"entry point {job['entry']}() is missing or not callable")
        args = decode(list(job.get("args", [])))
        if job.get("oracle"):
            args.append(Oracle(job_fd, out_fd))
        value = encode(function(*args))
        write_frame(out_fd, {"type": "result", "ok": True, "value": value})
    except BaseException as error:  # noqa: BLE001 - every failure is reported to the parent
        text = "".join(traceback.format_exception(type(error), error, error.__traceback__))[-3000:]
        write_frame(out_fd, {"type": "result", "ok": False, "error": f"{type(error).__name__}: {error}"[:2000], "traceback": text})


if __name__ == "__main__":
    main()
