"""Child process for kaggle_sandbox: loads one task's public inputs and runs the candidate's fit_predict.

Started as ``python -s kaggle_worker.py <job_fd> <out_fd>`` (on macOS inside ``sandbox-exec`` with a
profile that denies reading the task data except this task's public directory, denies writes outside
the working directory and denies network access). The job is one framed JSON message naming the
public input files; held-out labels are never named and live in a directory the child cannot read.

Order of events: on Linux, when the parent found it possible, move into private namespaces in which
the denied directories are empty (``isolate``); load the public inputs; install a Python audit hook
(no reads outside the Python installation, the public directory and the working directory; no writes
outside the working directory; no sockets or child processes); report ``ready`` (the parent starts the
time budget); execute the candidate source; call ``fit_predict(train, test_inputs, time_budget_s)``;
convert the result to float64 arrays, save them in the working directory and report ``result``.

This file must stay self-contained (standard library, numpy and pandas only): it never imports the
evaluator package. The audit hook is defence in depth for honest program search, not a security
boundary against deliberately hostile native code (see kaggle_sandbox.py for the residual risks).
"""
import json
import os
import struct
import sys
import traceback

HEADER = struct.Struct(">Q")
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
BLOCKED_EVENTS = (
    "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo", "socket.gethostbyname",
    "socket.gethostbyaddr", "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
    "os.forkpty", "os.kill", "os.killpg", "os.chdir", "os.fchdir", "os.chroot", "resource.setrlimit",
    "resource.prlimit", "webbrowser.open", "urllib.Request", "http.client.connect", "ftplib.connect", "smtplib.connect",
)
PATH_EVENTS = ("os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace", "os.mkdir", "os.truncate",
               "os.chmod", "os.chown", "os.link", "os.utime", "os.setxattr", "os.removexattr", "shutil.rmtree",
               "shutil.move", "shutil.copyfile", "shutil.copytree")
READ_DEVICES = ("/dev/null", "/dev/urandom", "/dev/random")
REAL_KINDS = "fiu"  # numpy dtype kinds accepted as predictions: floats and integers, not booleans or text
CLONE_NEWNS, CLONE_NEWIPC, CLONE_NEWUSER, CLONE_NEWPID, CLONE_NEWNET = 0x20000, 0x8000000, 0x10000000, 0x20000000, 0x40000000
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC, MS_BIND, MS_REC, MS_PRIVATE = 1, 2, 4, 8, 0x1000, 0x4000, 0x40000
SYS_MOUNT_SETATTR, AT_FDCWD, AT_RECURSIVE, MOUNT_ATTR_RDONLY = 442, -100, 0x8000, 1  # Linux 5.12, same number on every architecture


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


def _within(path, roots):
    return any(path == root or path.startswith(root + os.sep) for root in roots)


def isolate(deny_read, allow_read, workdir, close=()):
    """Linux: continue in a child process for which the directories of ``deny_read`` are empty.

    The caller moves into new user, mount, PID, network and IPC namespaces and forks; it then only waits
    and exits with the child's status. The child mounts a fresh /proc (no process outside is visible, so
    no other process's files or memory can be reached), an empty tmpfs over every denied directory with
    the ``allow_read`` directories bound back in place, and makes every mount read-only except the working
    directory, /dev/shm and /proc. A second user namespace then restores the user id and leaves the
    process without any capability over these mounts: native code cannot undo them, and every
    descendant inherits them. Must run while the process has one thread. Raises OSError where the
    kernel or its configuration does not permit this (the parent probes it first, see ``probe``).
    """
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)

    class MountAttr(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("attr_set", "attr_clr", "propagation", "userns_fd")]

    def check(result, what):
        if result != 0:
            code = ctypes.get_errno()
            raise OSError(code, f"{what}: {os.strerror(code)}")

    def mount(source, target, kind, flags, options=None):
        check(libc.mount(source, os.fsencode(target), kind, ctypes.c_ulong(flags), options), f"mount on {target}")

    def read_only(path, on, recursive=False):
        attr = MountAttr(MOUNT_ATTR_RDONLY if on else 0, 0 if on else MOUNT_ATTR_RDONLY, 0, 0)
        check(libc.syscall(SYS_MOUNT_SETATTR, AT_FDCWD, os.fsencode(path), AT_RECURSIVE if recursive else 0,
                           ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr))), f"mount_setattr on {path}")

    def identity(inside, outside):
        for name, text in (("setgroups", "deny"), ("uid_map", f"{inside[0]} {outside[0]} 1"), ("gid_map", f"{inside[1]} {outside[1]} 1")):
            with open(f"/proc/self/{name}", "w") as handle:
                handle.write(text)

    user, workdir = (os.getuid(), os.getgid()), os.path.realpath(workdir)
    check(libc.unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWIPC), "unshare")
    identity((0, 0), user)
    child = os.fork()
    if child:
        for fd in close:
            os.close(fd)
        code = os.waitstatus_to_exitcode(os.waitpid(child, 0)[1])
        os._exit(code if code >= 0 else 128 - code)
    libc.prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG: do not outlive the waiting process
    mount(None, "/", None, MS_REC | MS_PRIVATE)
    mount(b"proc", "/proc", b"proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)
    denied = []
    for path in sorted({os.path.realpath(p) for p in deny_read}, key=len):
        if os.path.isdir(path) and not _within(path, denied):
            denied.append(path)
    # descriptors of this mount namespace for what has to stay visible below a denied directory
    kept = [(path, os.open(path, os.O_RDONLY | os.O_DIRECTORY)) for path in sorted({os.path.realpath(p) for p in allow_read})
            if os.path.isdir(path) and _within(path, denied)]
    for path in denied:
        mount(b"tmpfs", path, b"tmpfs", MS_NOSUID | MS_NODEV, b"mode=0755,size=1m")
    for path, fd in kept:
        os.makedirs(path, exist_ok=True)
        mount(f"/proc/self/fd/{fd}".encode(), path, None, MS_BIND | MS_REC)
        os.close(fd)
    mount(os.fsencode(workdir), workdir, None, MS_BIND)
    read_only("/", True, recursive=True)
    read_only(workdir, False)
    read_only("/proc", False)  # the identity files below are written through it
    try:
        read_only("/dev/shm", False)  # shared memory of numerical libraries
    except OSError:
        pass  # not a mount of its own
    check(libc.unshare(CLONE_NEWUSER), "unshare")
    identity(user, (0, 0))
    os.chdir(workdir)  # enter the writable mount of the working directory


def probe(spec):
    """Exit status of ``kaggle_worker.py --probe``: 0 when ``isolate`` works here and hides the denied directories."""
    allow = [os.path.realpath(p) for p in spec["allow_read"]]
    before = {path: os.listdir(path) for path in map(os.path.realpath, spec["deny_read"]) if os.path.isdir(path)}
    try:
        isolate(spec["deny_read"], allow, os.getcwd())
        for directory, names in before.items():
            for name in names:
                path = os.path.join(directory, name)
                if os.path.lexists(path) and not any(_within(a, [path]) for a in allow):
                    return 3
        with open("probe.txt", "w") as handle:
            handle.write("writable")
    except OSError as error:
        sys.stderr.write(f"{error}\n")
        return 2
    return 0


def load_input(spec):
    """Public inputs: csv / jsonl tables (pandas.DataFrame), npz (dict of arrays) or npz_list (list of arrays)."""
    import numpy as np
    import pandas as pd
    kind, path = spec["kind"], spec.get("path")
    if kind == "csv":
        return pd.read_csv(path, dtype=spec.get("dtype"), keep_default_na=False, na_filter=False)
    if kind == "jsonl":
        # precise_float: the default decoder is one unit in the last place off for about a fifth of the
        # decimals, and not the same way on every architecture; scores of tree models move with it
        return pd.read_json(path, lines=True, dtype=spec.get("dtype", True), precise_float=True)
    if kind in ("npz", "npz_list"):
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        if kind == "npz":
            return arrays
        return [arrays[f"a{i}"] for i in range(len(arrays))]
    if kind == "bundle":  # dict of named parts, e.g. images plus an id table
        return {name: load_input(part) for name, part in spec["parts"].items()}
    raise ValueError(f"unknown input kind {kind!r}")


def install_policy(allow_read, workdir):
    roots = sorted({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, *allow_read, workdir) if p})
    roots = tuple(r.rstrip(os.sep) + os.sep for r in roots)
    work = os.path.realpath(workdir).rstrip(os.sep) + os.sep

    def refuse_process(*args, **kwargs):
        raise PermissionError("sandbox: operation fork_exec is not allowed")

    # subprocess.Popen and os.fork raise audit events, the function below them does not
    import _posixsubprocess
    import subprocess
    _posixsubprocess.fork_exec = refuse_process
    if hasattr(subprocess, "_fork_exec"):
        subprocess._fork_exec = refuse_process

    def resolve(path):
        """Absolute path without symbolic links, or None for something that is not a path."""
        try:
            return os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return None

    def readable(path):
        if isinstance(path, int):
            return True
        target = resolve(path)
        return target is not None and (target in READ_DEVICES or (target + os.sep).startswith(roots))

    def writable(path):
        if isinstance(path, int):
            return True
        target = resolve(path)
        return target is not None and (target == "/dev/null" or (target + os.sep).startswith(work))

    def hook(event, args):
        if event == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            if isinstance(path, int):
                return
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & WRITE_FLAGS)
            target = resolve(path)
            inside = target is not None and (target + os.sep).startswith(work)
            if not (inside or target == "/dev/null" or (not writing and target is not None
                                                        and (target in READ_DEVICES or (target + os.sep).startswith(roots)))):
                raise PermissionError(f"sandbox: file access denied ({path!r}, mode {mode!r})")
            # A descriptor of a directory makes later paths relative to it, which this hook cannot see
            # (the event carries no dir_fd): only directories of the working directory may be opened.
            if not inside and os.path.isdir(target):
                raise PermissionError(f"sandbox: file access denied (directory {path!r})")
        elif event in ("os.listdir", "os.scandir", "glob.glob", "glob.glob/2"):
            path = args[0] if args and args[0] not in (None, "") else "."
            if not readable(path):
                raise PermissionError(f"sandbox: directory listing denied ({path!r})")
        elif event == "os.symlink":
            # The target is resolved against the link's directory, not against the working directory:
            # only targets that cannot leave the link's directory are accepted.
            try:
                text = os.fsdecode(args[0])
            except (TypeError, ValueError):
                text = os.sep
            if os.path.isabs(text) or ".." in text.split(os.sep) or not writable(args[1]):
                raise PermissionError(f"sandbox: operation {event} denied ({args[0]!r})")
        elif event in PATH_EVENTS:
            for path in args[:2]:
                if isinstance(path, (str, bytes, os.PathLike)) and not writable(path):
                    raise PermissionError(f"sandbox: operation {event} denied ({path!r})")
        elif event == "import":
            if args and args[0] == "_posixsubprocess":
                raise ImportError("sandbox: import of _posixsubprocess is not allowed")
        elif event.startswith(BLOCKED_EVENTS):
            raise PermissionError(f"sandbox: operation {event} is not allowed")

    sys.addaudithook(hook)


def real_array(value, name):
    """float64 array of the returned predictions; booleans, text, objects and complex numbers are not predictions."""
    import numpy as np
    array = np.asarray(value)
    if array.dtype.kind not in REAL_KINDS:
        raise TypeError(f"{name} must hold real numbers (float or int), got dtype {array.dtype}")
    return np.asarray(array, dtype=np.float64)


def save_output(value, kind, workdir):
    import numpy as np
    path = os.path.join(workdir, "predictions.npz")
    if kind == "array_list":
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"fit_predict must return a list of arrays, got {type(value).__name__}")
        arrays = {f"a{i}": real_array(item, f"array {i} of the returned list") for i, item in enumerate(value)}
    else:
        arrays = {"a0": real_array(value, "the returned predictions")}
    np.savez(path, **arrays)
    return path


def main():
    job_fd, out_fd = int(sys.argv[1]), int(sys.argv[2])
    if sys.path and os.path.realpath(sys.path[0]) == os.path.realpath(os.path.dirname(os.path.abspath(__file__))):
        sys.path.pop(0)  # never expose the evaluator package directory
    job = read_frame(job_fd)
    workdir = os.getcwd()
    if job.get("namespaces"):
        try:
            isolate(job.get("deny_read", []), job.get("allow_read", []), workdir, close=(job_fd, out_fd))
        except OSError as error:
            write_frame(out_fd, {"type": "setup", "error": str(error)[:1500]})
            sys.exit(1)
    try:
        import random
        random.seed(int(job.get("seed", 0)))
        train = load_input(job["train"])
        test_inputs = load_input(job["test"])
        install_policy(job.get("allow_read", []), workdir)
        write_frame(out_fd, {"type": "ready"})  # the budget covers everything from here on
        module = type(sys)("candidate")
        module.__dict__["__builtins__"] = __builtins__
        exec(compile(job["source"], job.get("filename", "<candidate>"), "exec"), module.__dict__)
        sys.modules["candidate"] = module
        function = module.__dict__.get("fit_predict")
        if not callable(function):
            raise AttributeError("candidate must define fit_predict(train, test_inputs, time_budget_s)")
        if "torch" in sys.modules:
            sys.modules["torch"].set_num_threads(int(job.get("threads", 4)))
        value = function(train, test_inputs, float(job["budget_s"]))
        path = save_output(value, job.get("output", "array"), workdir)
        write_frame(out_fd, {"type": "result", "ok": True, "path": path})
    except BaseException as error:  # noqa: BLE001 - every failure is reported to the parent
        text = "".join(traceback.format_exception(type(error), error, error.__traceback__))[-3000:]
        write_frame(out_fd, {"type": "result", "ok": False, "error": f"{type(error).__name__}: {error}"[:1500], "traceback": text})


if __name__ == "__main__":
    if sys.argv[1:2] == ["--probe"]:
        sys.exit(probe(json.loads(sys.argv[2])))
    main()
