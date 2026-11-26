"""Static source policy and child-process audit hook shared by the jax_station workers.

Standard library only: the workers (separate interpreter processes) import this
file by path before anything else, and the evaluator side imports it to reject
candidates early. The policy is defence in depth for honest program search; it
is not a security boundary against deliberately hostile code: a LIMEN interface
is traced inside the process that trains the agent and measures its success, so
whatever reaches Python's builtins there can rewrite the measurement. The
static check therefore also closes the routes to builtins that were found in
review (module aliases, frame and code objects, class patterns, dunder names
and system modules exposed as attributes), without claiming to be complete.
"""
from __future__ import annotations

import ast
import os
import sys

# Attribute names a LIMEN interface never needs: file I/O and callbacks into Python (on any object) ...
DENIED_ATTRIBUTES = frozenset({
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap",
    "pure_callback", "io_callback", "callback", "system", "popen", "environ", "getenv", "putenv",
})
# ... and configuration, experimental, export and private APIs of an imported module (``jax.config``, ``jax.extend``).
MODULE_DENIED_ATTRIBUTES = frozenset({
    "config", "experimental", "debug", "ffi", "extend", "interpreters", "lib", "dlpack", "profiler", "distributed",
    "monitoring", "export", "clear_caches", "live_arrays", "make_jaxpr", "effects_barrier", "device_get", "sharding",
})
DENIED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "input", "breakpoint", "exit", "quit", "help", "__builtins__", "builtins", "memoryview", "sys", "os", "importlib",
})
# Frames, code objects and tracebacks (reachable from any generator or coroutine) lead to builtins and module globals ...
INTROSPECTION_ATTRIBUTES = frozenset({
    "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await", "cr_origin", "ag_frame", "ag_code", "ag_await",
    "f_back", "f_builtins", "f_globals", "f_locals", "f_code", "f_trace", "tb_frame", "tb_next", "func_globals", "func_code",
    "co_consts", "co_names", "co_code",
})
# ... and so do system modules that a library module happens to expose as an attribute (``some_module.os``).
SYSTEM_MODULE_ATTRIBUTES = frozenset({
    "os", "sys", "posix", "builtins", "importlib", "subprocess", "ctypes", "shutil", "pathlib", "socket", "pickle", "marshal",
    "inspect", "gc", "runpy", "tempfile",
})
ALLOWED_DUNDER_NAMES = frozenset({"__name__"})


def _denied_attribute(name):
    return (name.startswith("_") or name in DENIED_ATTRIBUTES or name in INTROSPECTION_ATTRIBUTES
            or name in SYSTEM_MODULE_ATTRIBUTES)


def check_interface_source(source, allowed_imports=("jax", "jax.numpy", "jax.lax", "jax.nn", "math")):
    """Problems of a LIMEN interface module (empty list = accepted).

    Only the listed modules may be imported, no attribute may be assigned or
    deleted, no private (underscore) or denied attribute may be read (by an
    attribute access or a class pattern) and no exec-like builtin or dunder
    name may be named. An imported module may only be used as the base of an
    attribute access (``jnp.where``), never passed around or renamed, so the
    rules for module attributes cannot be sidestepped through an alias.
    Module-level code may define constants, functions and classes.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"syntax error: {error}"]
    allowed = set(allowed_imports)
    modules = {(alias.asname or alias.name).split(".")[0] for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    # Names bound to a module object: ``import a.b [as c]`` and ``from a import b`` where a.b is an allowed module.
    module_objects = {(alias.asname or alias.name.split(".")[0]) for node in ast.walk(tree) if isinstance(node, ast.Import)
                      for alias in node.names}
    module_objects |= {(alias.asname or alias.name) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                       for alias in node.names if f"{node.module or ''}.{alias.name}" in allowed}
    attribute_bases = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    problems = []
    for node in ast.walk(tree):
        where = f"line {getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed:
                    problems.append(f"{where}: import of {alias.name} is not allowed (allowed: {', '.join(sorted(allowed))})")
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level or base not in allowed and not all(f"{base}.{a.name}" in allowed for a in node.names):
                problems.append(f"{where}: import from {base or '.'} is not allowed")
            elif base in allowed:
                for alias in node.names:
                    full = f"{base}.{alias.name}"
                    if alias.name == "*" or alias.name.startswith("_") or alias.name in DENIED_ATTRIBUTES | MODULE_DENIED_ATTRIBUTES:
                        problems.append(f"{where}: import of {full} is not allowed")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                problems.append(f"{where}: assigning or deleting attribute {node.attr} is not allowed")
            module_attr = isinstance(node.value, ast.Name) and node.value.id in modules
            if _denied_attribute(node.attr) or (module_attr and node.attr in MODULE_DENIED_ATTRIBUTES):
                problems.append(f"{where}: attribute {node.attr} is not allowed")
        elif isinstance(node, ast.MatchClass):
            for name in node.kwd_attrs:  # ``case object(__globals__=g)`` reads an attribute without an Attribute node
                if _denied_attribute(name) or name in MODULE_DENIED_ATTRIBUTES:
                    problems.append(f"{where}: attribute {name} is not allowed")
        elif isinstance(node, ast.Name):
            dunder = node.id.startswith("__") and node.id.endswith("__") and node.id not in ALLOWED_DUNDER_NAMES
            if node.id in DENIED_NAMES or dunder:
                problems.append(f"{where}: name {node.id} is not allowed")
            elif node.id in module_objects and isinstance(node.ctx, ast.Load) and id(node) not in attribute_bases:
                problems.append(f"{where}: module {node.id} may only be used as {node.id}.<attribute>")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"{where}: global/nonlocal statements are not allowed")
    return problems


BLOCKED_EVENT_PREFIXES = (
    "socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty",
    "os.kill", "os.killpg", "os.chmod", "os.chown", "os.link", "os.symlink", "os.chdir", "os.chroot", "ctypes.",
    "pty.", "webbrowser.", "urllib.Request", "http.client.", "ftplib.", "smtplib.", "sqlite3.", "resource.setrlimit",
    "resource.prlimit", "fcntl.", "_winapi.",
)
BLOCKED_MODULES = frozenset({
    "socket", "_socket", "ssl", "_ssl", "subprocess", "_posixsubprocess", "multiprocessing", "ctypes", "_ctypes", "cffi",
    "urllib.request", "http.client", "ftplib", "smtplib", "webbrowser", "pty", "sqlite3", "_sqlite3",
})
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def install_audit_hook(read_roots, write_roots):
    """Block processes, sockets, ctypes and new imports of such modules; confine file access.

    Reads are allowed below ``read_roots`` (plus the Python installation and
    /dev/null, /dev/urandom), writes only below ``write_roots``. Modules that
    are already imported stay usable by the trusted harness.
    """
    def norm(paths):
        return tuple(os.path.realpath(p).rstrip(os.sep) + os.sep for p in paths if p)

    writable = norm(write_roots)
    readable = norm(list(read_roots) + [sys.prefix, sys.base_prefix, sys.exec_prefix]) + writable

    def allowed(path, roots):
        if isinstance(path, int):
            return True
        try:
            real = os.path.realpath(os.fsdecode(path))
        except (TypeError, ValueError):
            return False
        return real in ("/dev/null", "/dev/urandom") or (real + os.sep).startswith(roots)

    def hook(event, args):
        if event == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & WRITE_FLAGS)
            if not allowed(path, writable if writing else readable):
                raise PermissionError(f"sandbox: file access denied ({path!r}, mode {mode!r})")
        elif event in ("os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace", "os.mkdir", "os.truncate", "os.utime"):
            if args and not allowed(args[0], writable):
                raise PermissionError(f"sandbox: operation {event} denied ({args[0]!r})")
        elif event == "import":
            name = str(args[0])
            if name in BLOCKED_MODULES or name.split(".")[0] in BLOCKED_MODULES:
                if name not in sys.modules:
                    raise ImportError(f"sandbox: import of {name} is not allowed")
        elif event.startswith(BLOCKED_EVENT_PREFIXES):
            raise PermissionError(f"sandbox: operation {event} is not allowed")

    sys.addaudithook(hook)


def apply_limits(cpu_seconds, file_mb):
    """CPU-time and file-size limits for the current (child) process."""
    import resource
    if cpu_seconds:
        resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds) + 5))
    for name, value in (("RLIMIT_FSIZE", int(file_mb) << 20), ("RLIMIT_CORE", 0)):
        if hasattr(resource, name):
            try:
                resource.setrlimit(getattr(resource, name), (value, value))
            except (ValueError, OSError):
                pass


def missing_modules(names):
    """Message naming the modules of ``names`` that this interpreter cannot import, or None."""
    import importlib.util
    absent = [name for name in names if importlib.util.find_spec(name) is None]
    if not absent:
        return None
    return f"the evaluator's interpreter ({sys.executable}) lacks {', '.join(absent)}; install them (python -m pip install ...)"
