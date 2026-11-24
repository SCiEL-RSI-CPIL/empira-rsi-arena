"""Run finance and economics candidates in a sandboxed child process.

A scoring loop is written as a generator. It yields the next call of candidate code,
``step("forecast_premium", history)``, and receives the value the candidate returned;
its own ``return`` value is the result of the evaluation. ``run`` drives the generator.

For a candidate module every call is executed in the child process of earth_sandbox. The
child holds the candidate's source and the arguments released so far and nothing else:
realised returns, later histories, benchmark forecasts, reference solutions and the
evaluator package stay in this process, and the arguments of the next call are sent only
after the value of the current call has been received. A candidate therefore cannot read
the answer from the scoring loop, and a value cannot be revised once the next call has been
released.

For a plain callable the calls are made in this process. That route is for trusted code
only: the benchmarks of the scoring modules and their tests.

The evaluators load the candidate with ``sandboxed.run``, so no statement of the
candidate is executed in this process. The residual risk is that of earth_sandbox: outside
macOS deliberately hostile native code can bypass the file policy of the child.
"""
from __future__ import annotations

from types import ModuleType

from . import earth_sandbox
from .runtime import CandidateError

CPU_SECONDS = 600  # CPU time of the child for one evaluation; the evaluator's own time limit applies first
MEMORY_MB = 3000

HARNESS = '''
def __finance_session__(oracle):
    objects = {}
    request = oracle({"op": "start"})
    while request is not None:
        owner, _, name = request["call"].rpartition(".")
        function = getattr(objects[owner], name) if owner else globals()[name]
        value = function(*request["args"])
        if request.get("keep"):
            objects[request["keep"]] = value
            value = None
        request = oracle({"op": "result", "value": value})
'''


def step(call, *args, keep=None):
    """One call of candidate code.

    ``call`` names a function of the candidate, or ``handle.method`` for an object kept by an
    earlier step. With ``keep`` the returned object stays in the child under that handle and the
    loop receives None, for candidates that return a model with methods.
    """
    return {"call": call, "args": list(args), "keep": keep}


class Session:
    """Trusted side of the protocol: answers each request of the child with the loop's next call.

    The first value received for a call is that call's value. The next call is produced by
    sending that value into the loop, so nothing about a later step leaves this process before
    the current step is answered.
    """

    def __init__(self, steps):
        self.steps, self.state, self.value, self.error = steps, "new", None, None

    def __call__(self, request):
        op = request.get("op") if isinstance(request, dict) else None
        try:
            if op == "start" and self.state == "new":
                call = next(self.steps)
            elif op == "result" and self.state == "running":
                call = self.steps.send(request.get("value"))
            else:
                raise CandidateError(f"protocol violation: unexpected request {op!r}")
        except StopIteration as finished:
            self.state, self.value = "done", finished.value
            return None
        except CandidateError as error:
            self.state, self.error = "failed", str(error)
            raise
        self.state = "running"
        return call


def _run_trusted(function, entry, steps):
    """Drive the loop with a trusted callable standing for the candidate's entry function."""
    objects, namespace = {}, {entry: function}
    try:
        call = next(steps)
        while True:
            owner, _, name = call["call"].rpartition(".")
            target = getattr(objects[owner], name, None) if owner else namespace[name]
            if not callable(target):
                raise CandidateError(f"{entry} must return an object with {name}()")
            value = target(*call["args"])
            if call["keep"]:
                objects[call["keep"]] = value
                value = None
            call = steps.send(value)
    except StopIteration as finished:
        return finished.value


def run(candidate, steps, entry, *, cpu_seconds=CPU_SECONDS, memory_mb=MEMORY_MB):
    """Drive the scoring loop ``steps`` and return its result.

    ``candidate`` is the candidate module (sandboxed) or a trusted callable (this process).
    ``entry`` is the function the candidate must define. A failure of the candidate, in its
    own code or in the loop's validation of a returned value, is a CandidateError.
    """
    if not isinstance(candidate, ModuleType):
        if not callable(candidate):
            raise CandidateError(f"{entry} must be callable")
        return _run_trusted(candidate, entry, steps)
    source, filename = earth_sandbox.candidate_source(candidate, entry)
    session = Session(steps)
    try:
        earth_sandbox.execute(source, "__finance_session__", [], harness=HARNESS, oracle=session, filename=filename,
                              cpu_seconds=cpu_seconds, memory_mb=memory_mb)
    except earth_sandbox.SandboxFailure as failure:
        if session.error:
            raise CandidateError(session.error) from None
        last = failure.traceback_text.strip().splitlines()[-1] if failure.traceback_text.strip() else ""
        detail = f"; {last}" if failure.kind == "exception" and last and last not in str(failure) else ""
        raise CandidateError(f"candidate {entry}() failed in the sandbox ({failure.kind}): {failure}{detail}"[:2000]) from None
    finally:
        steps.close()
    if session.state != "done":
        raise CandidateError(f"the candidate stopped before the evaluation of {entry}() was complete")
    return session.value
