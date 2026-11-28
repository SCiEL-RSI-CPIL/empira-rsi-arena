"""Child process of clrs_sandbox: trains one candidate processor and predicts the test outputs.

Started as ``python -s clrs_worker.py <job_fd> <out_fd>`` with a clean
environment in an empty temporary directory. It reads one framed JSON job;
when the parent found Linux namespaces usable (``namespaces`` in the job), it
first moves into private namespaces in which the task cache is empty
(``kaggle_worker.isolate``; this must happen while the process has one thread,
so before numpy or torch is imported). It then imports the trusted trainer
(clrs_train, clrs_net) and torch, loads the job's data (train/validation splits
and test inputs; no test outputs), runs one tiny warm-up step (so torch's lazy
imports happen first), then applies
the resource limits and audit-hook policy of software_worker.py, unloads the
evaluator package, executes the candidate source and runs
``clrs_train.train_and_predict``. The predictions (compact integer arrays) and
diagnostics go back as one framed JSON message on ``out_fd``.
"""
import json
import os
import struct
import sys
import traceback

HEADER = struct.Struct(">Q")


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


def load_splits(path, numpy):
    splits = {}
    with numpy.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            split, name = key.split("/", 1)
            splits.setdefault(split, {})[name] = archive[key]
    return splits


def warm_up(clrs_train, algorithm, splits, torch):
    """One tiny checkpointed training step with a linear processor, so that torch's lazy imports
    (torch._dynamo via torch.optim, activation checkpointing) happen before the import policy is installed."""
    class Linear(torch.nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.layer = torch.nn.Linear(hidden, hidden)

        def forward(self, node_fts, edge_fts, graph_fts, adj, hidden):
            return self.layer(node_fts + hidden), edge_fts

    val = {key: value[:1] for key, value in splits["val"].items()}
    test = {key: value for key, value in val.items() if not key.startswith("output_")}
    tiny = {"train": splits["train"], "val": val, "test": test}
    clrs_train.train_and_predict(Linear, algorithm, tiny, steps=1, eval_every=1, hidden=8, batch_size=2, threads=1,
                                 checkpoint_steps=True)


def main():
    job_fd, out_fd = int(sys.argv[1]), int(sys.argv[2])
    if sys.path and os.path.realpath(sys.path[0]) == os.path.realpath(os.path.dirname(os.path.abspath(__file__))):
        sys.path.pop(0)  # the script directory is the evaluator package; import it only as a package
    job = read_frame(job_fd)
    sys.path.insert(0, job["lib"])
    if job.get("namespaces"):
        try:
            from autoresearch_suite.kaggle_worker import isolate  # standard library only
            isolate(job.get("deny_read", []), job.get("allow_read", []), os.getcwd(), close=(job_fd, out_fd))
        except OSError as error:
            write_frame(out_fd, {"type": "setup", "error": str(error)[:1500]})
            sys.exit(1)
    try:
        import numpy
        import torch
        torch.set_num_threads(int(job["budget"].get("threads", 4)))
        from autoresearch_suite import clrs_sandbox, clrs_train, software_worker
        splits = load_splits(job["data_path"], numpy)
        warm_up(clrs_train, job["algorithm"], splits, torch)
        encode = clrs_sandbox.encode_arrays
        sys.path.remove(job["lib"])
        software_worker.apply_limits(job.get("limits", {}))
        software_worker.install_policy({"blocked_modules": ["autoresearch_suite", "arena_task_paths"]})
    except BaseException as error:  # noqa: BLE001 - setup failures are evaluator problems
        write_frame(out_fd, {"type": "result", "ok": False, "error": f"evaluator setup failed: {type(error).__name__}: {error}"})
        return
    try:
        module = type(sys)("candidate_program")
        module.__dict__["__builtins__"] = __builtins__
        exec(compile(job["source"], job.get("filename", "<candidate>"), "exec"), module.__dict__)
        sys.modules["candidate_program"] = module
        build = module.__dict__.get("build_processor")
        if not callable(build):
            raise AttributeError("the program must define build_processor(hidden_dim)")
        predictions, details = clrs_train.train_and_predict(build, job["algorithm"], splits, **job["budget"])
        message = {"type": "result", "ok": True, "predictions": encode(predictions), "details": details}
    except BaseException as error:  # noqa: BLE001 - every candidate failure is reported to the parent
        if type(error).__name__ == "CandidateError":  # a contract violation detected by the trainer
            message = {"type": "result", "ok": False, "error": str(error)}
        else:
            text = "".join(traceback.format_exception(type(error), error, error.__traceback__))[-2000:]
            message = {"type": "result", "ok": False, "error": f"{type(error).__name__}: {error}", "traceback": text}
    try:
        write_frame(out_fd, message)
    except (TypeError, ValueError) as error:
        write_frame(out_fd, {"type": "result", "ok": False, "error": f"diagnostics are not JSON-serialisable: {error}"})


if __name__ == "__main__":
    main()
