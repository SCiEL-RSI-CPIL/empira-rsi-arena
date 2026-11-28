"""Clean CP-SAT solving process for the or_csp reformulation tasks.

The evaluator starts this file as a separate interpreter (``python -I``). It
never imports candidate code: it receives serialized ``CpModelProto`` messages
on stdin, solves each with fixed parameters (one search worker, random seed 42,
wall-clock limit given by the caller) and replies with the status, the CPU
seconds spent inside ``CpSolver.Solve`` (``getrusage`` of this process), the
solver's own wall and deterministic time, search counters and the full solution
vector.

Framing: request = 8-byte header ``<II`` (JSON length, proto length), JSON,
proto bytes; reply = 4-byte ``<I`` length and a JSON document. The JSON names
the encoding of the proto bytes: ``binary`` (wire format, OR-Tools <= 9.14) or
``text`` (protobuf text format, the only one OR-Tools >= 9.15 can load in Python;
the evaluator sends text). A model with an objective, a solution hint, a search
strategy, symmetry information or assumptions is answered ``MODEL_REJECTED``:
the CPMpy transformation of a satisfaction model produces none of them.
"""
import json
import resource
import struct
import sys

from ortools.sat.python import cp_model

STATUS = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE", cp_model.INFEASIBLE: "INFEASIBLE",
          cp_model.MODEL_INVALID: "MODEL_INVALID", cp_model.UNKNOWN: "UNKNOWN"}


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def read_exact(stream, size):
    chunks, remaining = [], size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def load_model(request, proto_bytes):
    """The CpModel of a request: binary wire format (OR-Tools <= 9.14) or protobuf text format (>= 9.15)."""
    model = cp_model.CpModel()
    if request.get("encoding", "binary") == "text" and hasattr(model.Proto(), "parse_text_format"):
        if not model.Proto().parse_text_format(proto_bytes.decode()):
            raise ValueError("the model text could not be parsed")
    elif request.get("encoding", "binary") == "text":  # OR-Tools <= 9.14: a Python protobuf message
        from google.protobuf import text_format
        text_format.Parse(proto_bytes.decode(), model.Proto())
    else:
        from ortools.sat import cp_model_pb2
        proto = cp_model_pb2.CpModelProto()
        proto.ParseFromString(proto_bytes)
        model.Proto().CopyFrom(proto)
    return model


def extras(proto):
    """Parts of a CpModelProto that the CPMpy transformation of a satisfaction model never produces."""
    found = []
    for name in ("objective", "floating_point_objective", "solution_hint", "symmetry"):
        has = getattr(proto, "has_" + name, None)
        if has() if callable(has) else proto.HasField(name):
            found.append(name)
    return found + [name for name in ("search_strategy", "assumptions") if len(getattr(proto, name))]


def solve(request, proto_bytes):
    model = load_model(request, proto_bytes)
    proto = model.Proto()
    rejected = extras(proto)
    if rejected:  # the model text comes from the sandboxed model process, which a candidate controls
        return {"status": "MODEL_REJECTED", "reason": ", ".join(rejected)}
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = int(request["seed"])
    solver.parameters.max_time_in_seconds = float(request["time_limit"])
    solver.parameters.log_search_progress = False
    started = cpu_seconds()
    status = solver.Solve(model)
    elapsed = cpu_seconds() - started
    response = solver.ResponseProto()
    name = STATUS.get(status, str(status))
    reply = {"status": name, "cpu_seconds": elapsed, "wall_seconds": response.wall_time,
             "deterministic_time": response.deterministic_time, "conflicts": response.num_conflicts,
             "branches": response.num_branches, "num_variables": len(proto.variables),
             "num_constraints": len(proto.constraints)}
    if name in ("OPTIMAL", "FEASIBLE"):
        reply["solution"] = list(response.solution)
    if name == "MODEL_INVALID":
        reply["validation"] = model.Validate()[:500]
    return reply


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = read_exact(stdin, 8)
        if header is None:
            return 0
        json_size, proto_size = struct.unpack("<II", header)
        request = json.loads(read_exact(stdin, json_size))
        proto_bytes = read_exact(stdin, proto_size)
        try:
            reply = solve(request, proto_bytes)
        except Exception as error:  # noqa: BLE001 - reported to the evaluator, never fatal here
            reply = {"status": "WORKER_ERROR", "error": f"{type(error).__name__}: {error}"[:500]}
        payload = json.dumps(reply).encode()
        stdout.write(struct.pack("<I", len(payload)) + payload)
        stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
