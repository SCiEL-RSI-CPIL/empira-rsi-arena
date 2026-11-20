"""History-based test case prioritisation in continuous integration (ABB Paint Control and IOF/ROL data).

The data sets of Spieker et al. (ISSTA 2017; repository HelgeS/atcs-data, MIT
licence) record, for every CI cycle of two ABB Robotics systems, the executed
test cases with their approximate duration and verdict. Preparation merges
repeated executions of one test within a cycle (failed if any execution
failed) and keeps the cycles in order; the original row order, execution ids
and the data set's LastRun/LastResults columns are not passed on.

The candidate orders each cycle's tests before that cycle's verdicts are
revealed: a trusted driver in the sandboxed child asks the parent for the next
cycle, calls the candidate, commits the order to the parent and only then
receives the verdicts, which it appends to the history. The first order
committed for a cycle is final. The score is the mean APFD over the cycles
that contain both failing and passing tests (each failing test counts as one
fault; ties are impossible because the order is a permutation).
"""
from __future__ import annotations

import csv
import io
import json
import statistics
from pathlib import Path

from . import software_data, software_sandbox
from .runtime import CandidateError

DATASETS = {"paintcontrol": "atcs-paintcontrol", "iofrol": "atcs-iofrol"}
SOURCES = tuple(DATASETS.values())
DRIVER = '''

def _artask_tcp_run(oracle):
    history = []
    while True:
        cycle = oracle({"request": "next"})
        if cycle is None:
            return len(history)
        tests = [dict(t) for t in cycle["tests"]]
        order = prioritize(cycle, history)
        verdicts = oracle({"request": "order", "order": order})
        history.append({"cycle": cycle["cycle"], "tests": [dict(t, failed=bool(v)) for t, v in zip(tests, verdicts)]})
'''


def derive(raw):
    files = {}
    for name, source in DATASETS.items():
        cycles = {}
        rows = sorted(csv.DictReader(io.StringIO(raw[source].decode("utf-8")), delimiter=";"), key=lambda r: int(r["Id"]))
        for row in rows:
            tests = cycles.setdefault(int(row["Cycle"]), {})
            test, duration, failed = int(row["Name"]), int(row["Duration"]), row["Verdict"].strip() == "1"
            if row["Verdict"].strip() not in ("0", "1"):
                raise ValueError(f"{name}: unexpected verdict {row['Verdict']!r}")
            previous = tests.get(test)
            tests[test] = [test, duration, failed or (previous is not None and previous[2])]
        data = [{"cycle": c, "tests": [tests[t] for t in sorted(tests)]} for c, tests in sorted(cycles.items())]
        files[f"tcp/{name}.json"] = json.dumps(data, separators=(",", ":")).encode()
    return files


def apfd(order, failing):
    """Average percentage of faults detected for one order (1-based positions)."""
    n, m = len(order), len(failing)
    positions = [i + 1 for i, test in enumerate(order) if test in failing]
    return 1.0 - sum(positions) / (n * m) + 1.0 / (2 * n)


class _Protocol:
    """Parent side of the online protocol: reveals a cycle's verdicts only after its order is committed."""

    def __init__(self, cycles):
        self.cycles, self.index, self.waiting, self.orders = cycles, 0, False, []

    def __call__(self, request):
        if not isinstance(request, dict) or request.get("request") not in ("next", "order"):
            raise CandidateError("malformed protocol request")
        if request["request"] == "next":
            if self.waiting:
                raise CandidateError("the previous cycle has not been ordered")
            if self.index == len(self.cycles):
                return None
            self.waiting = True
            cycle = self.cycles[self.index]
            return {"cycle": cycle["cycle"], "tests": [{"id": t, "duration": d} for t, d, _ in cycle["tests"]]}
        if not self.waiting:
            raise CandidateError("no cycle is waiting for an order")
        cycle = self.cycles[self.index]
        order = request.get("order")
        ids = [t for t, _, _ in cycle["tests"]]
        if not isinstance(order, list) or not all(type(t) is int for t in order) or sorted(order) != ids:
            raise CandidateError(f"prioritize() must return a permutation of the {len(ids)} test ids of cycle {cycle['cycle']}")
        self.orders.append(order)
        self.waiting = False
        self.index += 1
        return [failed for _, _, failed in cycle["tests"]]


def tcp_score(module, dataset, *, cpu_seconds=60):
    path = software_sandbox.require_entry(module, "prioritize")
    cycles = software_data.load_json(f"tcp/{dataset}.json")
    protocol = _Protocol(cycles)
    source = Path(path).read_text(encoding="utf-8") + DRIVER
    try:
        software_sandbox.execute(source, "_artask_tcp_run", [], oracle=protocol, cpu_seconds=cpu_seconds,
                                 memory_mb=2048, filename="candidate.py")
    except software_sandbox.SandboxFailure as failure:
        raise CandidateError(f"candidate prioritize() failed in the sandbox ({failure.kind}): {failure}"[:2000]) from None
    if protocol.index != len(cycles):
        raise CandidateError(f"only {protocol.index} of {len(cycles)} cycles were ordered")
    values, optimal, all_failing = [], [], []
    for cycle, order in zip(cycles, protocol.orders):
        failing = {t for t, _, failed in cycle["tests"] if failed}
        if failing:
            all_failing.append(apfd(order, failing))
        if 0 < len(failing) < len(cycle["tests"]):
            values.append(apfd(order, failing))
            optimal.append(apfd(sorted(failing) + [t for t, _, f in cycle["tests"] if not f], failing))
    return statistics.fmean(values), {"dataset": dataset, "cycles": len(cycles), "scored_cycles": len(values),
                                      "optimal_score": statistics.fmean(optimal), "random_order_expectation": 0.5,
                                      "mean_apfd_all_failing_cycles": statistics.fmean(all_failing)}
