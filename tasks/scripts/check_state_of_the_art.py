#!/usr/bin/env python3
"""Recompute the state-of-the-art values that rest on a finite computation.

Several prize-bank instances have an optimum that follows from an exact
computation (sometimes together with a published theorem) rather than from a
published table alone. For each of them this script

* rebuilds a witness and scores it with the task's own ``evaluator.py``, so the
  recorded value is attained under the published contract, and
* where the optimality argument is computational, repeats that computation
  (the functions are in scripts/state_of_the_art_computations.py).

The recorded values are read from the task manifests (``catalog/tasks.json``).
A task whose manifest has no record gets its witness scored and no value
compared. Run from the task library:

    python3 scripts/check_state_of_the_art.py
    python3 scripts/check_state_of_the_art.py --only science-lonely-runner-rational-time

Results go to ``receipts/prize-expansion/state-of-the-art-checks.json`` in the
task cache (see ARENA_TASK_CACHE_DIR), never into the task library. ``--slow``
adds the two integer-programming optimality proofs (a few minutes; needs SciPy).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "suite" / "_lib"), str(Path(__file__).resolve().parent)]
from arena_task_paths import receipt_path  # noqa: E402
from state_of_the_art_computations import (  # noqa: E402
    anti_ramsey, anti_ramsey_small, blocking_set, cluster_prime, distinct_subset_sums, egyptian, fano,
    farey_pair, gaussian_moat, goldbach, lonely_runner, modular_inverse_cover, no_singleton, permutation_order,
    permutation_progression, prime_complement, primitive_roots, property_p, smooth_antichain, squarefree_gap,
    tournament, triple_free, two_square_gap, van_der_waerden,
)

RECEIPT = "prize-expansion/state-of-the-art-checks.json"

CHECKS = {
    "science-cluster-prime-difference-coverage": cluster_prime,
    "science-consecutive-squarefree-gap": squarefree_gap,
    "science-consecutive-two-square-gap": two_square_gap,
    "science-lonely-runner-rational-time": lonely_runner,
    "science-multiplicative-triple-free-4096": triple_free,
    "science-prime-additive-complement-256": prime_complement,
    "science-property-p-divisibility-256": property_p,
    "science-gaussian-prime-moat-path-50": gaussian_moat,
    "science-projective-blocking-intersections-pg27": blocking_set,
    "science-oppositely-ordered-farey-pair-1000": farey_pair,
    "science-smooth-divisibility-antichain-sums-128": smooth_antichain,
    "science-distinct-modular-inverse-cover-101": modular_inverse_cover,
    "science-small-prime-primitive-roots": primitive_roots,
    "science-most-frequent-permutation-order-16": permutation_order,
    "science-anti-ramsey-four-cycle-colors-16": anti_ramsey,
    "science-transitive-subtournament-minimum-20": tournament,
    "science-two-color-four-progression-avoidance": van_der_waerden,
    "science-permutation-monotone-progression-128": permutation_progression,
    "science-goldbach-prime-decompositions": goldbach,
    "science-distinct-three-unit-fractions-four-over-n": egyptian,
    "science-three-chromatic-triple-system": fano,
    "science-distinct-subset-sums-12": distinct_subset_sums,
    "science-uniform-no-singleton-intersections": no_singleton,
}
# Witness-only rows: the value is the published record; the certificate is scored.
WITNESSES = {
    "science-coprime-abc-quality": [2, 6436341, 6436343],
    "science-jacobsthal-five-prime-gaps": 2310,
    "science-number-theory-powerful-consecutive-counterexample": {"n": 12167, "factorizations": [[[23, 3]], [[2, 3], [3, 2], [13, 2]]]},
}


def recorded_values():
    """The reviewed record of each checked task, or None when its manifest has none."""
    tasks = {t["id"]: t for t in json.loads((ROOT / "catalog/tasks.json").read_text())}
    missing = [tid for tid in list(CHECKS) + list(WITNESSES) if tid not in tasks]
    if missing:
        raise SystemExit("not in catalog/tasks.json: " + ", ".join(missing))
    return {tid: tasks[tid].get("state_of_the_art") for tid in list(CHECKS) + list(WITNESSES)}


def score(tid, witness):
    with tempfile.TemporaryDirectory() as tmp:
        program = Path(tmp) / "witness.py"
        program.write_text("def construct():\n    return " + repr(witness) + "\n")
        run = subprocess.run([sys.executable, str(ROOT / "suite" / tid / "evaluator.py"), str(program)],
                             cwd=ROOT, capture_output=True, text=True, timeout=600)
    try:
        report = json.loads(run.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"{tid}: evaluator did not return JSON: {run.stderr[-500:]}")
    return report


def close(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slow", action="store_true", help="also run the integer-programming optimality proofs")
    parser.add_argument("--only", help="check a single task id")
    args = parser.parse_args()
    recorded = recorded_values()
    rows, failures = [], []
    for tid in list(CHECKS) + list(WITNESSES):
        if args.only and tid != args.only:
            continue
        start = time.time()
        record = recorded[tid]
        target = None if record is None else record["value"]
        if tid in CHECKS:
            fn = CHECKS[tid]
            value, witness, method = fn(slow=args.slow) if fn in (prime_complement, property_p) else fn()
        elif record is None:
            value, witness, method = None, WITNESSES[tid], "witness scored; the manifest records no reference value"
        else:
            value, witness, method = target, WITNESSES[tid], "published record; witness scored"
        report = score(tid, witness)
        computed_ok = value is None or target is None or close(value, target)
        witness_ok = report.get("valid") is True and (target is None or close(report.get("score"), target))
        row = dict(id=tid, recorded_kind=None if record is None else record["kind"], recorded_value=target, computed_value=value,
                   witness_score=report.get("score"), witness_valid=report.get("valid"), method=method,
                   passed=computed_ok and witness_ok, seconds=round(time.time() - start, 2))
        rows.append(row)
        if not row["passed"]:
            failures.append(tid)
        print(f"{tid}: {'PASS' if row['passed'] else 'FAIL'} recorded={target} computed={value} witness={report.get('score')}", flush=True)
    formula_ok, small = anti_ramsey_small()
    report = dict(generated=datetime.now(timezone.utc).isoformat(), slow_checks=args.slow,
                  scope="Witnesses are scored by each task's own evaluator.py; computed optima are recomputed here. Values that rest on a published theorem alone are recorded with their sources in the manifests and are not re-proved.",
                  anti_ramsey_formula_small_cases=dict(passed=formula_ok, exhaustive_maximum=small),
                  tasks=rows, summary=dict(passed=sum(r["passed"] for r in rows), failed=len(failures),
                                           without_recorded_value=[r["id"] for r in rows if r["recorded_value"] is None]))
    if not args.only:
        output = receipt_path(RECEIPT)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {output}")
    if failures or not formula_ok:
        raise SystemExit("Failed: " + ", ".join(failures or ["anti-Ramsey small cases"]))


if __name__ == "__main__":
    main()
