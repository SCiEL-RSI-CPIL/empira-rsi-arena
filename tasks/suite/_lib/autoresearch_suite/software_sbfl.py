"""Spectrum-based fault localisation on seeded faults (mutants) of three CPython modules.

Faults are the first-order mutants of the mutation tasks (software_mutate) of
textwrap, urllib.parse and fractions. Each faulty version is run on a fixed
test pool (software_sbfl_pools); a test fails when its outcome differs from
the original module's. The program spectrum records, for every block probe of
the module (software_harness_pymod), how many failing (ef) and passing (ep)
tests reach it (block hit spectra as in Abreu et al., 2007). Faults that no
test detects, or whose mutated line cannot be assigned to one block, are left
out, as are mutants that break the module's import. The faulty block (the statement-level block containing the mutated
statement) is kept by the evaluator; the candidate sees only the spectra.

The candidate receives each spectrum with the blocks in a fixed pseudo-random
order per fault, so a block's position carries no information about which
blocks tend to be faulty. Score: mean EXAM, the faulty block's rank (ties
broken by the average rank) divided by the number of blocks of the module;
lower is better.
"""
from __future__ import annotations

import io
import json
import math
import random
import statistics
import tarfile

from . import software_data, software_harness_pymod, software_mutate, software_pymod, software_sandbox
from . import software_sbfl_pools
from .runtime import CandidateError

SOURCES = (software_pymod.SOURCE,)
SUBJECTS = ("textwrap", "urllib.parse", "fractions")


def fault_blocks(source):
    """{line: block probe id} for lines that belong to exactly one innermost statement block."""
    import ast
    visitor = software_harness_pymod._Instrument()
    visitor.visit(ast.parse(source))
    owner = {}
    for probe, spans in visitor.blocks.items():
        for first, last in spans:
            for line in range(first, last + 1):
                owner.setdefault(line, []).append((last - first, probe))
    out = {}
    for line, entries in owner.items():
        entries.sort()
        if len(entries) == 1 or entries[0][0] < entries[1][0]:
            out[line] = entries[0][1]
    return out


def derive(raw):
    faults, blocks = [], {}
    with tarfile.open(fileobj=io.BytesIO(raw[software_pymod.SOURCE]), mode="r:xz") as archive:
        for subject in SUBJECTS:
            spec = software_pymod.MUTATION[subject]
            files = software_pymod._patched(archive, spec["files"], {})
            target = spec["target"]
            mutants = software_mutate.mutants(files[target])
            tests = software_pymod.check_tests(subject, software_sbfl_pools.pool(subject), software_sbfl_pools.POOL_SIZE)
            ok, result = software_sandbox.run_harness(software_sandbox.harness_source(software_pymod.HARNESS), "spectra",
                                                      [files, target, subject, tests, [[m["start"], m["end"], m["replacement"]] for m in mutants]],
                                                      cpu_seconds=1800, wall_seconds=3600, memory_mb=3072)
            if not ok:
                raise RuntimeError(f"{subject}: spectra failed: {result}")
            owner = fault_blocks(files[target])
            blocks[subject] = result["blocks"]
            for index, (mutant, spectrum) in enumerate(zip(mutants, result["faults"])):
                if spectrum is None or spectrum["failed"] == 0 or mutant["line"] not in owner:
                    continue
                faults.append({"subject": subject, "mutant": index, "operator": mutant["operator"], "block": owner[mutant["line"]],
                               "failed": spectrum["failed"], "passed": spectrum["passed"], "ef": spectrum["ef"], "ep": spectrum["ep"]})
    data = {"blocks": blocks, "pool_size": software_sbfl_pools.POOL_SIZE, "faults": faults}
    return {"sbfl/faults.json": json.dumps(data, sort_keys=True, separators=(",", ":")).encode()}


def shuffled(fault, index):
    """The fault's spectrum with blocks in a fixed pseudo-random order (block identity carries no information)."""
    order = list(range(len(fault["ef"])))
    random.Random(index).shuffle(order)
    return {"subject": fault["subject"], "blocks": len(order), "failed": fault["failed"], "passed": fault["passed"],
            "ef": [fault["ef"][b] for b in order], "ep": [fault["ep"][b] for b in order]}, order.index(fault["block"])


def tie_bound(fault):
    """Best average rank any function of (ef, ep) can give the faulty block: it shares its pair with k blocks."""
    pair = (fault["ef"][fault["block"]], fault["ep"][fault["block"]])
    k = sum((ef, ep) == pair for ef, ep in zip(fault["ef"], fault["ep"]))
    return (k + 1) / 2 / len(fault["ef"])


def exam(scores, block):
    """Faulty block's average-tie rank divided by the number of blocks."""
    target = scores[block]
    higher = sum(s > target for s in scores)
    ties = sum(s == target for s in scores)
    return (higher + (ties + 1) / 2) / len(scores)


def _formula(fault, f):
    F, P = fault["failed"], fault["passed"]
    return [f(ef, ep, F, P) for ef, ep in zip(fault["ef"], fault["ep"])]


BASELINES = {
    "tarantula": lambda ef, ep, F, P: 0.0 if ef == 0 else (ef / F) / (ef / F + (ep / P if P else 0.0)),
    "ochiai": lambda ef, ep, F, P: 0.0 if ef == 0 else ef / math.sqrt(F * (ef + ep)),
    "dstar2": lambda ef, ep, F, P: ef * ef / (ep + (F - ef)) if ep + (F - ef) else float(ef * ef) + 1e9,
}
DRIVER = '''

def _artask_localize(spectra):
    return [localize(spectrum) for spectrum in spectra]
'''


def sbfl_score(module, *, cpu_seconds=120):
    from pathlib import Path
    path = software_sandbox.require_entry(module, "localize")
    data = software_data.load_json("sbfl/faults.json")
    faults = data["faults"]
    source = Path(path).read_text(encoding="utf-8") + DRIVER
    views = [shuffled(f, i) for i, f in enumerate(faults)]
    try:
        answers = software_sandbox.execute(source, "_artask_localize", [[v for v, _ in views]],
                                           cpu_seconds=cpu_seconds, memory_mb=2048, filename="candidate.py")
    except software_sandbox.SandboxFailure as failure:
        raise CandidateError(f"candidate localize() failed in the sandbox ({failure.kind}): {failure}"[:2000]) from None
    if not isinstance(answers, list) or len(answers) != len(faults):
        raise CandidateError("localize() must return one list of scores per fault")
    values, top = [], {1: 0, 5: 0, 10: 0}
    for fault, (_, position), scores in zip(faults, views, answers):
        if not isinstance(scores, list) or len(scores) != len(fault["ef"]) or not all(
                type(s) in (int, float) and math.isfinite(s) for s in scores):
            raise CandidateError(f"localize() must return {len(fault['ef'])} finite numbers for a {fault['subject']} fault")
        value = exam(scores, position)
        values.append(value)
        for n in top:
            top[n] += value * len(scores) <= n
    baselines = {name: statistics.fmean(exam(_formula(f, fn), f["block"]) for f in faults) for name, fn in BASELINES.items()}
    by_subject = {s: statistics.fmean(v for v, f in zip(values, faults) if f["subject"] == s) for s in data["blocks"]}
    return statistics.fmean(values), {"faults": len(faults), "blocks": data["blocks"], "mean_exam_by_subject": by_subject,
                                      "top_1": top[1], "top_5": top[5], "top_10": top[10], "baseline_mean_exam": baselines,
                                      "spectrum_tie_bound": statistics.fmean(tie_bound(f) for f in faults)}
