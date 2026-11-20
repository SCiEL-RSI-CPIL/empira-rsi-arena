"""Multiple protein sequence alignment accuracy on BAliBASE reference alignments.

BAliBASE 3 (Thompson, Koehl, Ripp and Poch 2005) provides structure-based reference
alignments with annotated core blocks. The candidate aligns the unaligned sequences of
each test case; the evaluator compares its alignment with the reference on the core-block
columns and reports the sum-of-pairs (SP) and total-column (TC) scores defined by Thompson,
Plewniak and Poch (1999). This is an independent implementation of those definitions, not
a port of the bundled bali_score program.
"""
from __future__ import annotations

import re
import tarfile

import numpy as np

from .bio_physics_common import load_derived, run_sandboxed
from .runtime import require

RAW_SOURCES = ("balibase-BAliBASE_R1-5.tar.gz",)
DATA = "align/balibase_rv11_rv12.json.gz"
DERIVED = {DATA: "de163cbb3425178dccff2ebe60b9f752038b02f883b4d1b6f027e5bdb55be962"}

# Reference 1 test cases (equidistant sequences; RV11 very divergent, RV12 medium divergence)
# small enough for a CPU-only aligner: at most 10 sequences, at most 300 reference columns.
INSTANCES = (
    "BBS11001", "BBS11008", "BBS11009", "BBS11010", "BBS11011", "BBS11017", "BBS11020", "BBS11021",
    "BBS11027", "BBS11029", "BBS11032", "BBS12002", "BBS12005", "BBS12006", "BBS12009", "BBS12014",
    "BBS12016", "BBS12020", "BBS12022", "BBS12024", "BBS12025", "BBS12028", "BBS12030", "BBS12036",
    "BBS12040", "BBS12041",
)


def parse_reference(text):
    """BAliBASE XML -> (names, aligned rows, core-block column mask)."""
    pairs = re.findall(r"<seq-name>(.*?)</seq-name>.*?<seq-data>\s*(.*?)\s*</seq-data>", text, re.S)
    names = [name.strip() for name, _ in pairs]
    rows = [re.sub(r"\s+", "", row).upper() for _, row in pairs]
    mask_match = re.search(r"<colsco-data>\s*([01\s]*?)</colsco-data>", text)
    require(mask_match is not None, "reference alignment has no core-block annotation")
    mask = [int(c) for c in mask_match.group(1).split()]
    require(rows and len({len(r) for r in rows}) == 1 and len(mask) == len(rows[0]),
            "reference alignment rows and core-block mask disagree")
    return names, rows, mask


def derive(paths):
    wanted = set(INSTANCES)
    cases = {}
    with tarfile.open(paths[RAW_SOURCES[0]]) as archive:
        for member in archive.getmembers():
            name = member.name.rsplit("/", 1)[-1]
            if not name.endswith(".xml") or name[:-4] not in wanted:
                continue
            names, rows, mask = parse_reference(archive.extractfile(member).read().decode("latin-1"))
            cases[name[:-4]] = {"names": names, "reference": rows, "core": mask,
                                "sequences": [row.replace("-", "") for row in rows]}
    missing = wanted - set(cases)
    require(not missing, f"instances missing from the archive: {sorted(missing)}")
    return {DATA: {"source": RAW_SOURCES[0], "reference_sets": ["RV11", "RV12"], "cases": cases}}


def load_cases():
    payload = load_derived(DATA, DERIVED[DATA])
    return [(name, payload["cases"][name]) for name in INSTANCES]


def _core_pairs(rows, mask):
    """Set of ((sequence i, residue index), (sequence j, residue index)) pairs in core columns."""
    positions = [-1] * len(rows)
    pairs, columns = set(), []
    for column in range(len(rows[0])):
        present = []
        for s, row in enumerate(rows):
            if row[column] != "-":
                positions[s] += 1
                present.append((s, positions[s]))
        if mask[column]:
            columns.append(tuple(present))
            for a in range(len(present)):
                for b in range(a + 1, len(present)):
                    pairs.add((present[a], present[b]))
    return pairs, columns


def _test_pairs(rows):
    positions = [-1] * len(rows)
    pairs, columns = set(), set()
    for column in range(len(rows[0])):
        present = []
        for s, row in enumerate(rows):
            if row[column] != "-":
                positions[s] += 1
                present.append((s, positions[s]))
        columns.add(tuple(present))
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                pairs.add((present[a], present[b]))
    return pairs, columns


def check_alignment(alignment, sequences):
    """Validate that the candidate's rows are the given sequences with gaps inserted."""
    require(isinstance(alignment, (list, tuple)) and len(alignment) == len(sequences),
            f"alignment: expected {len(sequences)} rows")
    rows = []
    for index, (row, sequence) in enumerate(zip(alignment, sequences)):
        require(isinstance(row, str), f"alignment row {index}: expected a string")
        upper = row.upper()
        require(upper.replace("-", "") == sequence,
                f"alignment row {index}: ungapped residues differ from the input sequence")
        rows.append(upper)
    require(len({len(r) for r in rows}) == 1, "alignment: all rows must have the same length")
    require(len(rows[0]) <= 6 * max(len(s) for s in sequences) + 10, "alignment: unreasonably long (too many gap columns)")
    return rows


def case_scores(case, alignment):
    """(SP, TC) of one alignment against the reference core blocks."""
    rows = check_alignment(alignment, case["sequences"])
    reference_pairs, reference_columns = _core_pairs(case["reference"], case["core"])
    test_pairs, test_columns = _test_pairs(rows)
    require(reference_pairs, "reference has no core-block residue pairs")
    sp = len(reference_pairs & test_pairs) / len(reference_pairs)
    tc = sum(1 for column in reference_columns if column in test_columns) / len(reference_columns)
    return sp, tc


DRIVER = '''

def _bp_align_driver(cases):
    _bp_seed()
    return [_bp_plain(align(case["sequences"])) for case in cases]
'''


def score(module, P):
    cases = load_cases()
    public = [{"name": name, "sequences": list(case["sequences"])} for name, case in cases]
    answers = run_sandboxed(module, DRIVER, "_bp_align_driver", [public], required=("align",),
                            cpu_seconds=int(P.get("cpu_seconds", 240)), memory_mb=2048)
    require(isinstance(answers, list) and len(answers) == len(cases), f"expected {len(cases)} alignments")
    rows = []
    for (name, case), alignment in zip(cases, answers):
        sp, tc = case_scores(case, alignment)
        rows.append({"case": name, "sequences": len(case["sequences"]), "sp": sp, "tc": tc})
    sp_values = [r["sp"] for r in rows]
    return float(np.mean(sp_values)), {
        "cases": rows, "mean_sp": float(np.mean(sp_values)), "mean_tc": float(np.mean([r["tc"] for r in rows])),
        "mean_sp_rv11": float(np.mean([r["sp"] for r in rows if r["case"].startswith("BBS11")])),
        "mean_sp_rv12": float(np.mean([r["sp"] for r in rows if r["case"].startswith("BBS12")])),
        "worst_case": min(rows, key=lambda r: r["sp"])["case"]}
