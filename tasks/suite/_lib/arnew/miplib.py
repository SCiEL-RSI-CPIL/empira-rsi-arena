"""MIPLIB 2017 open instances: the candidate returns a full variable vector; feasibility and objective are checked exactly.

The model is read with HiGHS (highspy) from the instance's .mps.gz. Feasibility tolerance 1e-6 absolute on bounds,
integrality and rows, checked on the values as returned (integer variables are not rounded first). This is stricter than
MIPLIB's own check: of 194 published best solutions replayed on 2026-11-11, 157 pass and 37 do not (33 of those violate it
by less than 1e-5), so a result that beats a record here also beats it there, but some records cannot be matched here.
MIPLIB's rule, from the source of its checker (miplib2017-testscript-v1.0.4.zip, checker/src/model.cpp and main.cpp): a bound may be
violated by tol * max(1, |bound|, |x|), a row by tol * max(1, positive part of its activity, |side|), integrality by tol, in exact
rational arithmetic, with tol = 1e-5 by default and 1e-4 in the MIPLIB run scripts (scripts/run.sh). Everything accepted here is
accepted there.
The score is the objective in minimisation form (maximisation models are negated); `objective` in the result is the
value in the model's own sense, which is what MIPLIB lists.
"""
from __future__ import annotations

import gzip
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from .common import CandidateError, data_dir, read_data, require

TOL = 1e-6


def _split_indicators(src_path, dst_path):
    """Copy an MPS file for HiGHS: drop the INDICATORS section (returned as [(row, column, value)]) and cut trailing
    text from ROWS lines (ivu59 has "E  s0  $t114  0"). Only used when HiGHS rejects the original."""
    indicators, section = [], ""
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        for line in src:
            if line[:1] not in (b" ", b"\t", b"*"):
                section = line.split()[0].decode() if line.split() else section
                if section == "INDICATORS":
                    continue
            elif section == "INDICATORS":
                parts = line.split()
                if len(parts) >= 4 and parts[0] == b"IF":
                    indicators.append((parts[1].decode(), parts[2].decode(), float(parts[3])))
                continue
            elif section == "ROWS":
                parts = line.split()
                if len(parts) > 2:
                    line = b" " + parts[0] + b"  " + parts[1] + b"\n"
            dst.write(line)
    return indicators


def _parse_mps(path):
    """Plain MPS reader for files HiGHS rejects (here: matrix entries of 1e20). Returns the arrays `load` needs."""
    import scipy.sparse as sp

    rows, rtype, cols, cidx, ints = {}, [], [], {}, []
    cost, entries, rhs, ranges = {}, [], {}, {}
    lower, upper, section, in_int, objname, offset, maximise = {}, {}, "", False, None, 0.0, False
    bounded = set()  # columns named in BOUNDS
    with open(path) as f:
        for line in f:
            if not line.strip() or line[0] == "*":
                continue
            t = line.split()
            if line[0] != " ":
                section = t[0]
                maximise = maximise or (section == "OBJSENSE")
                continue
            if section in ("OBJSENSE",):
                maximise = t[0].upper().startswith("MAX")
            elif section == "ROWS":
                if t[0] == "N":
                    objname = objname or t[1]
                else:
                    rows[t[1]] = len(rtype)
                    rtype.append(t[0])
            elif section == "COLUMNS":
                if len(t) >= 3 and t[1] == "'MARKER'":
                    in_int = t[2] == "'INTORG'"
                    continue
                if t[0] not in cidx:
                    cidx[t[0]] = len(cols)
                    cols.append(t[0])
                    ints.append(in_int)
                    lower[t[0]], upper[t[0]] = 0.0, np.inf
                for r, v in zip(t[1::2], t[2::2]):
                    if r == objname:
                        cost[t[0]] = float(v)
                    elif r in rows:
                        entries.append((rows[r], cidx[t[0]], float(v)))
            elif section == "RHS":
                for r, v in (zip(t[1::2], t[2::2]) if len(t) % 2 else zip(t[0::2], t[1::2])):
                    if r == objname:
                        offset = -float(v)
                    elif r in rows:
                        rhs[rows[r]] = float(v)
            elif section == "RANGES":
                for r, v in (zip(t[1::2], t[2::2]) if len(t) % 2 else zip(t[0::2], t[1::2])):
                    ranges[rows[r]] = float(v)
            elif section == "BOUNDS":
                kind, col = t[0], t[2]
                val = float(t[3]) if len(t) > 3 else 0.0
                bounded.add(col)
                if kind in ("UP", "UI"):
                    upper[col] = val
                    if val < 0 and lower[col] == 0.0:
                        lower[col] = -np.inf
                elif kind in ("LO", "LI"):
                    lower[col] = val
                elif kind == "FX":
                    lower[col] = upper[col] = val
                elif kind == "FR":
                    lower[col], upper[col] = -np.inf, np.inf
                elif kind == "MI":
                    lower[col] = -np.inf
                elif kind == "PL":
                    upper[col] = np.inf
                elif kind == "BV":
                    lower[col], upper[col] = 0.0, 1.0
                    ints[cidx[col]] = True
                else:
                    raise RuntimeError(f"unsupported MPS bound type {kind}")
                if kind in ("UI", "LI", "BV"):
                    ints[cidx[col]] = True
    for col, is_int in zip(cols, ints):
        if is_int and col not in bounded:
            upper[col] = 1.0  # MPS convention (MIPLIB's solution checker, HiGHS): an integer column without a BOUNDS entry is binary
    m, n = len(rtype), len(cols)
    ri, ci, vv = zip(*entries) if entries else ((), (), ())
    A = sp.csr_matrix((vv, (ri, ci)), shape=(m, n))
    rl, ru = np.full(m, -np.inf), np.full(m, np.inf)
    for i, kind in enumerate(rtype):
        b = rhs.get(i, 0.0)
        if kind == "E":
            rl[i] = ru[i] = b
            if i in ranges:
                rl[i], ru[i] = (b, b + ranges[i]) if ranges[i] >= 0 else (b + ranges[i], b)
        elif kind == "L":
            ru[i] = b
            rl[i] = b - abs(ranges[i]) if i in ranges else -np.inf
        else:
            rl[i] = b
            ru[i] = b + abs(ranges[i]) if i in ranges else np.inf
    c = np.array([cost.get(k, 0.0) for k in cols])
    return {"A": A, "col_cost": c, "col_lower": np.array([lower[k] for k in cols]), "col_upper": np.array([upper[k] for k in cols]),
            "row_lower": rl, "row_upper": ru, "integrality": np.array(ints, dtype=bool), "offset": float(offset),
            "maximise": bool(maximise), "row_names": list(rows), "col_names": cols}


def _highs():
    """A HiGHS instance with output switched off. The first Highs() of a process prints a banner to the C-level stdout before any
    option can be set (highspy 1.16), and the harness reads the evaluator's JSON result from stdout: the banner made every MIPLIB
    task fail in `run.py evaluate` ("Expecting value: line 1 column 1"). It is sent to stderr; later instances print nothing."""
    import ctypes

    import highspy

    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        h = highspy.Highs()
        ctypes.CDLL(None).fflush(None)  # the banner must leave the C buffer while stdout still points at stderr
    finally:
        os.dup2(saved, 1)
        os.close(saved)
    h.setOptionValue("output_flag", False)
    return h


def load(name, workdir):
    import highspy
    import scipy.sparse as sp

    read_data("miplib", name + ".mps.gz")  # existence and manifest check
    mps = Path(workdir) / (name + ".mps")
    with gzip.open(data_dir("miplib") / (name + ".mps.gz"), "rb") as src, open(mps, "wb") as dst:
        shutil.copyfileobj(src, dst)
    indicators = []
    h = _highs()
    status = h.readModel(str(mps))
    if status == highspy.HighsStatus.kError or h.getLp().num_col_ == 0:
        # rejected as is: retry on a cleaned copy (the seed's HiGHS runs then read the same cleaned file; indicator rows are checked by `check`)
        clean = Path(workdir) / (name + ".clean.mps")
        indicators = _split_indicators(mps, clean)
        clean.replace(mps)
        h = _highs()
        status = h.readModel(str(mps))
    lp = h.getLp()
    n, m = lp.num_col_, lp.num_row_
    if status == highspy.HighsStatus.kError or n == 0:
        # still rejected (e.g. matrix entries of 1e20): fall back to the plain reader in this module
        model = _parse_mps(str(mps))
        if not model["col_names"]:
            raise RuntimeError(f"could not read {name}.mps with HiGHS or the plain MPS reader (no columns)")
        model.update({"name": name, "mps_path": str(mps), "num_cols": len(model["col_names"]), "num_rows": len(model["row_names"]), "indicators": []})
        return model
    am = lp.a_matrix_
    start, index, value = np.array(am.start_), np.array(am.index_), np.array(am.value_)
    if am.format_ == highspy.MatrixFormat.kColwise:
        A = sp.csc_matrix((value, index, start), shape=(m, n)).tocsr()
    else:
        A = sp.csr_matrix((value, index, start), shape=(m, n))
    integrality = np.array([int(t) != 0 for t in lp.integrality_]) if len(lp.integrality_) else np.zeros(n, dtype=bool)
    maximise = lp.sense_ == highspy.ObjSense.kMaximize
    ind = []
    if indicators:
        rid = {r: i for i, r in enumerate(lp.row_names_)}
        cid = {c: i for i, c in enumerate(lp.col_names_)}
        ind = [(rid[r], cid[c], v) for r, c, v in indicators]
    return {"indicators": ind, "name": name, "mps_path": str(mps), "num_cols": n, "num_rows": m, "A": A, "col_cost": np.array(lp.col_cost_),
            "col_lower": np.array(lp.col_lower_), "col_upper": np.array(lp.col_upper_), "row_lower": np.array(lp.row_lower_),
            "row_upper": np.array(lp.row_upper_), "integrality": integrality, "offset": float(lp.offset_), "maximise": bool(maximise)}


def check(x, model):
    try:
        x = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise CandidateError("return a sequence of num_cols numbers") from None
    require(len(x) == model["num_cols"], f"return exactly {model['num_cols']} values")
    require(bool(np.isfinite(x).all()), "values must be finite")
    require(bool((x >= model["col_lower"] - TOL).all() and (x <= model["col_upper"] + TOL).all()), "a variable violates its bounds")
    ints = model["integrality"]
    require(bool((np.abs(x[ints] - np.rint(x[ints])) <= TOL).all()), "an integer variable is not integral")
    act = model["A"] @ x
    ok = (act >= model["row_lower"] - TOL) & (act <= model["row_upper"] + TOL)
    for row, col, val in model.get("indicators", ()):  # indicator rows only bind when their binary equals the trigger value
        if abs(x[col] - val) > TOL:
            ok[row] = True
    require(bool(ok.all()), "a constraint is violated")
    obj = float(model["col_cost"] @ x + model["offset"])
    return obj, {"objective": obj, "maximise": model["maximise"], "verification": "exact_feasibility_check"}
