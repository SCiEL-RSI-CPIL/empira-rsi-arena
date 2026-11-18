"""EPFL combinational benchmarks: the candidate returns an ABC command script; the evaluator runs it with ABC and then
checks the network that the script leaves, not the script's own output.

1. ABC runs `read <circuit>; <script>; write_blif <temporary file>` in an empty temporary directory.
2. The written network is parsed here: it must be a LUT network (`.names` nodes only) in which no node has more than 6 inputs.
3. A second ABC process, which runs no candidate command, checks `cec <circuit> <temporary file>`.
4. The score is the LUT count or the number of LUT levels by the convention of the EPFL best-results table: a node with two
   or more inputs is a LUT; single-input nodes (output buffers and inverters) and constants are not counted and add no level.

Requires the `abc` binary (Berkeley ABC): set ARNEW_ABC or put it on PATH. The script may not read or write files, leave
the session or call the shell (see `check_script`); a synthesis script needs none of that. The search can tune ABC scripts
(like ABC-DE); it cannot express the SAT-based exact synthesis that holds most records.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .common import CandidateError, read_data, require, data_dir

MAX_LUT_INPUTS = 6
SCRIPT_SECONDS = 1200
CEC_SECONDS = 900  # the equivalence check of the largest circuit (hypotenuse) takes about 450 s; 420 was too short for it
# Characters no synthesis command needs: shell escape and anything that can form a path.
FORBIDDEN_CHARACTERS = ("!", "/", "\\", "~")
# Commands (first word of each `;`-separated command, case-insensitive) that read or write files, change the session or
# start other programs, with the abc.rc aliases of the file commands. Matched on the whole command name, so `rewrite` and
# `resub` are allowed.
FORBIDDEN_COMMAND = re.compile(
    r"&?(read|write|load|dump|save)\w*|&r|&w|[rw]|r[ablptv]|rla|rvl|rsup|rlibc?|rty|rlut|w[abchlpv]|backup|restore|source|so|history|"
    r"alias|unalias|set|sa|unset|cd|ls|pwd|quit|q|exit|sis|mvsis|capo|starter|scrgen|trt", re.I)


def abc_binary():
    local = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools", "abc", "abc")
    path = os.environ.get("ARNEW_ABC") or (local if os.path.isfile(local) else None) or shutil.which("abc") or shutil.which("yosys-abc")
    if not path:
        raise FileNotFoundError("ABC not found; install Berkeley ABC and set ARNEW_ABC or put `abc` on PATH")
    return path


def check_script(script):
    """Reject scripts that could touch files or the session. The result is verified from the written network whatever the
    script does, so this is hygiene (no stray files, no reading of published circuits), not the soundness check."""
    require(isinstance(script, str) and script.strip(), "return a non-empty ABC script string")
    require(not any(ch in script for ch in FORBIDDEN_CHARACTERS), "the script may not contain '!', '/', '\\' or '~'")
    for command in re.split(r"[;\n]", script):
        words = command.split()
        if words:
            require(not FORBIDDEN_COMMAND.fullmatch(words[0]), f"the script may not use the command `{words[0]}` (no file access, no session commands)")
    ks = [int(k) for k in re.findall(r"-K\s*(\d+)", script)]
    require(not ks or max(ks) <= MAX_LUT_INPUTS, f"map to LUTs with at most {MAX_LUT_INPUTS} inputs (`if -K 6` or smaller)")


def blif_stats(path):
    """(LUT count, LUT levels, largest node input count) of a mapped BLIF by the EPFL convention (checked on 7 published
    circuits): nodes with two or more inputs are LUTs; single-input nodes and constants are not counted and add no level.
    Raises CandidateError when the file is not a plain LUT network."""
    nodes, inputs, outputs = {}, [], []
    text = Path(path).read_text().replace("\\\n", " ")
    for line in text.split("\n"):
        t = line.split()
        if not t:
            continue
        if t[0] == ".inputs":
            inputs += t[1:]
        elif t[0] == ".outputs":
            outputs += t[1:]
        elif t[0] == ".names":
            require(len(t) >= 2, "a .names line has no output")
            nodes[t[-1]] = t[1:-1]
        elif t[0] in (".gate", ".subckt", ".latch", ".mlatch"):
            raise CandidateError(f"the final network is not a LUT network (it contains `{t[0]}`)")
    require(bool(outputs), "the written network has no outputs")
    lvl = {x: 0 for x in inputs}
    for o in outputs:
        stack, on_path = [o], set()
        while stack:
            y = stack[-1]
            if y in lvl:
                stack.pop()
                continue
            require(y in nodes, f"signal {y} of the written network has no driver")
            pend = [i for i in nodes[y] if i not in lvl]
            if pend:
                require(not any(i in on_path for i in pend) and y not in pend, "the written network has a combinational loop")
                on_path.add(y)
                stack.extend(pend)
                continue
            lvl[y] = max([lvl[i] for i in nodes[y]], default=0) + (1 if len(nodes[y]) >= 2 else 0)
            on_path.discard(y)
            stack.pop()
    return sum(1 for v in nodes.values() if len(v) >= 2), max(lvl[o] for o in outputs), max((len(v) for v in nodes.values()), default=0)


def evaluate(script, circuit, metric):
    check_script(script)
    read_data("epfl", circuit + ".aig")
    src = (data_dir("epfl") / (circuit + ".aig")).resolve()
    abc = os.path.abspath(abc_binary())
    with tempfile.TemporaryDirectory(prefix="arnew_epfl_") as tmp:
        out = Path(tmp) / "mapped.blif"
        try:
            run = subprocess.run([abc, "-c", f"read {src}; {script.strip().rstrip(';')}; write_blif {out}"],
                                 capture_output=True, text=True, timeout=SCRIPT_SECONDS, cwd=tmp)
        except subprocess.TimeoutExpired:
            raise CandidateError(f"the script ran longer than {SCRIPT_SECONDS} s") from None
        tail = (run.stdout + run.stderr).strip()[-300:]
        require(out.is_file() and out.stat().st_size > 0, f"ABC did not finish the script and write the network; last output: {tail}")
        luts, levels, widest = blif_stats(out)
        require(widest <= MAX_LUT_INPUTS, f"the final network has a node with {widest} inputs; every node must be a LUT with at most {MAX_LUT_INPUTS} inputs")
        try:
            cec = subprocess.run([abc, "-c", f"cec {src} {out}"], capture_output=True, text=True, timeout=CEC_SECONDS, cwd=tmp)
        except subprocess.TimeoutExpired:
            raise CandidateError(f"ABC cec did not finish in {CEC_SECONDS} s, so equivalence is not confirmed") from None
        verdict = cec.stdout + cec.stderr
        require("Networks are equivalent" in verdict and "NOT EQUIVALENT" not in verdict.upper(),
                "ABC cec did not confirm that the final network is equivalent to the original circuit")
    value = luts if metric == "size" else levels
    return value, {"luts": luts, "levels": levels, "largest_lut_inputs": widest, "metric": metric,
                   "verification": "abc_cec_of_written_network_and_lut_input_check"}
