"""LiveCodeBench as used by SICA (self-improving coding agent): functional LeetCode problems.

SICA's loader (base_agent/src/benchmarks/livecodebench.py of
MaximeRobeyns/self_improving_coding_agent, MIT) reads ``release_v5`` of
livecodebench/code_generation_lite, shuffles it with ``random.seed(2)`` and keeps
the first problems whose tests are functional (stdin problems are skipped).
``LCB_SICA`` lists the first 50 such problems (the paper's subset size) as
byte ranges of the release files, found by running that procedure on the
full files; the prepare script fetches only these lines.  A problem's score is
the fraction of its public and private tests passed by ``solution(*args)``.
SICA's loader drops the private tests of 20 of these problems (their base64
text lacks '=' padding, so it never decompresses them); ``private_tests:
"sica_loader"`` reproduces that, the default ``"all"`` uses every test.
SICA's scorer puts LiveCodeBench's import block (``IMPORTS``) in front of a
solution that lacks it, so LeetCode-style code that uses ``List``, ``Counter``
or ``inf`` without importing them runs; the same names are provided here.
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import pickle
import zlib

from . import llm_bench_core as core

LCB = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LICENSE = ("livecodebench/code_generation_lite (dataset card: license 'cc'; problems scraped from LeetCode, AtCoder and Codeforces contests). "
           "Only 50 problem lines are fetched; not redistributed by this repository.")
LCB_SICA = [  # (release_v5 position, file, byte offset, byte length) in SICA's selection order
    (583, "test3.jsonl", 599652556, 2756), (875, "test5.jsonl", 549574163, 4110741), (483, "test2.jsonl", 691748665, 2848),
    (696, "test4.jsonl", 1174784515, 11772), (141, "test.jsonl", 90039424, 884917), (691, "test4.jsonl", 1171889825, 1875507),
    (504, "test2.jsonl", 705246778, 3135), (848, "test5.jsonl", 531106915, 1637104), (877, "test5.jsonl", 555468156, 2225),
    (494, "test2.jsonl", 696070832, 435863), (10, "test.jsonl", 10455959, 370062), (610, "test3.jsonl", 619727691, 240061),
    (687, "test4.jsonl", 1168587124, 2365), (146, "test.jsonl", 92694549, 509911), (9, "test.jsonl", 10449679, 6280),
    (579, "test3.jsonl", 596397957, 2591), (145, "test.jsonl", 92429189, 265360), (73, "test.jsonl", 48005214, 9506),
    (135, "test.jsonl", 87516595, 2976), (699, "test4.jsonl", 1177220630, 6415), (591, "test3.jsonl", 608256732, 10596),
    (593, "test3.jsonl", 608397127, 38348), (122, "test.jsonl", 81439213, 2305), (590, "test3.jsonl", 608254854, 1878),
    (480, "test2.jsonl", 683509875, 6113147), (53, "test.jsonl", 36090222, 34779), (189, "test.jsonl", 110549042, 2926),
    (854, "test5.jsonl", 539107989, 2417866), (54, "test.jsonl", 36125001, 696796), (55, "test.jsonl", 36821797, 3983581),
    (463, "test2.jsonl", 649875725, 273698), (568, "test3.jsonl", 574426134, 2123), (153, "test.jsonl", 97070423, 1289850),
    (106, "test.jsonl", 69070965, 14640), (148, "test.jsonl", 94931491, 3324), (608, "test3.jsonl", 618895264, 827647),
    (471, "test2.jsonl", 670032875, 2762698), (178, "test.jsonl", 106115617, 619462), (59, "test.jsonl", 41876928, 2473),
    (679, "test4.jsonl", 1148266043, 3584805), (62, "test.jsonl", 42930814, 3186), (112, "test.jsonl", 71573212, 2820),
    (67, "test.jsonl", 44209199, 2511), (506, "test2.jsonl", 705642791, 4879627), (477, "test2.jsonl", 683295056, 2418),
    (66, "test.jsonl", 42952598, 1256601), (172, "test.jsonl", 106073755, 2160), (508, "test2.jsonl", 710530697, 4911),
    (35, "test.jsonl", 22468343, 515399), (123, "test.jsonl", 81441518, 23022),
]
FILES = sorted({entry[1] for entry in LCB_SICA})
# The import block of LiveCodeBench's own runner, as SICA's score_problem prepends it (IMPORT_STRING of the loader named above).
IMPORTS = "".join(f"from {name} import *\n" for name in ("string", "re", "datetime", "collections", "heapq", "bisect", "copy", "math", "random", "statistics",
                                                         "itertools", "functools", "operator", "io", "sys", "json", "builtins", "typing")) + "".join(
    f"import {name}\n" for name in ("string", "re", "datetime", "collections", "heapq", "bisect", "copy", "math", "random", "statistics", "itertools",
                                    "functools", "operator", "io", "sys", "json"))


def source_specs():
    return {f"lcb-sica-{name[:-6]}": {"url": f"{LCB}/{name}", "ranges": [[offset, length] for _, file, offset, length in LCB_SICA if file == name],
                                      "license": LICENSE} for name in FILES}


class _PlainUnpickler(pickle.Unpickler):
    """LiveCodeBench pickles plain strings or lists; refuse every global so no code can run."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def private_tests(text):
    value = _PlainUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(text.encode("utf-8"))))).load()
    return value if isinstance(value, list) else json.loads(value)


def derive(paths):
    rows = {}
    for name in FILES:
        data = paths[f"lcb-sica-{name[:-6]}"].read_bytes()
        cursor = 0
        for _, file, offset, length in sorted((entry for entry in LCB_SICA if entry[1] == name), key=lambda entry: entry[2]):
            rows[(file, offset)] = json.loads(data[cursor:cursor + length])
            cursor += length
    out = []
    for rank, (position, file, offset, _) in enumerate(LCB_SICA):
        row = rows[(file, offset)]
        public = json.loads(row["public_test_cases"])
        if row["platform"] != "leetcode" or not public or public[0]["testtype"] != "functional":
            raise ValueError(f"unexpected problem at release position {position}")
        tests = [{"input": t["input"], "output": t["output"]} for t in public + private_tests(row["private_test_cases"])]
        # SICA's loader only decompresses private tests whose base64 text contains '=' (padding); for the
        # other problems it silently keeps the public tests alone. Recorded so that mode can be replicated.
        out.append({"id": row["question_id"], "rank": rank, "release_position": position, "title": row["question_title"], "difficulty": row["difficulty"],
                    "contest_date": row["contest_date"], "content": row["question_content"], "public": len(public), "tests": tests,
                    "sica_loads_private": "=" in row["private_test_cases"]})
    return json.dumps(out, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _example(test):
    args = ", ".join(str(json.loads(line)) for line in test["input"].strip().splitlines())
    return f"Call:\nsolution({args})\n\nExpected output:\n{json.loads(test['output'])}"


def statement(problem):
    """SICA's problem layout (title, difficulty, description, required function, public examples), worded locally."""
    examples = "\n\n".join(f"### Example {i}\n{_example(test)}" for i, test in enumerate(problem["tests"][:problem["public"]], 1))
    return (f"# {problem['title']}\n\nDifficulty: {problem['difficulty']}\n\n## Problem description\n\n{problem['content']}\n\n"
            "## What to submit\n\nWrite a Python function `solution(*args)`. It is called with the example's arguments in order "
            "and its return value is compared with the expected output. Return the complete code in one ```python block.\n\n"
            f"## Examples\n\n{examples}\n")


def _plain(value):
    """Decode the harness encoding; tuples become lists as in SICA's comparison."""
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        if "__tuple__" in value:
            return [_plain(v) for v in value["__tuple__"]]
        if "__set__" in value:
            return ("set", tuple(json.dumps(v, sort_keys=True) for v in value["__set__"]))
        if "__dict__" in value:
            return {k if k is None or isinstance(k, (str, int, float, bool)) else ("key", json.dumps(k, sort_keys=True)): _plain(v)
                    for k, v in value["__dict__"]}
    return value


def _close(result, expected):
    """abs(float(r) - float(e)) < 1e-6; an integer too large for a float fails the test, as SICA's template reports an error."""
    try:
        return abs(float(result) - float(expected)) < 1e-6
    except OverflowError:
        return False


def matches(result, expected):
    """SICA's functional check: equality, else numbers or lists of numbers within 1e-6 (bools count as ints, as there)."""
    result = _plain(result)
    if result == expected:
        return True
    number = (int, float)
    if isinstance(result, list) and isinstance(expected, list):
        return (len(result) == len(expected) and all(isinstance(r, number) for r in result) and all(isinstance(e, number) for e in expected)
                and all(_close(r, e) for r, e in zip(result, expected)))
    if isinstance(result, number) and isinstance(expected, number):
        return _close(result, expected)
    return False


def items(P):
    from .llm_bench_code import extract_code, run_outputs
    problems = json.loads(gzip.decompress(core.data_path("livecodebench_sica.json.gz").read_bytes()))
    per_test, total = float(P.get("per_test", 20.0)), float(P.get("time_limit", 60.0))
    sica_loader = P.get("private_tests", "all") == "sica_loader"
    out = [{"id": f"lcb-{p['id']}", "question": statement(p),
            "tests": p["tests"][:p["public"]] if sica_loader and not p["sica_loads_private"] else p["tests"]} for p in problems]

    def score(item, reply):
        outputs = run_outputs(extract_code(reply), [t["input"] for t in item["tests"]], "lcb", time_limit=total, per_test=per_test, prelude=IMPORTS)
        passed = [output is not None and matches(output, json.loads(t["output"])) for output, t in zip(outputs, item["tests"])]
        return {"score": sum(passed) / len(passed), "tests_passed": sum(passed), "tests": len(passed)}
    return out, score
