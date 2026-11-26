#!/usr/bin/env python3
"""Check the LMX sentiment scorer against the paper's printed values and notebook-style model calls.

LMX (Meyerson et al.) prints a representative Pareto front for each quote in
Tables 3-5: per sentence, its distance from the quote and the positivity score of
the sentiment model. This script parses those rows from the text of the paper
(arXiv 2302.12170, extracted from the PDF with its line breaks; default
primary-paper-text/lmx.txt in the task cache, or --paper), recomputes both
values with the pinned models through autoresearch_suite.llm_bench_lmx, and
also through notebook-style library calls (transformers sentiment pipeline
with top_k=3; SentenceTransformer.encode). It writes the agreement per table
to receipts/lmx-reference-check.json in the task cache (or --out, which must be
outside the task library); the receipt shipped as
validation/lmx-reference-check.json is the source repository's run and is not
overwritten. Paper sentences are identified by SHA-256 only; no paper text is
written.

    python scripts/check_lmx_reference.py [--paper FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite/_lib"))

from arena_task_paths import cache_path, external_path, receipt_path  # noqa: E402

TABLES = {3: "lmx-homer-simpson-sentence-sentiment", 4: "lmx-gore-vidal-sentence-sentiment", 5: "lmx-woody-allen-sentence-sentiment"}
ROW = re.compile(r"^(\d\.\d\d) (\d\.\d{3})(?: (.*))?$")
POSITIVITY_TOLERANCE = 0.0015  # printed with three decimals
DISTANCE_TOLERANCE = 0.0055    # printed with two decimals
MAX_DISTANCE = 1.25            # the tables print the distance clipped at 1.25 and divided by it (the notebook's niche axis)


def table_rows(text, number):
    """(distance, positivity, sentence) rows of the Pareto-front table ``number``, with PDF line wraps undone."""
    end = text.index(f"Table {number}. Full pareto front")
    start = text.rindex("Distance Positivity Sentence", 0, end)
    rows = []
    for line in text[start:end].splitlines()[1:]:
        match = ROW.match(line.strip())
        if match:
            rows.append([float(match.group(1)), float(match.group(2)), match.group(3) or ""])
        elif rows and line.strip() and not line.startswith("[PDF PAGE"):
            sentence = rows[-1][2]
            if sentence.endswith("-") and line[:1].islower():
                rows[-1][2] = sentence[:-1] + line.strip()   # hyphenation at a PDF line break
            else:
                rows[-1][2] = (sentence + " " + line.strip()).strip()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", help="text of the LMX paper (default: primary-paper-text/lmx.txt in the task cache)")
    parser.add_argument("--out", help="receipt to write (default: receipts/lmx-reference-check.json in the task cache)")
    args = parser.parse_args(argv)
    paper = Path(args.paper) if args.paper else cache_path("primary-paper-text", "lmx.txt")
    if not paper.is_file():
        raise SystemExit(f"paper text missing: {paper}; extract the text of arXiv 2302.12170 (PDF text with line breaks, "
                         "Tables 3-5 included) to this file or pass --paper")
    try:
        out = external_path(args.out) if args.out else receipt_path("lmx-reference-check.json")
    except ValueError as error:
        raise SystemExit(f"--out: {error}") from None

    import numpy as np
    from autoresearch_suite import llm_bench_lmx as lmx
    from transformers import pipeline

    tokenizer, classifier, _, embedder = lmx.models()
    notebook = pipeline("sentiment-analysis", model=classifier, tokenizer=tokenizer, top_k=3, device=-1)
    text = paper.read_text(encoding="utf-8", errors="replace")
    report = {"schema_version": "1.0", "checked_on": date.today().isoformat(),
              "paper_text_sha256": hashlib.sha256(paper.read_bytes()).hexdigest(),
              "module_sha256": hashlib.sha256((ROOT / "suite/_lib/autoresearch_suite/llm_bench_lmx.py").read_bytes()).hexdigest(),
              "models": lmx.MODELS, "tolerances": {"positivity": POSITIVITY_TOLERANCE, "distance": DISTANCE_TOLERANCE}, "max_distance": MAX_DISTANCE,
              "environment": {"python": platform.python_version(), **{m: __import__(m).__version__ for m in ("torch", "transformers", "sentence_transformers")}},
              "scope": "Paper rows are LMX's printed Pareto fronts (Tables 3-5). Agreement shows that the pinned models reproduce the "
                       "paper's printed positivity and distance for these sentences; it does not reproduce the paper's search or QD scores.",
              "tables": {}}
    for number, task in TABLES.items():
        rows = table_rows(text, number)
        quote = rows[0][2]
        root = lmx.embedding(quote)
        checked = []
        for distance, positivity, sentence in rows:
            local_positivity = lmx.positive_probability(sentence)
            local_distance = float(np.sqrt(((lmx.embedding(sentence) - root) ** 2).sum()))
            local_behaviour = min(local_distance, MAX_DISTANCE) / MAX_DISTANCE
            scores = {item["label"]: item["score"] for item in notebook(sentence)[0]}
            nb_distance = float(np.linalg.norm(embedder.encode([sentence])[0] - embedder.encode([quote])[0]))
            checked.append({"sentence_sha256": hashlib.sha256(sentence.encode()).hexdigest(),
                            "paper_distance": distance, "local_distance": round(local_distance, 6),
                            "local_normalised_distance": round(local_behaviour, 6),
                            "paper_positivity": positivity, "local_positivity": round(local_positivity, 6),
                            "notebook_positivity_difference": abs(scores["positive"] - local_positivity),
                            "notebook_distance_difference": abs(nb_distance - local_distance)})
        report["tables"][f"table_{number}"] = {
            "task": task, "rows": len(checked),
            "positivity_within_tolerance": sum(abs(r["paper_positivity"] - r["local_positivity"]) <= POSITIVITY_TOLERANCE for r in checked),
            "normalised_distance_within_tolerance": sum(abs(r["paper_distance"] - r["local_normalised_distance"]) <= DISTANCE_TOLERANCE for r in checked),
            "max_positivity_difference": round(max(abs(r["paper_positivity"] - r["local_positivity"]) for r in checked), 6),
            "max_normalised_distance_difference": round(max(abs(r["paper_distance"] - r["local_normalised_distance"]) for r in checked), 6),
            "max_notebook_difference": max(max(r["notebook_positivity_difference"], r["notebook_distance_difference"]) for r in checked),
            "row_checks": checked}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps({k: {x: v[x] for x in v if x != "row_checks"} for k, v in report["tables"].items()}, indent=1))


if __name__ == "__main__":
    main()
