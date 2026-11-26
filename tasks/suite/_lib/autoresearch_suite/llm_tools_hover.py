"""HoVer multi-hop document retrieval with the GEPA artifact's BM25 retriever and metric.

The candidate is a multi-hop retrieval program ``solve(claim, search, llm)``.
``search(query, k)`` is GEPA's retriever: ``bm25s`` (k1 0.9, b 0.4, English
stopwords, Snowball stemmer) over the 5.2 million 2017 Wikipedia abstracts,
each document rendered as ``"title | abstract"``; ``llm`` is the budgeted
OpenRouter client.  As in GEPA's HoverMultiHop program (3 hops retrieving
7, 7 and 10 documents) a claim may use at most 3 search calls of at most 10
documents and at most 24 documents in total.  A claim counts as solved when
the normalised titles of all its gold supporting documents are among the
titles of the documents that ``search`` returned for it (GEPA's
``discrete_retrieval_eval``); the score is the solved fraction on a fixed
seeded sample of GEPA's HoVer test split.  Gold titles stay in this process;
the candidate runs in the sandbox child.
"""
from __future__ import annotations

import json
import os
import re
import string
import unicodedata

import numpy as np

from . import llm_bench_core as core
from . import llm_tools_common as common

MAX_CALLS, MAX_K, MAX_DOCS = 3, 10, 24
_RETRIEVER = {}


def normalize_text(text):
    """dspy.evaluate.normalize_text: NFD, lower case, no punctuation, no articles, collapsed spaces."""
    text = unicodedata.normalize("NFD", str(text)).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def doc_title(document):
    return document.split(" | ")[0]


def solved(gold_titles, documents):
    found = {normalize_text(doc_title(doc)) for doc in documents}
    gold = {normalize_text(title) for title in gold_titles}
    return gold.issubset(found), len(gold & found), len(gold)


class Corpus:
    """Line-addressed document store: docs.jsonl (one JSON string per line) plus int64 line offsets."""

    def __init__(self, directory):
        self.path = directory / "docs.jsonl"
        self.offsets = np.load(directory / "doc_offsets.npy", mmap_mode="r")

    def __getitem__(self, index):
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[index]))
            return json.loads(handle.readline().decode("utf-8"))

    def __len__(self):
        return len(self.offsets)


def retriever():
    """The memory-mapped bm25s index, the stemmer and the corpus (loaded once per process)."""
    if not _RETRIEVER:
        # bm25s selects the top k with jax.lax.top_k when jax is installed (as in the GEPA artifact's environment); keep XLA on few threads.
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
        import bm25s
        import Stemmer
        directory = common.data_file("hover/index")
        _RETRIEVER.update(bm25s=bm25s, stemmer=Stemmer.Stemmer("english"),
                          index=bm25s.BM25.load(str(directory / "bm25s"), mmap=True), corpus=Corpus(directory))
    return _RETRIEVER


def bm25_search(query, k):
    """GEPA's search(): top-k documents, deduplicated by text in score order."""
    r = retriever()
    tokens = r["bm25s"].tokenize(query, stopwords="en", stemmer=r["stemmer"], show_progress=False)
    try:
        results, scores = r["index"].retrieve(tokens, k=k, n_threads=1, show_progress=False)
    except (ValueError, KeyError, IndexError):  # e.g. a query made only of stopwords; GEPA's program would fail the claim
        return []
    run = {}
    for doc, score in zip(results[0], scores[0]):
        run.setdefault(r["corpus"][int(doc)], float(score))
    return list(run)[:k]


class SearchBudget:
    """Per-claim proxy handler that enforces the hop budget and records every returned document."""

    def __init__(self, search=bm25_search):
        self.search, self.calls, self.requested, self.documents = search, 0, 0, []

    def __call__(self, query, k=7):
        if not isinstance(query, str) or not query.strip():
            raise common.ProxyError("search(query, k): query must be a non-empty string")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            raise common.ProxyError(f"search(query, k): k must be an integer from 1 to {MAX_K}")
        if self.calls >= MAX_CALLS or self.requested + k > MAX_DOCS:
            raise common.ProxyError(f"search budget exhausted (at most {MAX_CALLS} calls and {MAX_DOCS} documents per claim)")
        self.calls += 1
        self.requested += k
        found = self.search(query[:2000], k)
        self.documents.extend(found)
        return found


def claims(P):
    rows = common.read_json("hover/claims.json")["test"]
    return core.fixed_sample(rows, core.sample_size(P, 30), P.get("seed", 0), key="uid")


def evaluate(module, P, search=bm25_search, client=None):
    source = common.candidate_source(module, "solve")
    client = client or core.task_llm(P)
    records, total = [], 0.0
    for row in claims(P):
        budget = SearchBudget(search)
        ok, _, failure = common.call_candidate(source, "solve", [row["claim"]], proxies=["search", "llm"],
                                               handlers={"search": budget, "llm": common.llm_handler(client)},
                                               cpu_seconds=int(P.get("cpu_seconds", 60)), wall_seconds=int(P.get("wall_seconds", 600)))
        hit, found, gold = solved(row["gold_titles"], budget.documents) if ok else (False, 0, len(row["gold_titles"]))
        total += float(hit)
        record = {"id": row["uid"], "score": float(hit), "gold_found": found, "gold": gold, "search_calls": budget.calls,
                  "documents": len(budget.documents)}
        if not ok:
            record["crash"] = f"{failure['kind']}: {failure['message'][:200]}"
        records.append(record)
    n = len(records)
    return total / n, {"items": n, "crashes": sum("crash" in r for r in records), "mean_gold_recall": round(sum(r["gold_found"] / r["gold"] for r in records) / n, 6),
                       "usage": client.usage(), "per_item": records}


# ---------------------------------------------------------------------------
# Index construction (run by scripts/prepare_llm_tools.py with PYTHONHASHSEED=0)
# ---------------------------------------------------------------------------

def build_index(abstracts_jsonl, out_dir):
    """GEPA's corpus strings and bm25s index; documents are streamed so the raw corpus is never held in memory."""
    import bm25s
    import Stemmer
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets, position = [], 0
    with open(abstracts_jsonl, encoding="utf-8") as source, (out_dir / "docs.jsonl").open("wb") as docs:
        for line in source:
            row = json.loads(line)
            data = (json.dumps(f"{row['title']} | {' '.join(row['text'])}", ensure_ascii=False) + "\n").encode("utf-8")
            offsets.append(position)
            docs.write(data)
            position += len(data)
    np.save(out_dir / "doc_offsets.npy", np.array(offsets, dtype=np.int64))
    corpus = Corpus(out_dir)

    def texts():
        with corpus.path.open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)

    tokens = bm25s.tokenize(texts(), stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=bool(os.environ.get("ARTASK_PROGRESS")))
    index = bm25s.BM25(k1=0.9, b=0.4)
    index.index(tokens, show_progress=bool(os.environ.get("ARTASK_PROGRESS")))
    index.save(str(out_dir / "bm25s"))
    return len(offsets)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    print(build_index(sys.argv[1], Path(sys.argv[2])))
