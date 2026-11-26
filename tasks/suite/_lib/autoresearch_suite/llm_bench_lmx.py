"""LMX sentence-sentiment task: an archive of sentences scored like LMX's MAP-Elites demo.

Contract: ``solve(question, llm) -> list[str]``; ``question`` is the target quote.
Each returned sentence gets a fitness, the probability of the 'positive' class
of cardiffnlp/twitter-roberta-base-sentiment-latest, and a niche, the
Euclidean distance between its all-MiniLM-L6-v2 embedding and the quote's,
clipped to [0, max_distance], scaled to [0, 1] and cut into ``niches`` equal
bins (scripts/lmx_sentiment_demo.ipynb of jal278/lmx at
b8fec5dd16451cf7afb6e02ac621b71d5bdfee7f: 30 bins, distance 1.25). The score
is the QD score, the sum over occupied niches of the best fitness (at most 30).
Both models run locally on the CPU from pinned revisions; sentences are
scored one at a time, as in the notebook.
"""
from __future__ import annotations

import math

from . import llm_bench_core as core
from .llm_bench_sources import MODELS
from .runtime import CandidateError

_LOADED = None


def models():
    """(tokenizer, classifier, positive label index, sentence embedder), loaded offline once."""
    global _LOADED
    if _LOADED is None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise CandidateError(f"torch, transformers and sentence-transformers are required: {error}") from None
        torch.set_num_threads(1)  # fixed thread count keeps the floating-point results reproducible
        sentiment, embedding = MODELS["sentiment"], MODELS["embedding"]
        try:
            tokenizer = AutoTokenizer.from_pretrained(sentiment["repo"], revision=sentiment["revision"], local_files_only=True)
            classifier = AutoModelForSequenceClassification.from_pretrained(sentiment["repo"], revision=sentiment["revision"], local_files_only=True).eval()
            embedder = SentenceTransformer(embedding["repo"], revision=embedding["revision"], local_files_only=True, device="cpu")
        except OSError as error:
            raise CandidateError(f"pinned LMX models missing; run `python scripts/prepare_llm_bench.py --models`: {error}") from None
        _LOADED = (tokenizer, classifier, classifier.config.label2id["positive"], embedder)
    return _LOADED


def positive_probability(sentence):
    import torch
    tokenizer, classifier, positive, _ = models()
    encoded = tokenizer(sentence, return_tensors="pt")
    limit = classifier.config.max_position_embeddings - 2  # RoBERTa keeps two positions for its padding offset
    if encoded["input_ids"].shape[1] > limit:  # 400 characters of emoji or CJK text can be more tokens than the classifier reads
        raise CandidateError(f"a sentence is {encoded['input_ids'].shape[1]} tokens long for the sentiment classifier, which reads at most {limit}")
    with torch.no_grad():
        logits = classifier(**encoded).logits[0]
    return float(torch.softmax(logits, dim=-1)[positive])


def embedding(sentence):
    return models()[3].encode([sentence])[0]


def niche(distance, niches, max_distance):
    behaviour = min(max(distance, 0.0), max_distance) / max_distance
    return min(niches - 1, int(behaviour * niches))


def check_sentences(sentences, max_sentences):
    if not isinstance(sentences, (list, tuple)) or not sentences:
        raise CandidateError("solve must return a non-empty list of sentences")
    if len(sentences) > max_sentences:
        raise CandidateError(f"at most {max_sentences} sentences are scored")
    for sentence in sentences:
        if not isinstance(sentence, str):
            raise CandidateError(f"every sentence must be a str, got {type(sentence).__name__}")
        text = sentence.strip()
        if not 5 <= len(text) <= 400 or "\n" in text or "\r" in text:
            raise CandidateError("each sentence must be one line of 5 to 400 characters")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            raise CandidateError("each sentence must be text that can be encoded (no unpaired surrogates)") from None
    return [sentence.strip() for sentence in sentences]


def qd_score(quote, sentences, niches=30, max_distance=1.25):
    import numpy as np
    root = embedding(quote)
    grid = {}
    for sentence in dict.fromkeys(sentences):
        fitness = positive_probability(sentence)
        cell = niche(float(np.sqrt(((embedding(sentence) - root) ** 2).sum())), niches, max_distance)
        if cell not in grid or grid[cell][0] < fitness:
            grid[cell] = (fitness, sentence)
    score = sum(fitness for fitness, _ in grid.values())
    if not math.isfinite(score):
        raise ValueError("non-finite QD score")
    archive = {str(cell): {"fitness": round(fitness, 6), "sentence": sentence} for cell, (fitness, sentence) in sorted(grid.items())}
    return score, {"niches_filled": len(grid), "best_fitness": round(max(f for f, _ in grid.values()), 6), "archive": archive}


def evaluate(solve, P):
    core.require_signature(solve, 2, "solve(question, llm)")
    client = core.task_llm(P)  # optional for the candidate; the API key is read only on the first call
    sentences = check_sentences(solve(P["quote"], client), int(P.get("max_sentences", 300)))
    score, details = qd_score(P["quote"], sentences, int(P.get("niches", 30)), float(P.get("max_distance", 1.25)))
    details.update(sentences=len(sentences), usage=client.usage())
    return {"score": score, "details": details}
