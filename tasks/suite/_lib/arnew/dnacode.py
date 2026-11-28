"""DNA codes with constant GC-content. Records: the Gaborit and King tables of 2010 and, where one exists and passes this verifier,
a larger code published later (scripts/dnacode_newer_codes.json).

Alphabet A, C, G, T. A code is a set of distinct words of length n over it, all containing the same number w of letters from
{C, G} (constant GC-content; w is free here. The tables of King and of Gaborit and King are for w = floor(n/2): King 2003, section
"Tables"; Chee and Ling 2008, section V. A table entry is therefore a valid record for this relaxed task, but a larger code with
another w improves this task and not the table entry), with Hamming distance
at least d between any two different words. Until 2026-11-18 the verifier fixed w = n // 2, which put four thesis records out of reach:
the codes of Aboluion et al. for (n, d) = (10, 5), (14, 7), (14, 8) and (18, 9) have weights 6, 6, 8 and 10 (their codes have only even GC-weights). In the reverse-complement variant (A4GC,RC), also
d(x, rc(y)) >= d for ALL words x, y of the code, x = y included, where rc reverses the word and swaps A<->T, C<->G.
(The definitions were confirmed by exact clique search on the small optimal entries n = 4, 5, and by the codeword files of Aboluion, Smith and Perkins 2012.) Score: number of words.
"""
from __future__ import annotations

import numpy as np

from .common import CandidateError, require

MAX_WORDS = 20000
_CODE = {"A": 0, "T": 1, "C": 2, "G": 3}
_COMP = np.array([1, 0, 3, 2], dtype=np.int8)


def _encode(words, n):
    try:
        words = list(words)
    except TypeError:
        raise CandidateError("return a list of strings over ACGT") from None
    require(len(words) >= 1, "return at least one word")
    require(len(words) <= MAX_WORDS, f"at most {MAX_WORDS} words are checked")
    arr = np.empty((len(words), n), dtype=np.int8)
    for i, w in enumerate(words):
        require(isinstance(w, str) and len(w) == n and set(w) <= set("ACGT"), f"word {i} must be a string of {n} letters from ACGT")
        arr[i] = [_CODE[c] for c in w]
    return arr, words


def _min_distance(a, b, same):
    """Smallest Hamming distance between rows of a and rows of b (pairs i == j skipped when `same`)."""
    best = a.shape[1]
    step = max(1, 2_000_000 // max(1, b.shape[0] * a.shape[1]))
    for lo in range(0, a.shape[0], step):
        d = (a[lo:lo + step, None, :] != b[None, :, :]).sum(-1)
        if same:
            for k in range(d.shape[0]):
                d[k, lo + k] = a.shape[1] + 1
        best = min(best, int(d.min()))
    return best


def check(words, n, d, rc, w=None):
    """w = None: any constant GC-content (all words the same number of letters from C and G); w = k: exactly k."""
    arr, words = _encode(words, n)  # the materialised list: a generator is read once only
    require(len(set(words)) == len(arr), "words must be distinct")
    counts = (arr >= 2).sum(1)
    if w is None:
        w = int(counts[0])
        require(bool((counts == w).all()), "every word must have the same number of letters from C and G")
    else:
        require(bool((counts == w).all()), f"every word must have exactly {w} letters from C and G")
    dist = _min_distance(arr, arr, True) if len(arr) > 1 else n
    require(dist >= d, f"two words are at Hamming distance {dist} < {d}")
    if rc:
        rcs = _COMP[arr[:, ::-1]]
        rdist = _min_distance(arr, rcs, False)
        require(rdist >= d, f"a word and the reverse complement of a word are at distance {rdist} < {d}")
    return len(arr), {"size": len(arr), "n": n, "d": d, "gc_content": int(w), "reverse_complement": bool(rc), "verification": "exact_pairwise_distances"}
