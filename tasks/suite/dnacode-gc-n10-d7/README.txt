# DNA code A4GC(10,7)

**Task id:** `dnacode-gc-n10-d7`\
**Status:** ready\
**Direction:** maximize\
**Family:** `dnacode` (suite/_lib/arnew)\
**Areas:** theoretical computer science, biology, mathematics

## Task

Find as many DNA words of length 10 as possible, that is, strings over the letters A, C, G, T, such that all words have the same number of letters in {C, G} (that number is free) and any two different words differ in at least 7 positions.

## Candidate interface

construct() -> list of distinct strings of length 10 over the capital letters A, C, G, T (at least one word and at most 20000; a longer list is invalid). Score: number of words, maximized.

## The instance

Entry A4GC(10,7) of the Gaborit-King table of lower bounds for DNA codes of length 10, GC-content 5 and minimum Hamming distance 7.

## Why it matters

**Problem:** Find a DNA code of length 10 (words over A, C, G, T with exactly floor(10/2) letters from C and G, pairwise Hamming distance at least 7) with as many words as possible.

Such word sets serve as barcodes, microarray probes and tags in DNA computing and storage, where words must not hybridise with each other; a constant GC-content gives similar melting temperatures and a large Hamming distance makes mismatched pairing unlikely. The maximum size is unknown for almost all parameters and the pinned table has been the reference for lower bounds since 2010, so a larger code is a new lower bound.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Bounds for DNA Codes with Constant GC-Content](https://doi.org/10.37236/1726): Derive upper and lower bounds on the maximum size of DNA codes with constant GC-content and minimum Hamming distance, with and without the reverse-complement constraint, and construct larger codes than previously published. The paper defines the quantity the task maximises for length 10 and distance 7 (GC-content floor(10/2)); its lexicographic constructions and Johnson-type bounds underlie the table the task starts from. The task covers the plain constant-GC variant.
- [Linear constructions for DNA codes](https://doi.org/10.1016/j.tcs.2004.11.004): Construct DNA codes with constant GC-content from linear and related codes and tabulate the resulting lower bounds. Its constructions give many entries of the table for length 10 and distance 7 (superscripts c, l and m on the pinned page); the task asks to beat the tabulated size. Covers the plain constant-GC variant.
- [Improved Lower Bounds for Constant GC-Content DNA Codes](https://doi.org/10.1109/TIT.2007.911167): Improve the lower bounds on the size of constant GC-content DNA codes, with and without the reverse-complement constraint, by stochastic local search, and prove some entries optimal by clique computations. Supplies table entries marked a, b and colon-optimal on the pinned page; the task asks for a larger code than the tabulated one for length 10, distance 7.
- [Construction of Constant GC-Content DNA Codes via a Variable Neighbourhood Search Algorithm](https://doi.org/10.1007/s10852-008-9087-8): Construct large constant GC-content DNA codes with a variable neighbourhood search algorithm. Supplies the many table entries marked v on the pinned page; the task asks for a larger code for length 10, distance 7.

## Record

Lower bound 44, not proven optimal: a code of that size was published on 2026-10-14 (https://github.com/owhs/DNA_GC_codes_2026) and passes this task's verifier. The table of Gaborit and King, last updated 23 June 2010, has 34 (construction 'v'). The tables list the maximum over the GC-content w (the same number of letters from C and G in every word); this task lets the code choose any one w, so a table value is reachable whichever w it uses (until 2026-11-18 the task fixed w = floor(n/2), which put four records of Aboluion et al. out of reach).

**Record validation:** reproduced. code file codes/GC_n10_d7.txt of https://github.com/owhs/DNA_GC_codes_2026 (commit e3b6ed8a4d1b, dated 2026-10-14) has 44 words and passes our verifier; the table of Gaborit and King (pinned page, dated 23 June 2010) has 34 (construction 'v'). Note: The code comes from a repository, not from a paper: found by an automated search, credited to its owner, and replayed here word by word. Not proven optimal; the tables of Montemanni, Smith and Koul 2014 show no larger value.

## Data and requirements

No data files.

Requires: `py:numpy`.

## Limitations

- The record is a code from a 2026 repository, found by an automated search and not published in a paper; it was replayed word by word through this task's verifier. Papers published after 2014 were not searched systematically.
- Only open entries with a bound of at most 2000 words in the table are tasks, so that the pairwise check stays cheap.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
