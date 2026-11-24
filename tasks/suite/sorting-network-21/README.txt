# Sorting network on 21 inputs

**Task id:** `sorting-network-21`\
**Status:** ready\
**Direction:** minimize\
**Family:** `sortnet` (suite/_lib/arnew)\
**Areas:** theoretical computer science, systems and hardware

## Task

Construct a sorting network on 21 wires with as few comparators as possible. A comparator (i, j) with i < j compares the values on wires i and j and puts the smaller one on wire i and the larger one on wire j; the comparators are applied in the given order. The network must sort every input, that is, leave the values in non-decreasing order along wires 0 to 20.

## Candidate interface

construct() -> list of comparators (i, j) with 0 <= i < j < 21, in the order in which they are applied (at least one). The network is verified on all 2^21 inputs of zeros and ones, which is equivalent to sorting every input. Score: number of comparators, minimized (the depth is not scored).

## The instance

The size-minimisation problem for sorting networks on 21 wires, from Dobbelaere's list of smallest known sorting networks.

## Why it matters

**Problem:** Construct a sorting network on 21 wires with as few comparators as possible.

Sorting networks are oblivious sorting procedures used in hardware, SIMD and GPU sorting, and the minimum number of comparators is known only for up to 12 inputs; for 13 and more inputs the best known sizes are upper bounds that have been lowered by evolutionary and SAT searches, so a smaller network is a new upper bound.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Using Symmetry and Evolutionary Search to Minimize Sorting Networks](https://jmlr.org/papers/v14/valsalam13a.html): Minimise the number of comparators of sorting networks with a method combining problem symmetry and evolutionary search (SENSO). The paper treats the same size-minimisation problem up to 32 inputs: its Table 2 lists the then best known sizes for 17 to 32 inputs and its abstract reports new minimal networks for 17 to 22 inputs and ties for larger n. The task's 21-input records (Dobbelaere's list) are lower than the paper's for 19 to 29 inputs and equal for 30 to 32; the paper is not the source of the current record.

## Record

Best known size 99 (Dobbelaere's list), proven lower bound 80 (the list withdrew its stronger bounds for 13 or more inputs on 2026-11-20 because of an error in a proof of Van Voorhis 1972; 80 is 39, the proven optimum for 12 inputs, plus ceil(log2 k) for each further input k).

**Record validation:** reproduced. smallest listed network for n=21 has 99 comparators, depth 15; sorts every input (exact check by our verifier).

## Data and requirements

No data files.

Requires: `py:numpy`.

## Limitations

- Exact 0-1 verification over all 2^21 inputs, in blocks of 2^24.
- SorterHunter and symmetry-based searches have worked on these sizes for years.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
