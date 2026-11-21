# Sorting network on 31 inputs

**Task id:** `sorting-network-31`\
**Status:** ready\
**Direction:** minimize\
**Family:** `sortnet` (suite/_lib/arnew)\
**Areas:** theoretical computer science, systems and hardware

## Task

Construct a sorting network on 31 wires with as few comparators as possible. A comparator (i, j) with i < j compares the values on wires i and j and puts the smaller one on wire i and the larger one on wire j; the comparators are applied in the given order. The network must sort every input, that is, leave the values in non-decreasing order along wires 0 to 30.

## Candidate interface

construct() -> list of comparators (i, j) with 0 <= i < j < 31, in the order in which they are applied (at least one). The network is verified on all 2^31 inputs of zeros and ones, which is equivalent to sorting every input. Score: number of comparators, minimized (the depth is not scored).

## The instance

The size-minimisation problem for sorting networks on 31 wires, from Dobbelaere's list of smallest known sorting networks.

## Why it matters

**Problem:** Construct a sorting network on 31 wires with as few comparators as possible.

Sorting networks are oblivious sorting procedures used in hardware, SIMD and GPU sorting, and the minimum number of comparators is known only for up to 12 inputs; for 13 and more inputs the best known sizes are upper bounds that have been lowered by evolutionary and SAT searches, so a smaller network is a new upper bound.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [Using Symmetry and Evolutionary Search to Minimize Sorting Networks](https://jmlr.org/papers/v14/valsalam13a.html): Minimise the number of comparators of sorting networks with a method combining problem symmetry and evolutionary search (SENSO). Its Table 2 gives the same size for 31 inputs as the task's reference (a merge of smaller networks), so the reference is not a search result of the paper; the paper addresses the same minimisation problem up to 32 inputs.

## Record

Best known size 180 (Dobbelaere's list), proven lower bound 141.

**Record validation:** reproduced. smallest listed network for n=31 has 180 comparators, depth 14; sorts every input (exact check by our verifier).

## Data and requirements

No data files.

Requires: `py:numpy`.

## Limitations

- Exact 0-1 verification over all 2^31 inputs, in blocks of 2^24; one evaluation takes a few seconds for n = 30 and up to about half a minute for n = 32.
- SorterHunter and symmetry-based searches have worked on these sizes for years.
- For this n the list gives the size of a Batcher odd-even merge of the best known smaller networks, not a dedicated search result, so the record is easier to reach than for n <= 29.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
