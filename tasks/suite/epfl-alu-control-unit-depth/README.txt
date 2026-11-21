# EPFL alu-control-unit, LUT-6 levels

**Task id:** `epfl-alu-control-unit-depth`\
**Status:** requires_setup\
**Direction:** minimize\
**Family:** `epfl` (suite/_lib/arnew)\
**Areas:** systems and hardware, theoretical computer science

## Task

Optimise the EPFL combinational benchmark circuit alu-control-unit (the control unit of an ALU; 7 inputs, 26 outputs) with the logic-synthesis tool ABC and map it to lookup tables (LUTs) with at most 6 inputs, minimising the number of LUT levels, the largest number of LUTs on a path from an input to an output. The result must compute the same function as the original circuit.

## Candidate interface

script() -> string of ABC commands separated by `;` or newlines. The evaluator runs `read <circuit>; <script>` in ABC and writes out the network the script leaves. That network must consist only of LUTs with at most 6 inputs (in ABC a LUT mapper such as `if -K 6` produces one; library gates or a wider node are invalid), and it must be equivalent to the original circuit, which a separate ABC process checks with `cec`. The script may not contain the characters ! / \ ~, may not use commands that read or write files, change the session or start other programs (read*, write*, load*, dump*, save* and their short forms, source, alias, set, unset, history, cd, ls, pwd, quit, exit), and may not ask for LUTs wider than 6 inputs (-K above 6). Time limits: 1200 s for the script, 900 s for the equivalence check. A script that is refused, fails, runs out of time or changes the function is invalid. Score: the largest number of LUTs with two or more inputs on a path from an input to an output (buffers, inverters and constants are not counted), minimized.

## The instance

EPFL combinational benchmark 'Alu control unit' (7 inputs, 26 outputs; Amaru, Gaillardon and De Micheli, IWLS 2015, tables I and II): a simple ALU control unit producing signals for ALU operation, register destination, memory operations and jumps. The task maps it to 6-input LUTs and scores the number of LUT levels.

## Why it matters

**Problem:** Implement the EPFL benchmark circuit ctrl as a network of lookup tables with at most six inputs, with an ABC command script, minimising the number of LUT levels.

Mapping logic to k-input lookup tables fixes the area and delay of an FPGA implementation, and the EPFL suite is the community reference for logic optimisation, with best LUT-6 counts and levels lowered by new synthesis methods every few years. A better result is a smaller or shallower implementation of a standard circuit; the best-known networks mostly come from SAT-based exact synthesis, which a script of ABC commands cannot express.

Papers whose main objective is this problem (the evidence for `important_to_science`):

- [FlowMap: an optimal technology mapping algorithm for delay optimization in lookup-table based FPGA designs](https://doi.org/10.1109/43.273754): Give a mapping algorithm that is optimal in depth for K-input LUT networks. Defines depth-optimal LUT mapping for a given network structure; the task scores the LUT levels after any ABC script, so restructuring can go below the mapping depth.
- [Circuit Minimization with QBF-Based Exact Synthesis](https://doi.org/10.1609/aaai.v37i4.25524): Minimise Boolean circuits by exact resynthesis of small subcircuits encoded as QBF. The record holder's method family (exact resynthesis of subcircuits) applied to benchmark circuits; an ABC script cannot express it, so the task is the script-restricted version.

## Record

Best known LUT-6 levels: 2.

**Record validation:** reproduced. README table lists 2; the published circuit has 25 LUTs and 2 levels (EPFL convention), largest LUT 6 inputs; ABC cec (inputs and outputs matched by name): equivalent to the benchmark circuit.

## Data and requirements

random_control/ctrl.aig from github.com/lsils/benchmarks

Requires: `abc`, `file:data/epfl/ctrl.aig`. Run `python3 scripts/prepare_new_data.py --family epfl` for the data (see that script; nothing is downloaded without `--yes`).

## Limitations

- Most records come from SAT-based exact synthesis of subcircuits, which an ABC script cannot express.
- Requires the Berkeley ABC binary (set ARNEW_ABC).
- LUTs and levels are counted from the written network by the convention of the EPFL table (single-input nodes are free), not from ABC's print_stats.
- The starting program already reaches the best known value (run of 2026-11-11), so only a better value shows anything on this task.
- A record is the best published value that could be confirmed (see "Record validation" above and VALIDITY.md), not a proof that nothing better exists; read the source again before claiming an improvement.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value that is maximised), `score` (the task metric in its own direction), `valid` and `error`.
