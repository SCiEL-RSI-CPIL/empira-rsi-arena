# Software engineering: superoptimisation of the 25 Hacker's Delight kernels

**Source collection:** [Software engineering research: source-based research tasks](../../papers/science-software_engineering.md)\
**Task id:** `science-software-superoptimization-hackers-delight`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Rewrite each of 25 bit-manipulation kernels (Hacker's Delight, as used by Brahma and STOKE) into the shortest straight-line program over 32-bit operations that Z3 proves equal to the reference for every input. The score is the total instruction count; unproven programs count at the reference length.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

optimize(problem) -> list of instructions. problem = {'name': str, 'description': str, 'inputs': [names], 'reference': [[dest, op, arg, ...], ...], 'operations': {op: arity}, 'max_instructions': 64}. An instruction [dest, op, args...] defines a new name; arguments are input names, earlier destinations or integer constants in [-2^31, 2^32 - 1] (taken modulo 2^32); the value of the last instruction is the result. All values are 32-bit words: add, sub, mul, and, or, xor, not, neg wrap modulo 2^32; shl, lshr and ashr take the full word as shift amount (32 or more gives 0, or the sign fill for ashr); udiv, urem, sdiv and srem follow SMT-LIB (x udiv 0 = 2^32 - 1, x urem 0 = x, x sdiv 0 = 1 for negative x and 2^32 - 1 otherwise, x srem 0 = x); eq, ne, ult, ule, ugt, uge, slt, sle, sgt, sge give 1 or 0; select(c, a, b) gives a if c != 0 else b. Score: total number of instructions over the 25 kernels, minimised; a kernel's program counts only if it agrees with the reference on 300 test inputs and Z3 proves it equal to the reference for every input within a fixed resource limit, otherwise the reference's length counts. A structurally invalid program (unknown operation, wrong arity, undefined or reused name, more than 64 instructions) makes the evaluation invalid. optimize() is called once per kernel, all kernels in one sandboxed subprocess (no files, network or subprocesses; 120 s CPU in total, 2 GB); it should be deterministic (seed any randomness, do not read clocks).

## State of the art

No published total exists for this instruction set; the reference programs total 135 instructions and are not minimal.

Reference value: no documented value for this finite instance (reviewed 2026-10-10).

- [Synthesis of loop-free programs](https://doi.org/10.1145/1993498.1993506)

Parameters: `{"cpu_seconds": 120}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `systems`, data `public_benchmark`, verification `exact`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `systems_and_hardware`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Superoptimisation: find the shortest (or cheapest) instruction sequence that is equivalent to a given program on all inputs, a search problem at the core of compiler code generation and program synthesis.

Why it matters: Compilers rely on hand-written rewrite rules and miss short sequences for bit-level kernels; superoptimisers find provably equivalent shorter code and generate peephole rules, and the Hacker's Delight kernels are the standard benchmark of this research line.

- [Superoptimizer: a look at the smallest program](https://doi.org/10.1145/36206.36194): Main research objective: Find the shortest machine-instruction program that computes a given function. Task connection: The task asks for the shortest equivalent instruction sequence for each kernel.
- [Synthesis of loop-free programs](https://doi.org/10.1145/1993498.1993506): Main research objective: Synthesise straight-line bit-manipulation programs that are provably equivalent to a specification. Task connection: The task uses this paper's 25 Hacker's Delight kernels and proves equivalence with an SMT solver.
- [Stochastic superoptimization](https://doi.org/10.1145/2451116.2451150): Main research objective: Find shorter or faster code equivalent to a target program by stochastic search with testing and verification. Task connection: The task's references come from STOKE's versions of the kernels and candidates are verified the same way.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Reference kernels are STOKE's examples/hacker/p01-p25/fxn.cc (Apache-2.0, commit 98d8a0f0) parsed into a 32-bit IR; p12, p13, p20 and p24 are corrected to the Hacker's Delight functions named by Gulwani et al. (2011). The IR references agree with the clang-compiled (corrected) C files on 100,000 inputs and with independent Python specifications of the 25 functions; the interpreter agrees with Z3's bit-vector semantics for every operation; hence recreated_verified. The instruction set and unit costs are this dataset's choice (STOKE optimises x86-64 code for latency and Brahma synthesises from given components), so published program sizes are not directly comparable.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:z3`
- `file:data/software/superopt/problems.json`

## Sources

- paper: https://doi.org/10.1145/36206.36194 (Massalin (ASPLOS 1987): the superoptimizer, an exhaustive search for the shortest instruction sequence computing a function.)
- paper: https://doi.org/10.1145/1993498.1993506 (Gulwani, Jha, Tiwari and Venkatesan (PLDI 2011): synthesis of loop-free programs; defines the 25 Hacker's Delight benchmarks P1-P25.)
- paper: https://doi.org/10.1145/2451116.2451150 (Schkufza, Sharma and Aiken (ASPLOS 2013): stochastic superoptimization (STOKE), evaluated on the same 25 kernels.)
- repository: https://github.com/StanfordPL/stoke/tree/98d8a0f028f2daf2052bfe607dbc32ec8d55ba9e/examples/hacker (STOKE's straight-line C versions of p01-p25 (Apache-2.0); the reference programs are parsed from these files, with p12, p13, p20 and p24 corrected.)
- paper: https://doi.org/10.1145/2872362.2872387 (Phothilimthana, Thakur, Bodik and Dhurjati (ASPLOS 2016): scaling up superoptimization with an enumerative and stochastic search.)

## Limitations

- The 32-bit instruction set with unit costs is this dataset's choice; STOKE optimises x86-64 code for latency and Brahma synthesises from given component multisets, so published program sizes are not directly comparable.
- STOKE's C files for p12, p13, p20 and p24 compute other functions than the named kernels (operand swap, signed instead of unsigned shift or division, and a wrong final operand); these four were corrected, and all 25 references were checked against independent Python specifications and against the clang-compiled corrected C on 100,000 inputs.
- C's undefined cases (division by zero in p20 at x = 0, shifts by 32 or more in p19) follow SMT-LIB semantics, so equivalence is required on those inputs too.
- A program counts only if Z3 proves equivalence within a fixed resource limit; a correct program whose proof exceeds the limit counts at the reference length. Proofs for all programs tried here took well under a second.
- The reference total is 135 instructions; minimal lengths in this instruction set are unknown. Shorter verified programs exist (for example 3 instead of 10 instructions for the power-of-two test), so the reference is not the state of the art.
- The kernels are public and textbook solutions are well known; returning remembered programs is allowed by the score but the contract asks for a search procedure.
