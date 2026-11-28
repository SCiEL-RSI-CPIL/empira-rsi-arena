# Software engineering: optimal register coalescing on the 22 largest challenge graphs

**Source collection:** [Software engineering research: source-based research tasks](../../papers/science-software_engineering.md)\
**Task id:** `science-software-coalescing-large`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Colour the 22 interference graphs with 3,000 to 28,640 nodes from the Appel-George Optimal Coalescing Challenge with 6 registers so that as much move (affinity) weight as possible joins nodes of the same colour. The score is the mean share of affinity weight left uncoalesced; lower is better.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

coalesce(problem) -> list of int. problem = {'name': str, 'nodes': N, 'registers': K (6 in every instance), 'interference': [[i, j], ...], 'affinities': [[i, j, w], ...]} with nodes numbered 0..N-1: nodes 0..K-1 are machine registers and node i < K must get colour i; every other node interferes with fewer than K nodes, so a legal colouring always exists; affinities are move-related pairs with positive integer weights. Return N Python ints in 0..K-1 such that no interference pair shares a colour. Cost of an instance: total weight of the affinity pairs whose two nodes get different colours. Score: mean over the task's instances of cost divided by the instance's total affinity weight (the share of move weight not coalesced), minimised; 0 means every move is coalesced. An illegal colouring of any instance makes the evaluation invalid. coalesce() is called once per instance, all instances in one sandboxed subprocess (no files, network or subprocesses; 120 s CPU in total, 3 GB); it should be deterministic (seed any randomness, do not read clocks).

## State of the art

Mean uncoalesced share of the Grund-Hack ILP solutions (19 of 22 proven optimal), recomputed here. The published iterated-register-coalescing and optimistic-coalescing solution sets score 0.0268 and 0.0155.

Reference value: best known value `0.01047752171243062` (reviewed 2026-10-10).

- [A Fast Cutting-Plane Algorithm for Optimal Coalescing](https://doi.org/10.1007/978-3-540-71229-9_8)
- [Optimal Coalescing Challenge: Solutions (Grund)](https://www.cs.princeton.edu/~appel/coalesce/grund)

Parameters: `{"group": "large", "cpu_seconds": 120}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `systems`, data `public_benchmark`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `systems_and_hardware`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Register coalescing in compilers: colour an interference graph with the available registers so that the endpoints of as many weighted copy instructions as possible share a register, which removes those copies.

Why it matters: Register allocation is a core compiler phase; coalescing removes move instructions in hot code, and finding the optimal coalescing is NP-hard, so heuristics and exact methods are compared on the published challenge graphs.

- [Iterated register coalescing](https://doi.org/10.1145/229542.229546): Main research objective: Remove as many move instructions as possible during graph-colouring register allocation without introducing spills. Task connection: The task optimises the same objective (coalesced move weight) under the same colouring constraints.
- [Optimal spilling for CISC machines with few registers](https://doi.org/10.1145/378795.378854): Main research objective: Allocate registers optimally for machines with few registers by separating optimal spilling from coalescing. Task connection: The challenge graphs are the coalescing problems left after this optimal spilling; the task solves them.
- [A Fast Cutting-Plane Algorithm for Optimal Coalescing](https://doi.org/10.1007/978-3-540-71229-9_8): Main research objective: Solve the register coalescing problem to optimality fast enough for compilers. Task connection: Their optimal solutions of the challenge graphs are the task's reference values.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Instances are the 474 graphs of the Appel-George Optimal Coalescing Challenge, split by size (fewer than 3000 nodes, 452 instances; at least 3000 nodes, 22 instances). Duplicate pairs are merged and self-affinities dropped, which leaves every colouring's cost unchanged. The evaluator's cost function reproduces the challenge checker's published costs of all 474 Grund-Hack ILP solutions, and the published iterated-register-coalescing and optimistic-coalescing solution sets validate; hence recreated_verified. The mean uncoalesced weight share is this dataset's aggregate; the challenge itself scores instances separately. Published solvers had no CPU limit comparable to the sandbox's.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `file:data/software/coalesce/large.json`

## Sources

- benchmark: https://www.cs.princeton.edu/~appel/coalesce/ (Appel and George, Optimal Coalescing Challenge (2000): 474 interference graphs with affinities, problem and solution format, checker and published solution sets.)
- paper: https://doi.org/10.1145/378795.378854 (Appel and George (PLDI 2001): optimal spilling by integer programming followed by coalescing; source of the challenge graphs.)
- paper: https://doi.org/10.1145/229542.229546 (George and Appel (TOPLAS 1996): iterated register coalescing, the heuristic behind the published greedy solution set.)
- paper: https://doi.org/10.1007/978-3-540-71229-9_8 (Grund and Hack (CC 2007): cutting-plane ILP for optimal coalescing; their solutions of all 474 challenge instances are on the challenge page.)
- paper: https://doi.org/10.1145/1011508.1011512 (Park and Moon (TOPLAS 2004): optimistic register coalescing, the basis of the published optimistic solution set.)

## Limitations

- Best-known costs are the published Grund-Hack ILP solutions; CPLEX proved 19 of the 22 optimal, while instances 139, 304 and 387 stopped after five minutes. Blazy and Robillard (LCTES 2009) report optimal solutions for these three, which were not checked here.
- The score averages per-instance shares of uncoalesced affinity weight; the challenge itself evaluates instances separately, so totals in the literature are not the same number.
- All graphs come from one compiler (SML/NJ) with six registers after optimal spilling; methods tuned on them may not transfer to other register files or allocators.
- The instances are public; a candidate could embed precomputed colourings, which the contract excludes.
- The candidate gets 120 s of CPU time in total in a Python sandbox; published solvers had no comparable limit.
