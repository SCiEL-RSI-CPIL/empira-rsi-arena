# Physics: exact quantum circuit synthesis within published CNOT budgets

**Source collection:** [Physics research: source-based research tasks](../../papers/science-physics_research.md)\
**Task id:** `science-physics-cnot-budgeted-unitary-synthesis`\
**Protocol:** `recreated_verified`\
**Direction:** minimize\

## Task

Compile seven fixed two- and three-qubit unitaries (CNOT, iSWAP, SWAP, the Toffoli gate and three generic unitaries) into single-qubit rotations and CNOT gates without exceeding the CNOT counts known to suffice for each. The evaluator rebuilds every circuit's unitary exactly and scores the mean phase-insensitive distance to the targets.

Source-derived local benchmark; the exact objective and adaptations are specified below.

## Candidate interface

synthesize(problem) -> list of gates. problem has keys name, qubits (2 or 3), cnot_budget and the target unitary as real and imaginary (nested lists of 2**qubits rows). A gate is ['u3', qubit, theta, phi, lam] for the standard single-qubit rotation or ['cx', control, target] for a CNOT; qubit 0 is the most significant bit of the basis index. At most cnot_budget CNOTs and 400 gates per circuit. synthesize is called once per target in a sandboxed subprocess with a 150 s CPU budget for all seven targets. Score (minimize): mean over the targets of the phase-insensitive distance 1 - |tr(target^dagger U)| / 2**qubits, where the evaluator rebuilds U from the gate list; 0 means exact synthesis of every target within its budget

## State of the art

Exact synthesis within every budget is known to be possible: three CNOTs for any two-qubit unitary (two for iSWAP, one for CNOT), six for the Toffoli gate and twenty for any three-qubit unitary, so the minimum score 0 is attainable.

Reference value: proven optimum `0.0` (reviewed 2026-10-10).

- [Synthesis of quantum-logic circuits](https://doi.org/10.1109/TCAD.2005.855930)
- [Optimal quantum circuits for general two-qubit gates](https://doi.org/10.1103/PhysRevA.69.032315)
- [On the CNOT-cost of TOFFOLI gates](https://arxiv.org/abs/0803.2316)

Parameters: `{"cpu_seconds": 150}`

## Labels

Scientific contribution: `method` (tier 2); artifact `program`, domain `natural_science`, data `none`, verification `exact`, reference `published_value`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

Research area: `quantum_information`.

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **supported**.

Underlying problem: Decompose a given unitary operation into single-qubit rotations and two-qubit CNOT gates using as few CNOTs as possible.

Why it matters: CNOT count is the dominant cost of running a quantum circuit on hardware; the optimal counts for two- and three-qubit unitaries are classical results of quantum circuit synthesis, and numerically finding circuits that attain them is what compilers must do.

- [Synthesis of quantum-logic circuits](https://doi.org/10.1109/TCAD.2005.855930): Main research objective: Develop circuit-synthesis procedures for arbitrary unitaries with as few CNOT gates as possible, and prove bounds on the required number. Task connection: The task fixes the budgets this literature establishes and scores how close a candidate's circuits come to the targets.
- [Optimal quantum circuits for general two-qubit gates](https://doi.org/10.1103/PhysRevA.69.032315): Main research objective: Construct optimal CNOT-count circuits for arbitrary two-qubit unitaries. Task connection: Provides the two-qubit budgets (three CNOTs in general, fewer for special gates) used by the task.
- [On the CNOT-cost of TOFFOLI gates](https://arxiv.org/abs/0803.2316): Main research objective: Determine the minimum number of CNOT gates needed to implement the Toffoli gate. Task connection: Provides the proven budget of six CNOTs for the Toffoli target.

The cited papers' main goal is CNOT-efficient synthesis of unitaries, the problem the evaluator scores with an exact unitary distance.

## Evaluation protocol

Locally implemented evaluator with recorded reference checks; consult the task-specific tests and evidence for their exact scope. This label alone does not establish reproduction of the paper's full experiment.

- Targets and CNOT budgets are taken from the synthesis literature: one CNOT for CNOT, two for iSWAP, three for SWAP and for any two-qubit unitary, six for the Toffoli gate (proved optimal by Shende and Markov) and twenty for any three-qubit unitary (quantum Shannon decomposition). Each budget is known to be achievable, so a score of 0 exists; the task is to find such circuits numerically. The three generic unitaries are fixed matrix exponentials of Hermitian matrices defined in the evaluator (no random number generator), and the distance ignores a global phase.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `py:scipy`

## Sources

- paper: https://doi.org/10.1109/TCAD.2005.855930 (Shende, Bullock and Markov (2006), Synthesis of quantum-logic circuits: quantum Shannon decomposition with twenty CNOTs for any three-qubit unitary and the lower bound of fourteen.)
- paper: https://doi.org/10.1103/PhysRevA.69.032315 (Vatan and Williams (2004), Optimal quantum circuits for general two-qubit gates: three CNOTs suffice for every two-qubit unitary and SWAP needs three.)
- paper: https://arxiv.org/abs/0803.2316 (Shende and Markov (2009), On the CNOT-cost of TOFFOLI gates: six CNOTs are necessary and sufficient for the Toffoli gate.)

## Limitations

- Each CNOT budget is known to be achievable, so the optimum score (0) is known; the task measures whether a synthesis procedure finds exact circuits numerically within a fixed CPU budget, not whether it beats a published bound.
- The gate set is limited to U3 rotations and CNOTs on all-to-all connectivity; hardware topology, gate errors and compilation time are not modelled.
- The two-qubit and three-qubit generic targets are three fixed matrices defined in the evaluator, so the score does not average over the unitary group.
- Distances below about 1e-12 are at floating-point resolution and count as exact.
